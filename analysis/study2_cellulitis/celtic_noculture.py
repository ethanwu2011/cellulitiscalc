"""CELTIC: every quantity the manuscript reports, computed without any culture result.

CELTIC is defined WITHOUT any microbiology input at every tier. This script re-derives, from the
saved feature parquet + the local NLP laterality extractions, all of:
  tiers (Bedside 5 / Rapid 10 / Standard / Comprehensive, no culture anywhere) with bootstrap CIs,
  the Bedside odds ratios + integer points, the three-setting head-to-head vs the original ALT-70
  (MIMIC inpatient, MIMIC-IV-ED, eICU-CRD) with paired bootstrap + DeLong, bands, operating points,
  confusion counts, calibration, subgroups, stewardship simulation, temporal validation, label-noise
  sensitivity, per-mimic-category discrimination, NRI, AUPRC, feature importance, and Table 1 with
  counts and IQRs.

DUA RULE: reads gitignored row-level features locally; writes AGGREGATE metrics only.
Run from analysis/:  PYTHONPATH=. python study2_cellulitis/celtic_noculture.py
"""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from shared.mimic_io import from_config
from study2_cellulitis.delong import delong_roc_test
from study2_cellulitis.external_eicu_v2 import SHARED, eicu_cohort_features
from study2_cellulitis.per_category import CATEGORY, condition_hadms

FEATURES = "data/cohorts/study2_features.parquet"
NLP = "data/nlp/cohort_extractions.jsonl"
ED = str(Path("~/mimic-iv-ed-2.2/ed").expanduser())
H = str(Path("~/mimiciv-3.1/hosp").expanduser())
OUT = Path("study2_cellulitis/celtic_noculture_metrics.json")
OUT_T1 = Path("study2_cellulitis/celtic_table1.json")
SEED = 20260707
NBOOT = 2000
RAFF_PREV = 0.70
VITAL_PREFIX = ("hr_", "temp_", "rr_", "spo2", "sysbp")
META = {"subject_id", "hadm_id", "admittime", "gender", "label", "split"}
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- helpers
def strat_idx(rng, y):
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    return np.concatenate([rng.choice(pos, pos.size), rng.choice(neg, neg.size)])


def paired_boot(y, scores: dict, pairs: list[tuple[str, str]], nboot=NBOOT, seed=SEED):
    """Stratified bootstrap; every score is evaluated on the SAME resample so differences are paired."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    S = {k: np.asarray(v, float) for k, v in scores.items()}
    keys = list(S)
    A = {k: np.empty(nboot) for k in keys}
    for b in range(nboot):
        i = strat_idx(rng, y)
        yi = y[i]
        for k in keys:
            A[k][b] = roc_auc_score(yi, S[k][i])
    out = {}
    for k in keys:
        out[k] = {"auroc": r3(roc_auc_score(y, S[k])), "lo": r3(np.percentile(A[k], 2.5)),
                  "hi": r3(np.percentile(A[k], 97.5)), "n": int(len(y)), "n_pos": int(y.sum())}
    diffs = {}
    for a, b_ in pairs:
        d = A[a] - A[b_]
        p = 2 * min((d <= 0).mean(), (d >= 0).mean())
        diffs[f"{a}_minus_{b_}"] = {"delta": r3(roc_auc_score(y, S[a]) - roc_auc_score(y, S[b_])),
                                    "lo": r3(np.percentile(d, 2.5)), "hi": r3(np.percentile(d, 97.5)),
                                    "p_boot": round(float(max(p, 1 / nboot)), 4),
                                    "delong": delong_roc_test(y, S[a], S[b_])}
    return out, diffs


def r3(x):
    return round(float(x), 3)


def r1(x):
    return round(float(x), 1)


def boot_auprc(y, s, seed=SEED, nboot=NBOOT):
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    s = np.asarray(s, float)
    a = [average_precision_score(y[i], s[i]) for i in (strat_idx(rng, y) for _ in range(nboot))]
    return {"auprc": r3(average_precision_score(y, s)), "lo": r3(np.percentile(a, 2.5)), "hi": r3(np.percentile(a, 97.5))}


def ppv_npv_at(sens, spec, prev):
    den1 = sens * prev + (1 - spec) * (1 - prev)
    den0 = spec * (1 - prev) + (1 - sens) * prev
    return (sens * prev / den1 if den1 else np.nan), (spec * (1 - prev) / den0 if den0 else np.nan)


def opchar(flag, y, prev_std=None):
    flag = np.asarray(flag, bool)
    y = np.asarray(y)
    tp = int((flag & (y == 1)).sum()); fp = int((flag & (y == 0)).sum())
    fn = int((~flag & (y == 1)).sum()); tn = int((~flag & (y == 0)).sum())
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    out = {"pct_flagged": r1(100 * flag.mean()), "sens": r1(100 * sens), "spec": r1(100 * spec),
           "ppv": r1(100 * ppv) if tp + fp else None, "npv": r1(100 * npv) if tn + fn else None,
           "tp": tp, "fn": fn, "fp": fp, "tn": tn}
    if prev_std is not None:
        p70, n70 = ppv_npv_at(sens, spec, prev_std)
        out["ppv_at70prev"] = r1(100 * p70); out["npv_at70prev"] = r1(100 * n70)
    return out


def boot_op(score, y, t, seed=SEED, nboot=NBOOT, prev_std=RAFF_PREV):
    rng = np.random.default_rng(seed)
    score = np.asarray(score, float); y = np.asarray(y)
    S = []
    for _ in range(nboot):
        i = strat_idx(rng, y); yb = y[i]; fb = score[i] >= t
        tp = (fb & (yb == 1)).sum(); fp = (fb & (yb == 0)).sum(); fn = (~fb & (yb == 1)).sum(); tn = (~fb & (yb == 0)).sum()
        sens = tp / (tp + fn) if tp + fn else 0.0; spec = tn / (tn + fp) if tn + fp else 0.0
        ppv = tp / (tp + fp) if tp + fp else np.nan; npv = tn / (tn + fn) if tn + fn else np.nan
        p70, n70 = ppv_npv_at(sens, spec, prev_std)
        S.append((sens, spec, ppv, npv, p70, n70))
    S = np.array(S) * 100
    lab = ["sens", "spec", "ppv", "npv", "ppv_at70prev", "npv_at70prev"]
    return {lab[i] + "_ci": [r1(np.nanpercentile(S[:, i], 2.5)), r1(np.nanpercentile(S[:, i], 97.5))] for i in range(6)}


def op_table(score, y, thresholds, prev_std=RAFF_PREV):
    return [{"threshold": float(t) if not float(t).is_integer() else int(t), **opchar(np.asarray(score) >= t, y, prev_std)}
            for t in thresholds]


def roc_points(y, s, max_pts=160):
    fpr, tpr, thr = roc_curve(y, s)
    spec = 100 * (1 - fpr); sens = 100 * tpr
    if len(spec) > max_pts:
        keep = np.unique(np.concatenate([[0, len(spec) - 1], np.linspace(0, len(spec) - 1, max_pts).astype(int)]))
        spec, sens = spec[keep], sens[keep]
    return {"spec": [r1(v) for v in spec], "sens": [r1(v) for v in sens]}


def calib(p, y):
    p = np.asarray(p, float); y = np.asarray(y)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, 11)))
    b = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    bins = []
    for i in range(len(edges) - 1):
        m = b == i
        if m.sum() == 0:
            continue
        obs = float(y[m].mean()); n = int(m.sum())
        se = float(np.sqrt(obs * (1 - obs) / n)) if 0 < obs < 1 else 0.0
        bins.append({"pred_mean": round(float(p[m].mean()), 4), "obs": round(obs, 4),
                     "obs_lo": round(max(0, obs - 1.96 * se), 4), "obs_hi": round(min(1, obs + 1.96 * se), 4), "n": n})
    eps = 1e-6; pc = np.clip(p, eps, 1 - eps); lp = np.log(pc / (1 - pc))
    rc = LogisticRegression(max_iter=5000).fit(lp.reshape(-1, 1), y)
    return {"brier": round(float(brier_score_loss(y, p)), 4), "slope": r3(rc.coef_[0][0]),
            "intercept": r3(rc.intercept_[0]), "deciles": bins}


def pipe_lr(seed=SEED):
    return Pipeline([("i", SimpleImputer(strategy="median")), ("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=5000, random_state=seed))])


def pipe_lr_ind(seed=SEED):
    return Pipeline([("i", SimpleImputer(strategy="median", add_indicator=True)), ("s", StandardScaler()),
                     ("c", LogisticRegression(max_iter=5000, random_state=seed))])


def xgb(seed=SEED):
    return XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=5, eval_metric="logloss", random_state=seed, n_jobs=6)


def logit_binary(X, y):
    """Unpenalized logistic regression on binary inputs: coefficients, Wald 95% CIs, ORs, integer points."""
    Xc = sm.add_constant(X.astype(float), has_constant="add")
    fit = sm.Logit(np.asarray(y, float), Xc).fit(disp=0, maxiter=200)
    ci = fit.conf_int()
    terms = {}
    mx = max(abs(float(fit.params[c])) for c in X.columns)
    for c in X.columns:
        b = float(fit.params[c])
        terms[c] = {"coef": round(b, 4), "or": round(float(np.exp(b)), 2),
                    "or_lo": round(float(np.exp(ci.loc[c, 0])), 2), "or_hi": round(float(np.exp(ci.loc[c, 1])), 2),
                    "p": float(fit.pvalues[c]), "points": int(np.round(b / mx * 3))}
    return {"intercept": round(float(fit.params["const"]), 4), "terms": terms, "n_fit": int(len(y))}


def predict_binary(model, X):
    z = model["intercept"] + sum(model["terms"][c]["coef"] * X[c].to_numpy(float) for c in model["terms"])
    return 1 / (1 + np.exp(-z))


def points_score(model, X):
    return sum(model["terms"][c]["points"] * X[c].to_numpy(int) for c in model["terms"])


def bands_prob(p, y, t_lo, t_hi):
    p = np.asarray(p); y = np.asarray(y)
    ro = p < t_lo; ri = p >= t_hi; ind = ~ro & ~ri
    def frac(m, v): return r1(100 * float((y[m] == v).mean())) if m.sum() else None
    def lr(m):
        p1 = m[y == 1].mean(); p0 = m[y == 0].mean()
        return round(float(p1 / p0), 2) if p0 > 0 else None
    return {"rule_out": {"n": int(ro.sum()), "pct_of_cohort": r1(100 * ro.mean()), "pct_pseudocellulitis": frac(ro, 0),
                         "pct_cellulitis": frac(ro, 1), "LR": lr(ro)},
            "indeterminate": {"n": int(ind.sum()), "pct_of_cohort": r1(100 * ind.mean()), "pct_cellulitis": frac(ind, 1)},
            "rule_in": {"n": int(ri.sum()), "pct_of_cohort": r1(100 * ri.mean()), "pct_cellulitis": frac(ri, 1), "LR": lr(ri)},
            "pct_actionable": r1(100 * (ro.mean() + ri.mean()))}


def nri(y, ba, br):
    ev, ne = y == 1, y == 0
    e = (br > ba)[ev].mean() - (br < ba)[ev].mean()
    n = (br < ba)[ne].mean() - (br > ba)[ne].mean()
    return float(e), float(n), float(e + n)


def race_bucket(r):
    r = r.upper() if isinstance(r, str) else ""
    return ("White" if r.startswith("WHITE") else "Black" if "BLACK" in r else
            "Hispanic" if ("HISPANIC" in r or "LATINO" in r) else "Asian" if "ASIAN" in r else "Other/Unknown")


# ----------------------------------------------------------------------------- data
def load_mimic():
    f = pd.read_parquet(FEATURES)
    f["hadm_id"] = f.hadm_id.astype("int64"); f["subject_id"] = f.subject_id.astype("int64")
    f["y"] = (f.label == "positive").astype(int)
    f["leuko_max"] = (f.wbc_max >= 10).fillna(False).astype(int)
    f["tachy_max"] = (f.hr_max >= 90).fillna(False).astype(int)
    f["age70"] = (f.age >= 70).astype(int)
    f["alt70_struct"] = 2 * f.age70 + f.leuko_max + f.tachy_max
    n = pd.read_json(NLP, lines=True)
    n = n[n["ok"].astype(str) == "True"].copy()
    n["asymmetry"] = n.laterality.isin(["unilateral_left", "unilateral_right"]).astype(int)
    n["hadm_id"] = n.hadm_id.astype("int64")
    d = f.merge(n[["hadm_id", "asymmetry", "laterality"]], on="hadm_id", how="inner").copy()
    d["alt70_complete"] = 3 * d.asymmetry + d.leuko_max + d.tachy_max + 2 * d.age70
    # calculator-faithful binary items (first values at presentation)
    d["leuko"] = (d.wbc_first >= 10).fillna(False).astype(int)
    d["crp_hi"] = (d.crp_first >= 50).fillna(False).astype(int)
    d["neut_hi"] = (d.neutrophils_abs_first >= 7.5).fillna(False).astype(int)
    for c in ["como_malignancy_active", "como_diabetes", "como_chf", "como_ckd", "como_obesity"]:
        d[c] = d[c].fillna(0).astype(int)
    return f, d


def tier_columns(d):
    allcols = [c for c in d.columns if c not in META and not c.startswith("alt70") and not c.startswith("micro")
               and c not in {"y", "leuko_max", "tachy_max", "age70", "leuko", "crp_hi", "neut_hi", "laterality",
                             "alt70_struct", "alt70_complete"}]
    standard = (["age", "female", "bmi", "asymmetry"]
                + [c for c in allcols if c.endswith("_first") and not c.startswith(VITAL_PREFIX)]
                + [c for c in allcols if c.startswith("como_")])
    standard = [c for c in dict.fromkeys(standard) if c in allcols]
    comprehensive = list(dict.fromkeys(allcols))  # every structured feature (first/min/max labs, vitals, comorbidities) + asymmetry; NO culture
    bedside = ["asymmetry", "leuko", "age70", "como_malignancy_active", "como_diabetes"]
    rapid = bedside + ["crp_hi", "neut_hi", "como_chf", "como_ckd", "como_obesity"]
    assert not any(c.startswith("micro") for c in comprehensive)
    return {"Bedside": bedside, "Rapid": rapid, "Standard": standard, "Comprehensive": comprehensive}


# ----------------------------------------------------------------------------- main
def main(default_cfg="config/default.yaml"):
    seed = yaml.safe_load(Path(default_cfg).read_text())["seed"]
    R = {"meta": {"seed": seed, "n_boot": NBOOT, "culture": "EXCLUDED from every model and every tier",
                  "git_hash": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                  "built_at": dt.datetime.now().isoformat(timespec="seconds"),
                  "software": {"python": sys.version.split()[0], "sklearn": __import__("sklearn").__version__,
                               "statsmodels": sm.__version__, "xgboost": __import__("xgboost").__version__,
                               "numpy": np.__version__, "pandas": pd.__version__, "duckdb": duckdb.__version__}}}
    f, d = load_mimic()
    tiers = tier_columns(d)
    tr, te = d[d.split.isin(["train", "val"])].copy(), d[d.split == "test"].copy()
    ftr, fte = f[f.split.isin(["train", "val"])].copy(), f[f.split == "test"].copy()
    yte = te.y.to_numpy(); ytr = tr.y.to_numpy()
    log(f"MIMIC: full n={len(f)} (test {len(fte)}), NLP-covered n={len(d)} (test {len(te)}, prev {yte.mean():.3f})")
    R["cohort"] = {"n_full": int(len(f)), "n_full_test": int(len(fte)), "prev_full_test": r3(fte.y.mean()),
                   "n_nlp_covered": int(len(d)), "n_nlp_test": int(len(te)), "prev_nlp_test": r3(yte.mean()),
                   "n_nlp_test_pos": int(yte.sum()), "n_nlp_test_neg": int((yte == 0).sum()),
                   "nlp_uncovered_test_n": int(len(fte) - len(te)),
                   "nlp_uncovered_test_prev": r3(fte[~fte.hadm_id.isin(te.hadm_id)].y.mean()),
                   "laterality_distribution_nlp_covered": d.laterality.value_counts().to_dict(),
                   "tier_feature_counts": {k: len(v) for k, v in tiers.items()},
                   "coverage_pct": {"wbc_first": r1(100 * f.wbc_first.notna().mean()),
                                    "hr_any_icu": r1(100 * f.hr_max.notna().mean()),
                                    "crp_first": r1(100 * f.crp_first.notna().mean()),
                                    "note_processed_of_cohort": r1(100 * len(d) / len(f))}}

    # ---- Bedside + Rapid: unpenalized logistic on binary items (calculator model), ORs, points
    bed = logit_binary(tr[tiers["Bedside"]], ytr)
    rap = logit_binary(tr[tiers["Rapid"]], ytr)
    R["bedside_model"] = bed; R["rapid_model"] = rap
    te["p_bed"] = predict_binary(bed, te); te["s_bed"] = points_score(bed, te)
    tr["p_bed"] = predict_binary(bed, tr); tr["s_bed"] = points_score(bed, tr)
    te["p_rap"] = predict_binary(rap, te)
    log("Bedside points: " + str({c: v["points"] for c, v in bed["terms"].items()}))
    log("Bedside ORs: " + str({c: (v["or"], v["or_lo"], v["or_hi"]) for c, v in bed["terms"].items()}))

    # ---- Standard + Comprehensive: regularized logistic pipelines; XGBoost benchmark on Comprehensive
    fits = {}
    for name in ("Standard", "Comprehensive"):
        m = pipe_lr(seed).fit(tr[tiers[name]].astype(float).to_numpy(), ytr)
        te[f"p_{name}"] = m.predict_proba(te[tiers[name]].astype(float).to_numpy())[:, 1]; fits[name] = m
    xg = xgb(seed).fit(tr[tiers["Comprehensive"]].astype(float).to_numpy(), ytr)
    te["p_xgb"] = xg.predict_proba(te[tiers["Comprehensive"]].astype(float).to_numpy())[:, 1]
    gain = sorted(zip(tiers["Comprehensive"], xg.feature_importances_), key=lambda x: -x[1])
    R["xgb_feature_importance_top20"] = [{"feature": c, "gain": round(float(g), 4)} for c, g in gain[:20]]

    # ---- inpatient head-to-head (paired, same resamples)
    scores = {"alt70_complete": te.alt70_complete, "alt70_structured": te.alt70_struct,
              "bedside_prob": te.p_bed, "bedside_points": te.s_bed, "rapid_prob": te.p_rap,
              "standard_prob": te.p_Standard, "comprehensive_prob": te.p_Comprehensive, "xgb_comprehensive": te.p_xgb}
    pairs = [("bedside_prob", "alt70_complete"), ("bedside_points", "alt70_complete"), ("rapid_prob", "alt70_complete"),
             ("standard_prob", "alt70_complete"), ("comprehensive_prob", "alt70_complete"),
             ("xgb_comprehensive", "comprehensive_prob"), ("comprehensive_prob", "bedside_prob")]
    aucs, diffs = paired_boot(yte, scores, pairs)
    R["inpatient_nlp_test"] = {"aurocs": aucs, "paired_differences": diffs}
    log("inpatient AUROCs: " + ", ".join(f"{k} {v['auroc']}" for k, v in aucs.items()))
    log("bedside - ALT-70: " + str(diffs["bedside_prob_minus_alt70_complete"]))
    # full-test structured ALT-70 (no asymmetry) and a structured-only 93-feature reference, for completeness
    fcols = [c for c in f.columns if c not in META and not c.startswith("alt70") and not c.startswith("micro")
             and c not in {"y", "leuko_max", "tachy_max", "age70", "alt70_struct"}]
    mfull = pipe_lr_ind(seed).fit(ftr[fcols].astype(float).to_numpy(), ftr.y.to_numpy())
    p_full = mfull.predict_proba(fte[fcols].astype(float).to_numpy())[:, 1]
    a2, _ = paired_boot(fte.y.to_numpy(), {"alt70_structured": fte.alt70_struct, "structured_only_logistic": p_full}, [])
    R["inpatient_full_test"] = {"n_features_structured_only": len(fcols), **a2}
    R["auprc_nlp_test"] = {k: boot_auprc(yte, v) for k, v in
                           {"alt70_complete": te.alt70_complete, "bedside_prob": te.p_bed,
                            "comprehensive_prob": te.p_Comprehensive, "xgb_comprehensive": te.p_xgb}.items()}
    log("full-test: " + str(a2))

    # ---- ROC curves for the figure
    R["roc_inpatient"] = {"bedside": roc_points(yte, te.p_bed), "comprehensive": roc_points(yte, te.p_Comprehensive),
                          "alt70_complete": roc_points(yte, te.alt70_complete)}

    # ---- bands (prespecified probability thresholds used by the calculator) + per-point-score prevalence
    t_lo, t_hi = 0.30, 0.65
    R["bands"] = {"thresholds": {"rule_out_below": t_lo, "rule_in_at_or_above": t_hi},
                  "bedside_test": bands_prob(te.p_bed, yte, t_lo, t_hi),
                  "comprehensive_test": bands_prob(te.p_Comprehensive, yte, t_lo, t_hi),
                  "alt70_bands_same_patients": [], "raff2017_published": {"0-2": 9, "3-4": 72, "5-7": 95}}
    for name, lo, hi in [("0-2 (rule-out)", -1, 2), ("3-4 (indeterminate)", 3, 4), ("5-7 (rule-in)", 5, 7)]:
        m = (te.alt70_complete >= lo) & (te.alt70_complete <= hi)
        R["bands"]["alt70_bands_same_patients"].append({"band": name, "n": int(m.sum()), "pct_cellulitis": r1(100 * yte[m].mean())})
    per_score = {}
    for s in sorted(te.s_bed.unique()):
        m = te.s_bed == s
        per_score[int(s)] = {"n": int(m.sum()), "obs_pct_cellulitis": r1(100 * yte[m].mean()),
                             "mean_prob": r3(te.p_bed[m].mean())}
    R["bedside_points_to_observed"] = per_score
    profiles = te.groupby(tiers["Bedside"]).agg(n=("y", "size"), obs=("y", "mean"), pred=("p_bed", "mean")).reset_index()
    R["bedside_profiles_test"] = [{**{c: int(r[c]) for c in tiers["Bedside"]}, "n": int(r.n), "obs_pct": r1(100 * r.obs),
                                   "pred_pct": r1(100 * r.pred)} for _, r in profiles.iterrows() if r.n >= 20]
    log("bands bedside: " + str(R["bands"]["bedside_test"]))

    # ---- operating points: integer score vs ALT-70 (published cut-points)
    smin, smax = int(te.s_bed.min()), int(te.s_bed.max())
    rev = op_table(te.s_bed, yte, range(smin, smax + 1))
    alt = op_table(te.alt70_complete, yte, range(1, 8))
    alt5 = next(r for r in alt if r["threshold"] == 5)
    matched = min(rev, key=lambda r: abs(r["spec"] - alt5["spec"]))
    rule_in_t = next((r for r in rev if r["spec"] >= 90 and r["tp"] + r["fp"] >= 50), rev[-1])
    R["operating_inpatient"] = {
        "n": int(len(te)), "prevalence": r3(yte.mean()), "standardized_prevalence": RAFF_PREV,
        "bedside_points_table": rev, "alt70_table": alt,
        "published_alt70": {"raff2017_ge5": {"sens": 61.3, "spec": 70.9, "ruleIn_band_PPV": 95},
                            "li2018_ge3": {"sens": 97.8, "spec": 47.6, "ruleIn_ppv": 80.4}},
        "key_points": {"alt70_ge5": {**alt5, **boot_op(te.alt70_complete, yte, 5)},
                       "bedside_matched_spec": {**matched, **boot_op(te.s_bed, yte, matched["threshold"])},
                       "bedside_rule_in": {**rule_in_t, **boot_op(te.s_bed, yte, rule_in_t["threshold"])}}}
    log(f"matched-spec bedside >= {matched['threshold']}: sens {matched['sens']} spec {matched['spec']} | ALT-70>=5 sens {alt5['sens']} spec {alt5['spec']}")

    # ---- calibration, AUPRC, subgroups
    R["calibration"] = {"bedside": calib(te.p_bed, yte), "comprehensive": calib(te.p_Comprehensive, yte)}
    src = from_config(default_cfg)
    race = src.query("SELECT DISTINCT hadm_id, race FROM admissions"); race["hadm_id"] = race.hadm_id.astype("int64")
    ayg = src.query("SELECT subject_id, anchor_year_group FROM patients"); ayg["subject_id"] = ayg.subject_id.astype("int64")
    abx = yaml.safe_load(Path("config/stewardship.yaml").read_text())["antibiotics"]
    abx_like = " OR ".join(f"lower(drug) LIKE '%{a}%'" for a in abx)
    dot = src.query(f"SELECT p.hadm_id, sum(greatest(date_diff('day',p.starttime,p.stoptime)+1,1)) dot "
                    f"FROM prescriptions p WHERE ({abx_like}) AND p.starttime IS NOT NULL GROUP BY p.hadm_id")
    dot["hadm_id"] = dot.hadm_id.astype("int64")
    adm = src.query("SELECT hadm_id, date_diff('day',admittime,dischtime) los, hospital_expire_flag died FROM admissions")
    adm["hadm_id"] = adm.hadm_id.astype("int64")
    src.close()
    te2 = te.merge(race, on="hadm_id", how="left")
    te2["race_grp"] = te2.race.map(race_bucket)
    te2["age_grp"] = pd.cut(te2.age, [0, 50, 70, 200], labels=["<50", "50-69", ">=70"]).astype(str)
    te2["sex"] = np.where(te2.gender == "F", "Female", "Male")
    sub = {}
    for model in ("p_bed", "p_Comprehensive"):
        s = {"overall": paired_boot(te2.y, {"s": te2[model]}, [])[0]["s"]}
        for key, col in [("by_sex", "sex"), ("by_age", "age_grp"), ("by_race", "race_grp")]:
            s[key] = {}
            for g, gd in te2.groupby(col):
                if len(gd) >= 100 and gd.y.nunique() == 2:
                    s[key][str(g)] = {**paired_boot(gd.y, {"s": gd[model]}, [])[0]["s"], "prevalence": r3(gd.y.mean())}
        sub[{"p_bed": "bedside", "p_Comprehensive": "comprehensive"}[model]] = s
    R["subgroups_nlp_test"] = sub
    log("subgroups done")

    # ---- stewardship simulation on the NLP-covered test (Bedside and Comprehensive)
    te3 = te.merge(dot, on="hadm_id", how="left"); te3["dot"] = te3["dot"].fillna(0)
    cel, mim = te3[te3.y == 1], te3[te3.y == 0]
    stew = {"n_test": int(len(te3)), "n_cellulitis": int(len(cel)), "n_mimic": int(len(mim)),
            "total_mimic_abx_days": float(mim["dot"].sum()), "pct_mimic_any_abx_test": r1(100 * (mim["dot"] > 0).mean())}
    for model in ("p_bed", "p_Comprehensive"):
        for target in (0.95, 0.90):
            t = float(np.quantile(cel[model], 1 - target))
            withheld = mim[mim[model] < t]; missed = int((cel[model] < t).sum()); tot_dot = float(mim["dot"].sum())
            stew[f"{model}_at_sens_{int(target*100)}"] = {
                "threshold": r3(t), "sensitivity_cellulitis": r3((cel[model] >= t).mean()),
                "specificity_mimic": r3((mim[model] < t).mean()), "avertable_abx_days": float(withheld["dot"].sum()),
                "pct_of_mimic_abx_days_averted": r1(100 * withheld["dot"].sum() / tot_dot) if tot_dot else 0,
                "avertable_abx_days_per_100_pts": r1(100 * withheld["dot"].sum() / len(te3)),
                "missed_cellulitis_n": missed, "missed_cellulitis_pct": r1(100 * missed / len(cel))}
    R["stewardship_nlp_test"] = stew
    log("stewardship: " + str(stew["p_bed_at_sens_95"]))

    # ---- NRI (Bedside bands vs ALT-70 bands), per-category, label noise, temporal
    ba = np.where(te.alt70_complete <= 2, 0, np.where(te.alt70_complete <= 4, 1, 2))
    br = np.where(te.p_bed < t_lo, 0, np.where(te.p_bed >= t_hi, 2, 1))
    e, nn, tot = nri(yte, ba, br)
    rng = np.random.default_rng(seed); S = []
    for _ in range(NBOOT):
        i = strat_idx(rng, yte); S.append(nri(yte[i], ba[i], br[i])[2])
    R["nri_bedside_vs_alt70_bands"] = {"nri_total": r3(tot), "ci": [r3(np.percentile(S, 2.5)), r3(np.percentile(S, 97.5))],
                                       "events_component": r3(e), "nonevents_component": r3(nn)}
    cond = condition_hadms(te[te.y == 0].hadm_id.tolist())
    pc = {"overall_auroc_bedside": r3(roc_auc_score(yte, te.p_bed)), "by_category": {}, "by_condition": {}}
    mim_ids = set(te[te.y == 0].hadm_id)
    for cat, conds in CATEGORY.items():
        hs = set().union(*(cond[c] for c in conds))
        subm = te[(te.y == 1) | te.hadm_id.isin(hs)]
        n_m = int((subm.y == 0).sum())
        pc["by_category"][cat] = {"n_mimic": n_m, "pct_of_mimics": r1(100 * n_m / len(mim_ids)),
                                  "auroc_bedside": r3(roc_auc_score(subm.y, subm.p_bed)) if n_m >= 20 else None,
                                  "auroc_comprehensive": r3(roc_auc_score(subm.y, subm.p_Comprehensive)) if n_m >= 20 else None}
    for name, hs in cond.items():
        pc["by_condition"][name] = int(len(hs & mim_ids))
    R["per_mimic_category"] = pc
    log("per-category: " + str({k: v["auroc_bedside"] for k, v in pc["by_category"].items()}))

    def refit_bed(yt):
        m = logit_binary(tr[tiers["Bedside"]], yt)
        return {"auroc": r3(roc_auc_score(yte, predict_binary(m, te))),
                **{f"{c}_OR": m["terms"][c]["or"] for c in ("asymmetry", "age70", "como_malignancy_active", "como_diabetes", "leuko")}}
    rng = np.random.default_rng(seed); ln = {"baseline": refit_bed(ytr)}
    for pf in (0.05, 0.10, 0.20):
        yt = ytr.copy(); fl = rng.random(len(yt)) < pf; yt[fl] = 1 - yt[fl]; ln[f"nondiff_flip_{int(pf*100)}pct"] = refit_bed(yt)
    for pf in (0.10, 0.20, 0.30):
        yt = ytr.copy(); mm = np.where(yt == 0)[0]; idx = rng.choice(mm, int(pf * len(mm)), False); yt[idx] = 1
        ln[f"diff_mimic_to_case_{int(pf*100)}pct"] = refit_bed(yt)
    R["label_noise_bedside"] = ln
    d2 = d.merge(ayg, on="subject_id", how="left")
    early = d2.anchor_year_group.isin(["2008 - 2010", "2011 - 2013", "2014 - 2016"]).to_numpy()
    late = d2.anchor_year_group.isin(["2017 - 2019", "2020 - 2022"]).to_numpy()
    def tsplit(trn, tst):
        m = logit_binary(d2.loc[trn, tiers["Bedside"]], d2.loc[trn, "y"])
        p = predict_binary(m, d2.loc[tst]); y = d2.loc[tst, "y"].to_numpy()
        return {"n_train": int(trn.sum()), "n_test": int(tst.sum()), "test_prevalence": r3(y.mean()),
                "bedside_auroc": r3(roc_auc_score(y, p)), "alt70_complete_auroc": r3(roc_auc_score(y, d2.loc[tst, "alt70_complete"]))}
    R["temporal_bedside"] = {"train_2008_2016_test_2017_2022": tsplit(early, late), "train_2017_2022_test_2008_2016": tsplit(late, early)}
    log("temporal: " + str(R["temporal_bedside"]))

    # ---- Table 1 (counts, IQRs, race) on the full analytic cohort
    t1 = f.merge(race, on="hadm_id", how="left").merge(adm, on="hadm_id", how="left")
    t1["race_grp"] = t1.race.map(race_bucket)
    t1["note_processed"] = t1.hadm_id.isin(d.hadm_id).astype(int)
    grp = {"cellulitis": t1[t1.y == 1], "mimic": t1[t1.y == 0], "overall": t1}
    def smd_cont(a, b):
        a, b = a.dropna(), b.dropna(); sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
        return round(float((a.mean() - b.mean()) / sd), 2) if sd else 0.0
    def smd_bin(a, b):
        pa, pb = a.mean(), b.mean(); sd = np.sqrt((pa * (1 - pa) + pb * (1 - pb)) / 2)
        return round(float((pa - pb) / sd), 2) if sd else 0.0
    T = {"n": {k: int(len(v)) for k, v in grp.items()}, "rows": []}
    def add_bin(label, col):
        row = {"label": label, "type": "binary"}
        for k, v in grp.items():
            s = v[col].fillna(0).astype(int); row[k] = {"n": int(s.sum()), "pct": r1(100 * s.mean())}
        row["smd"] = smd_bin(grp["cellulitis"][col].fillna(0), grp["mimic"][col].fillna(0)); T["rows"].append(row)
    def add_cont(label, col, kind="median"):
        row = {"label": label, "type": kind}
        for k, v in grp.items():
            s = v[col].dropna()
            row[k] = ({"mean": r1(s.mean()), "sd": r1(s.std())} if kind == "mean"
                      else {"median": r1(s.median()), "q1": r1(s.quantile(.25)), "q3": r1(s.quantile(.75)), "n_avail": int(len(s))})
        row["smd"] = smd_cont(grp["cellulitis"][col], grp["mimic"][col]); T["rows"].append(row)
    add_cont("Age, y", "age", "mean"); add_bin("Female sex", "female")
    for rg in ["Asian", "Black", "Hispanic", "White", "Other/Unknown"]:
        t1[f"race_{rg}"] = (t1.race_grp == rg).astype(int); grp = {"cellulitis": t1[t1.y == 1], "mimic": t1[t1.y == 0], "overall": t1}
        add_bin(f"Race and ethnicity: {rg}", f"race_{rg}")
    for label, col in [("Diabetes", "como_diabetes"), ("Heart failure", "como_chf"), ("Chronic kidney disease", "como_ckd"),
                       ("Obesity", "como_obesity"), ("Liver disease", "como_liver_disease"), ("HIV infection", "como_hiv"),
                       ("Solid-organ transplant", "como_transplant"), ("Active malignancy", "como_malignancy_active")]:
        add_bin(label, col)
    add_cont("White blood cell count, first, x10^3/uL", "wbc_first"); add_cont("C-reactive protein, first, mg/L", "crp_first")
    add_cont("Creatinine, first, mg/dL", "creatinine_first"); add_cont("Hemoglobin, first, g/dL", "hemoglobin_first")
    add_cont("Platelet count, first, x10^3/uL", "platelet_first"); add_cont("Sodium, first, mmol/L", "sodium_first")
    add_bin("Heart rate recorded (ICU monitoring)", "has_icu_hr"); add_bin("Discharge summary processed for laterality", "note_processed")
    t1["unilateral"] = t1.hadm_id.map(d.drop_duplicates("hadm_id").set_index("hadm_id").asymmetry).astype(float)
    grp = {"cellulitis": t1[t1.y == 1], "mimic": t1[t1.y == 0], "overall": t1}
    row = {"label": "Unilateral involvement, among processed notes", "type": "binary"}
    for k, v in grp.items():
        s = v.unilateral.dropna(); row[k] = {"n": int(s.sum()), "pct": r1(100 * s.mean()), "denominator": int(len(s))}
    row["smd"] = smd_bin(grp["cellulitis"].unilateral.dropna(), grp["mimic"].unilateral.dropna()); T["rows"].append(row)
    t1 = t1.merge(dot, on="hadm_id", how="left"); t1["dot"] = t1["dot"].fillna(0); t1["dot_any"] = (t1["dot"] > 0).astype(int)
    grp = {"cellulitis": t1[t1.y == 1], "mimic": t1[t1.y == 0], "overall": t1}
    add_bin("Any systemic antibiotic during admission", "dot_any")
    add_cont("Length of stay, d", "los"); add_bin("In-hospital death", "died")
    OUT_T1.write_text(json.dumps(T, indent=1))
    R["table1_summary"] = {"overall_age_mean": T["rows"][0]["overall"], "overall_female": T["rows"][1]["overall"]}
    log("Table 1 written")
    OUT.write_text(json.dumps(R, indent=1))
    log("checkpoint written (MIMIC inpatient complete)")

    # ---- MIMIC-IV-ED: ALT-70's native setting
    con = duckdb.connect(config={"memory_limit": "8GB", "threads": 6})
    MIMIC = ("icd_code LIKE '4534%' OR icd_code LIKE 'I82%' OR lower(icd_title) LIKE '%deep vein thromb%' "
             "OR icd_code LIKE '274%' OR icd_code LIKE 'M10%' OR lower(icd_title) LIKE '%gout%' "
             "OR lower(icd_title) LIKE '%stasis%' OR lower(icd_title) LIKE '%venous%' OR lower(icd_title) LIKE '%lymphedema%' "
             "OR lower(icd_title) LIKE '%dermatitis%' OR lower(icd_title) LIKE '%thrombophleb%'")
    con.execute(f"CREATE VIEW dx AS SELECT * FROM read_csv_auto('{ED}/diagnosis.csv.gz')")
    con.execute(f"CREATE VIEW st AS SELECT * FROM read_csv_auto('{ED}/edstays.csv.gz')")
    con.execute(f"CREATE VIEW tri AS SELECT * FROM read_csv_auto('{ED}/triage.csv.gz')")
    con.execute(f"CREATE VIEW pat AS SELECT * FROM read_csv_auto('{H}/patients.csv.gz')")
    con.execute(f"""CREATE TEMP TABLE flag AS SELECT stay_id,
        max(CASE WHEN icd_code LIKE '682%' OR icd_code LIKE 'L03%' OR lower(icd_title) LIKE '%cellulitis%' THEN 1 ELSE 0 END) cell,
        max(CASE WHEN {MIMIC} THEN 1 ELSE 0 END) mim FROM dx GROUP BY stay_id""")
    con.execute("""CREATE TEMP TABLE cohort AS
        WITH lab AS (SELECT stay_id, cell FROM flag WHERE cell + mim = 1),
        j AS (SELECT s.subject_id, s.stay_id, s.hadm_id, s.intime, l.cell AS y,
                p.anchor_age + (year(s.intime) - p.anchor_year) AS age,
                row_number() OVER (PARTITION BY s.subject_id ORDER BY s.intime) rn
              FROM lab l JOIN st s USING (stay_id) JOIN pat p ON p.subject_id = s.subject_id)
        SELECT subject_id, stay_id, hadm_id, intime, y, age FROM j WHERE rn = 1 AND age >= 18""")
    con.execute("""CREATE TEMP TABLE vit AS SELECT c.stay_id,
        max(CASE WHEN t.heartrate BETWEEN 20 AND 250 THEN t.heartrate END) hr,
        max(CASE WHEN t.temperature BETWEEN 90 AND 110 THEN t.temperature END) temp_f
        FROM cohort c JOIN tri t USING (stay_id) GROUP BY c.stay_id""")
    log("ED: scanning labevents for WBC (large file)")
    con.execute(f"""CREATE TEMP TABLE wbc AS SELECT c.stay_id, max(le.valuenum) wbc
        FROM cohort c JOIN read_csv_auto('{H}/labevents.csv.gz') le
          ON le.subject_id = c.subject_id AND le.itemid IN (51300, 51301) AND le.valuenum IS NOT NULL
          AND le.charttime BETWEEN c.intime - INTERVAL 6 HOUR AND c.intime + INTERVAL 24 HOUR
        GROUP BY c.stay_id""")
    ed = con.execute("SELECT c.*, v.hr, v.temp_f, w.wbc FROM cohort c LEFT JOIN vit v USING(stay_id) LEFT JOIN wbc w USING(stay_id)").df()
    con.close()
    ed["age70"] = (ed.age >= 70).astype(int); ed["tachy"] = (ed.hr >= 90).fillna(False).astype(int)
    ed["leuko"] = (ed.wbc >= 10).fillna(False).astype(int); ed["fever"] = (ed.temp_f >= 100.4).fillna(False).astype(int)
    ed["alt70"] = 2 * ed.age70 + ed.leuko + ed.tachy
    n = pd.read_json(NLP, lines=True); n = n[n["ok"].astype(str) == "True"].copy()
    n["asymmetry"] = n.laterality.isin(["unilateral_left", "unilateral_right"]).astype(int); n["hadm_id"] = n.hadm_id.astype("int64")
    ed["hadm_id"] = pd.to_numeric(ed.hadm_id, errors="coerce")
    ed = ed.merge(n[["hadm_id", "asymmetry"]], on="hadm_id", how="left")
    rng = np.random.default_rng(SEED); uniq = np.unique(ed.subject_id.to_numpy()); rng.shuffle(uniq)
    tem = ed.subject_id.isin(set(uniq[: int(0.3 * len(uniq))])).to_numpy()
    ed_cols = ["leuko", "tachy", "age70", "fever"]
    edm = logit_binary(ed.loc[~tem, ed_cols], ed.loc[~tem, "y"])
    p_ed = predict_binary(edm, ed.loc[tem]); y_ed = ed.loc[tem, "y"].to_numpy().astype(int); alt_ed = ed.loc[tem, "alt70"].to_numpy()
    a_ed, d_ed = paired_boot(y_ed, {"celtic_structured": p_ed, "alt70_triage_vitals": alt_ed}, [("celtic_structured", "alt70_triage_vitals")])
    a_all, _ = paired_boot(ed.y.to_numpy().astype(int), {"alt70_triage_vitals_all": ed.alt70}, [])
    alt_tab = {f">= {t}": opchar(alt_ed >= t, y_ed) for t in range(1, 5)}
    thr = np.unique(p_ed)
    def at_spec(target):
        best = min(thr, key=lambda t: abs(float((p_ed >= t)[y_ed == 0].mean()) - (1 - target)))
        return {"prob_threshold": r3(best), **opchar(p_ed >= best, y_ed)}
    adm_ = ed[ed.asymmetry.notna()].copy()
    adm_["alt70_complete"] = 3 * adm_.asymmetry.astype(int) + adm_.leuko + adm_.tachy + 2 * adm_.age70
    ya = adm_.y.to_numpy().astype(int)
    a_adm, _ = paired_boot(ya, {"alt70_complete_4component": adm_.alt70_complete}, [])
    bands_ed = []
    for name, lo, hi in [("0-2 (rule-out)", -1, 2), ("3-4 (indeterminate)", 3, 4), ("5-7 (rule-in)", 5, 7)]:
        m = (adm_.alt70_complete >= lo) & (adm_.alt70_complete <= hi)
        bands_ed.append({"band": name, "n": int(m.sum()), "pct_cellulitis": r1(100 * ya[m].mean())})
    R["ed"] = {"cohort": {"n": int(len(ed)), "cellulitis": int(ed.y.sum()), "prevalence": r3(ed.y.mean()),
                          "hr_coverage_pct": r1(100 * ed.hr.notna().mean()), "wbc_coverage_pct": r1(100 * ed.wbc.notna().mean()),
                          "temp_coverage_pct": r1(100 * ed.temp_f.notna().mean()), "n_test": int(tem.sum()),
                          "test_prevalence": r3(y_ed.mean()), "admitted_with_nlp_asym": int(len(adm_))},
               "celtic_structured_model": edm, "aurocs_test": a_ed, "alt70_all_stays": a_all, "paired": d_ed,
               "alt70_operating": alt_tab, "celtic_matched_spec_alt70_ge3": at_spec(alt_tab[">= 3"]["spec"] / 100),
               "celtic_high_spec_90": at_spec(0.90), "roc": roc_points(y_ed, p_ed),
               "admitted_nlp_subset": {"n": int(len(adm_)), "prevalence": r3(ya.mean()), **a_adm, "alt70_bands": bands_ed}}
    log("ED: " + str(a_ed) + " paired " + str(d_ed))
    OUT.write_text(json.dumps(R, indent=1))

    # ---- eICU-CRD external institutions (structured only, no notes)
    log("eICU: building cohort (large vitals scan)")
    con = duckdb.connect(config={"memory_limit": "8GB", "threads": 6})
    e = eicu_cohort_features(con); con.close()
    shared = [c for c in SHARED if not c.startswith("micro")]
    me = pipe_lr_ind(seed).fit(ftr[shared].astype(float).to_numpy(), ftr.y.to_numpy())
    p_e = me.predict_proba(e[shared].astype(float).to_numpy())[:, 1]
    p_e_int = me.predict_proba(fte[shared].astype(float).to_numpy())[:, 1]
    ye = e.y.to_numpy().astype(int)
    alt_e = (2 * (e.age >= 70).astype(int) + (e.wbc_max >= 10).fillna(False).astype(int) + (e.hr_max >= 90).fillna(False).astype(int)).to_numpy()
    a_e, d_e = paired_boot(ye, {"celtic_structured": p_e, "alt70_structured": alt_e}, [("celtic_structured", "alt70_structured")])
    alt_tab_e = {f">= {t}": opchar(alt_e >= t, ye) for t in range(1, 5)}
    thr = np.unique(p_e)
    def at_spec_e(target):
        best = min(thr, key=lambda t: abs(float((p_e >= t)[ye == 0].mean()) - (1 - target)))
        return {"prob_threshold": r3(best), **opchar(p_e >= best, ye)}
    R["eicu"] = {"cohort": {"n": int(len(e)), "cellulitis": int(ye.sum()), "prevalence": r3(ye.mean()), "n_shared_features": len(shared),
                            "shared_features": shared},
                 "mimic_internal_test_same_features": {"celtic_structured": r3(roc_auc_score(fte.y, p_e_int)),
                                                       "alt70_structured": r3(roc_auc_score(fte.y, fte.alt70_struct))},
                 "aurocs": a_e, "paired": d_e, "alt70_operating": alt_tab_e,
                 "celtic_matched_spec_alt70_ge3": at_spec_e(alt_tab_e[">= 3"]["spec"] / 100),
                 "celtic_high_spec_90": at_spec_e(0.90), "roc": roc_points(ye, p_e)}
    log("eICU: " + str(a_e) + " paired " + str(d_e))
    OUT.write_text(json.dumps(R, indent=1))
    log("DONE -> " + str(OUT))


if __name__ == "__main__":
    main()
