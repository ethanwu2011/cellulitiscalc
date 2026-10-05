"""Why discrimination is lower in the emergency department: structured-item models on the MIMIC-IV-ED test set, plus the
like-for-like comparison the ED allows once a note exists (admitted stays with a processed discharge summary, so BOTH
scores have the examination item). Merges into celtic_noculture_metrics.json under ed["decomposition"] and
ed["admitted_like_for_like"]. No culture result is used.
Aggregate-only. Run from analysis/ after celtic_noculture.py:  PYTHONPATH=. python study2_cellulitis/celtic_noculture_ed3.py
"""
from __future__ import annotations
import json
from pathlib import Path
import duckdb, numpy as np, pandas as pd, yaml
from sklearn.metrics import roc_auc_score
from study2_cellulitis.celtic_noculture import (ED, H, NLP, OUT, SEED, load_mimic, logit_binary, opchar,
                                                 paired_boot, predict_binary, roc_points)

M = json.loads(OUT.read_text())
comos = yaml.safe_load(Path("config/features_study2.yaml").read_text())["comorbidities"]
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
con.execute(f"""CREATE TEMP TABLE wbc AS SELECT c.stay_id, max(le.valuenum) wbc
    FROM cohort c JOIN read_csv_auto('{H}/labevents.csv.gz') le
      ON le.subject_id = c.subject_id AND le.itemid IN (51300, 51301) AND le.valuenum IS NOT NULL
      AND le.charttime BETWEEN c.intime - INTERVAL 6 HOUR AND c.intime + INTERVAL 24 HOUR
    GROUP BY c.stay_id""")
def expr(name, spec):
    cl = [f"(d.icd_version = 9 AND d.icd_code LIKE '{p}%')" for p in spec.get("icd9_prefix", [])]
    cl += [f"(d.icd_version = 10 AND d.icd_code LIKE '{p}%')" for p in spec.get("icd10_prefix", [])]
    return f"max(CASE WHEN {' OR '.join(cl)} THEN 1 ELSE 0 END) AS como_{name}"
sel = ", ".join(expr(n, s) for n, s in comos.items() if n in ("diabetes", "malignancy_active"))
con.execute(f"""CREATE TEMP TABLE como AS SELECT c.stay_id, {sel}
    FROM cohort c JOIN read_csv_auto('{H}/admissions.csv.gz') a ON a.subject_id = c.subject_id AND a.admittime <= c.intime + INTERVAL 2 DAY
    JOIN read_csv_auto('{H}/diagnoses_icd.csv.gz', types={{'icd_code': 'VARCHAR'}}) d ON d.hadm_id = a.hadm_id
    GROUP BY c.stay_id""")
ed = con.execute("""SELECT c.*, v.hr, v.temp_f, w.wbc, k.como_diabetes, k.como_malignancy_active
    FROM cohort c LEFT JOIN vit v USING(stay_id) LEFT JOIN wbc w USING(stay_id) LEFT JOIN como k USING(stay_id)""").df()
con.close()
for c in ("como_diabetes", "como_malignancy_active"):
    ed[c] = ed[c].fillna(0).astype(int)
ed["age70"] = (ed.age >= 70).astype(int); ed["tachy"] = (ed.hr >= 90).fillna(False).astype(int)
ed["leuko"] = (ed.wbc >= 10).fillna(False).astype(int); ed["fever"] = (ed.temp_f >= 100.4).fillna(False).astype(int)
ed["alt70"] = 2 * ed.age70 + ed.leuko + ed.tachy
rng = np.random.default_rng(SEED); uniq = np.unique(ed.subject_id.to_numpy()); rng.shuffle(uniq)
tem = ed.subject_id.isin(set(uniq[: int(0.3 * len(uniq))])).to_numpy()
y = ed.loc[tem, "y"].to_numpy().astype(int)

dec = {"n_test": int(tem.sum()), "prevalence_test": round(float(y.mean()), 3),
       "auroc_test": {"alt70_triage_vitals": round(float(roc_auc_score(y, ed.loc[tem, "alt70"])), 3)}}
for key, cols in [("structured_4", ["leuko", "tachy", "age70", "fever"]),
                  ("bedside_items_no_exam", ["leuko", "age70", "como_malignancy_active", "como_diabetes"]),
                  ("bedside_items_plus_vitals", ["leuko", "age70", "como_malignancy_active", "como_diabetes", "tachy", "fever"])]:
    m = logit_binary(ed.loc[~tem, cols], ed.loc[~tem, "y"]); p = predict_binary(m, ed.loc[tem])
    dec["auroc_test"][key] = round(float(roc_auc_score(y, p)), 3)
print("DECOMPOSITION", json.dumps(dec, indent=1))

# ---- like-for-like inside the ED cohort: admitted stays with a processed note, both scores get the examination item.
f, d = load_mimic()
lat = pd.read_json(NLP, lines=True); lat = lat[lat["ok"].astype(str) == "True"].copy(); lat["hadm_id"] = lat.hadm_id.astype("int64")
lat["asymmetry"] = lat.laterality.isin(["unilateral_left", "unilateral_right"]).astype(int)
ed["hadm_id"] = pd.to_numeric(ed.hadm_id, errors="coerce")
a = ed[ed.hadm_id.notna()].copy(); a["hadm_id"] = a.hadm_id.astype("int64")
a = a.merge(lat[["hadm_id", "asymmetry"]], on="hadm_id", how="inner")
a = a.merge(f[["hadm_id", "split", "wbc_first"]], on="hadm_id", how="inner")   # inpatient cohort rows (their split label)
a["leuko_first"] = (a.wbc_first >= 10).fillna(False).astype(int)
a["alt70_complete"] = 3 * a.asymmetry + a.leuko + a.tachy + 2 * a.age70          # triage HR, first-day max WBC, note asymmetry
bed = M["bedside_model"]
X = pd.DataFrame({"asymmetry": a.asymmetry, "leuko": a.leuko_first, "age70": a.age70,
                  "como_malignancy_active": a.como_malignancy_active, "como_diabetes": a.como_diabetes})
a["p_bed"] = predict_binary(bed, X)
held = a[a.split == "test"].copy()                                              # never used to fit the Bedside coefficients
yh = held.y.to_numpy().astype(int)
au, dd = paired_boot(yh, {"bedside_tier": held.p_bed, "alt70_complete": held.alt70_complete}, [("bedside_tier", "alt70_complete")])
alt_tab = {f">= {t}": opchar(held.alt70_complete >= t, yh) for t in (3, 5)}
thr = np.unique(held.p_bed)
def at_spec(target):
    best = min(thr, key=lambda t: abs(float((held.p_bed >= t)[yh == 0].mean()) - (1 - target)))
    return {"prob_threshold": round(float(best), 3), **opchar(held.p_bed >= best, yh)}
like = {"n_admitted_with_note_in_inpatient_cohort": int(len(a)), "n_held_out": int(len(held)), "prevalence_held_out": round(float(yh.mean()), 3),
        "aurocs": au, "paired": dd, "alt70_operating": alt_tab, "bedside_matched_spec_alt70_ge5": at_spec(alt_tab[">= 5"]["spec"] / 100),
        "roc": {"bedside": roc_points(yh, held.p_bed), "alt70": roc_points(yh, held.alt70_complete)},
        "note": "Emergency department stays that were admitted and have a processed discharge summary; Bedside tier uses the inpatient-fitted coefficients and is evaluated only on patients in the inpatient test partition; ALT-70 uses triage heart rate."}
# the same, refitting the Bedside items within the ED-admitted subset (70/30 by subject) for a fit-free-of-inpatient check
rng2 = np.random.default_rng(SEED); u2 = np.unique(a.subject_id.to_numpy()); rng2.shuffle(u2)
te2 = a.subject_id.isin(set(u2[: int(0.3 * len(u2))])).to_numpy()
X2 = X.copy(); X2["y"] = a.y.values
m2 = logit_binary(X2.loc[~te2, ["asymmetry", "leuko", "age70", "como_malignancy_active", "como_diabetes"]], X2.loc[~te2, "y"])
p2 = predict_binary(m2, X2.loc[te2]); y2 = X2.loc[te2, "y"].to_numpy().astype(int)
au2, dd2 = paired_boot(y2, {"bedside_refit": p2, "alt70_complete": a.loc[te2, "alt70_complete"]}, [("bedside_refit", "alt70_complete")])
like["refit_within_ed_admitted"] = {"n_test": int(te2.sum()), "prevalence": round(float(y2.mean()), 3), "aurocs": au2, "paired": dd2,
                                    "ors": {k: (v["or"], v["or_lo"], v["or_hi"]) for k, v in m2["terms"].items()}}
print("LIKE-FOR-LIKE", json.dumps({k: v for k, v in like.items() if k != "roc"}, indent=1)[:2500])
M["ed"]["decomposition"] = dec; M["ed"]["admitted_like_for_like"] = like
OUT.write_text(json.dumps(M, indent=1)); print("merged into", OUT)
