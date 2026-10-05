"""MIMIC-IV-ED variant using the CELTIC Bedside tier's own culture-free items (no asymmetry, which needs a note):
leukocytosis (first WBC), age >=70, active malignancy, diabetes, with comorbidities taken from the hospital diagnosis
codes of admissions dated on or before the ED visit (+2 days, to include the linked admission). Also an ED-extended
variant adding triage tachycardia and fever. Same 70/30 subject split and the same ALT-70 comparator as the main run.
Merges results into celtic_noculture_metrics.json under ed["bedside_items"] and ed["bedside_items_plus_vitals"].
Aggregate-only. Run from analysis/:  PYTHONPATH=. python study2_cellulitis/celtic_noculture_ed2.py
"""
from __future__ import annotations
import json
from pathlib import Path
import duckdb, numpy as np, pandas as pd, yaml
from study2_cellulitis.celtic_noculture import (ED, H, OUT, SEED, logit_binary, opchar, paired_boot, predict_binary,
                                                 roc_points)

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
# comorbidities from hospital diagnosis codes of admissions on or before the ED visit (+2 days for the linked admission)
def expr(name, spec):
    cl = [f"(d.icd_version = 9 AND d.icd_code LIKE '{p}%')" for p in spec.get("icd9_prefix", [])]
    cl += [f"(d.icd_version = 10 AND d.icd_code LIKE '{p}%')" for p in spec.get("icd10_prefix", [])]
    return f"max(CASE WHEN {' OR '.join(cl)} THEN 1 ELSE 0 END) AS como_{name}"
sel = ", ".join(expr(n, s) for n, s in comos.items() if n in ("diabetes", "chf", "ckd", "obesity", "malignancy_active"))
con.execute(f"""CREATE TEMP TABLE como AS SELECT c.stay_id, {sel}
    FROM cohort c JOIN read_csv_auto('{H}/admissions.csv.gz') a ON a.subject_id = c.subject_id AND a.admittime <= c.intime + INTERVAL 2 DAY
    JOIN read_csv_auto('{H}/diagnoses_icd.csv.gz', types={{'icd_code': 'VARCHAR'}}) d ON d.hadm_id = a.hadm_id
    GROUP BY c.stay_id""")
ed = con.execute("""SELECT c.*, v.hr, v.temp_f, w.wbc, k.* EXCLUDE (stay_id)
    FROM cohort c LEFT JOIN vit v USING(stay_id) LEFT JOIN wbc w USING(stay_id) LEFT JOIN como k USING(stay_id)""").df()
con.close()
for c in [c for c in ed.columns if c.startswith("como_")]:
    ed[c] = ed[c].fillna(0).astype(int)
ed["age70"] = (ed.age >= 70).astype(int); ed["tachy"] = (ed.hr >= 90).fillna(False).astype(int)
ed["leuko"] = (ed.wbc >= 10).fillna(False).astype(int); ed["fever"] = (ed.temp_f >= 100.4).fillna(False).astype(int)
ed["alt70"] = 2 * ed.age70 + ed.leuko + ed.tachy
rng = np.random.default_rng(SEED); uniq = np.unique(ed.subject_id.to_numpy()); rng.shuffle(uniq)
tem = ed.subject_id.isin(set(uniq[: int(0.3 * len(uniq))])).to_numpy()
y = ed.loc[tem, "y"].to_numpy().astype(int); alt = ed.loc[tem, "alt70"].to_numpy()
res = {"comorbidity_prevalence_pct": {c: round(100 * float(ed[c].mean()), 1) for c in ed.columns if c.startswith("como_")}}
for key, cols in [("bedside_items", ["leuko", "age70", "como_malignancy_active", "como_diabetes"]),
                  ("bedside_items_plus_vitals", ["leuko", "age70", "como_malignancy_active", "como_diabetes", "tachy", "fever"])]:
    m = logit_binary(ed.loc[~tem, cols], ed.loc[~tem, "y"])
    p = predict_binary(m, ed.loc[tem])
    a, d = paired_boot(y, {"celtic": p, "alt70": alt}, [("celtic", "alt70")])
    alt_tab = {f">= {t}": opchar(alt >= t, y) for t in range(1, 5)}
    thr = np.unique(p)
    def at_spec(target):
        best = min(thr, key=lambda t: abs(float((p >= t)[y == 0].mean()) - (1 - target)))
        return {"prob_threshold": round(float(best), 3), **opchar(p >= best, y)}
    res[key] = {"items": cols, "model": m, "aurocs_test": a, "paired": d, "roc": roc_points(y, p),
                "celtic_matched_spec_alt70_ge3": at_spec(alt_tab[">= 3"]["spec"] / 100), "celtic_high_spec_90": at_spec(0.90),
                "alt70_operating": alt_tab, "n_test": int(tem.sum()), "test_prevalence": round(float(y.mean()), 3)}
    print(key, a, d["celtic_minus_alt70"]["delta"], d["celtic_minus_alt70"]["lo"], d["celtic_minus_alt70"]["hi"], d["celtic_minus_alt70"]["delong"])
    print("  ORs:", {c: (t["or"], t["or_lo"], t["or_hi"]) for c, t in m["terms"].items()})
M["ed"].update(res)
OUT.write_text(json.dumps(M, indent=1))
print("comorbidity prevalence in ED cohort:", res["comorbidity_prevalence_pct"])
