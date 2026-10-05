"""Feature builder.

At-presentation features for the locked cohort: demographics, first-48h labs
(hospital-wide via labevents), ICU vitals where present (chartevents = ICU-only), and
non-circular comorbidity flags. Also computes the structured ALT-70 score (3 of 4
components; asymmetry comes from the laterality extraction). No culture result is read.

Leakage guards: labs/vitals restricted to the first `lab_window_hours` from admittime
(at-presentation, not post-treatment); comorbidity flags EXCLUDE the label-defining
families (venous insufficiency / DVT / gout / cellulitis); see config/features_study2.yaml.

DUA RULE: row-level feature matrix -> data/ (gitignored); only aggregate coverage is printed.
Data framing: MIMIC-IV v3.1, single-center, retrospective, BIDMC, 2008-2022.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from shared.mimic_io import from_config

COHORT_PARQUET = "data/cohorts/study2_cohort.parquet"
OUT_PARQUET = "data/cohorts/study2_features.parquet"


def _case_map(mapping: dict[str, list[int]]) -> str:
    """SQL CASE mapping itemid -> analyte name."""
    whens = []
    for analyte, ids in mapping.items():
        idlist = ",".join(str(i) for i in ids)
        whens.append(f"WHEN itemid IN ({idlist}) THEN '{analyte}'")
    return "CASE " + " ".join(whens) + " END"


def _comorbidity_exprs(comos: dict) -> str:
    exprs = []
    for name, spec in comos.items():
        cl = []
        for p in spec.get("icd9_prefix", []):
            cl.append(f"(icd_version=9 AND icd_code LIKE '{p.replace('.', '')}%')")
        for p in spec.get("icd10_prefix", []):
            cl.append(f"(icd_version=10 AND icd_code LIKE '{p}%')")
        exprs.append(f"max(CASE WHEN {' OR '.join(cl)} THEN 1 ELSE 0 END) AS como_{name}")
    return ",\n           ".join(exprs)


def build(default_cfg="config/default.yaml", feat_cfg="config/features_study2.yaml",
          cohort_parquet=COHORT_PARQUET, out_parquet=OUT_PARQUET):
    fc = yaml.safe_load(Path(feat_cfg).read_text())
    win = fc["lab_window_hours"]
    src = from_config(default_cfg)
    con = src.con
    con.execute(f"CREATE OR REPLACE VIEW cohort AS SELECT * FROM read_parquet('{cohort_parquet}')")

    base = con.execute(
        "SELECT c.subject_id, c.hadm_id, c.admittime, c.age, c.label, c.split, "
        "p.gender FROM cohort c JOIN patients p USING (subject_id)"
    ).df()

    # --- first-48h labs (hospital-wide) ---
    lab_ids = [i for ids in fc["labs"].values() for i in ids]
    lab_case = _case_map(fc["labs"])
    labs = con.execute(f"""
        WITH w AS (
          SELECT c.hadm_id, {lab_case} AS analyte, l.valuenum, l.charttime
          FROM cohort c JOIN labevents l USING (hadm_id)
          WHERE l.itemid IN ({','.join(map(str, lab_ids))}) AND l.valuenum IS NOT NULL
            AND l.charttime BETWEEN c.admittime AND c.admittime + INTERVAL {win} HOUR)
        SELECT hadm_id, analyte,
               arg_min(valuenum, charttime) AS first, min(valuenum) AS min, max(valuenum) AS max
        FROM w GROUP BY hadm_id, analyte
    """).df()
    lab_wide = labs.pivot(index="hadm_id", columns="analyte", values=["first", "min", "max"])
    lab_wide.columns = [f"{a}_{stat}" for stat, a in lab_wide.columns]
    lab_wide = lab_wide.reset_index()

    # --- ICU vitals (chartevents = ICU only) ---
    vids = fc["vitals_icu"]
    all_vit_ids = sorted({i for ids in vids.values() for i in ids})
    vitals = con.execute(f"""
        WITH w AS (
          SELECT c.hadm_id, ce.itemid, ce.valuenum, ce.charttime
          FROM cohort c JOIN chartevents ce USING (hadm_id)
          WHERE ce.itemid IN ({','.join(map(str, all_vit_ids))})
            AND ce.valuenum IS NOT NULL
            AND ce.charttime BETWEEN c.admittime AND c.admittime + INTERVAL {win} HOUR)
        SELECT hadm_id,
               max(CASE WHEN itemid={vids['heart_rate'][0]} THEN valuenum END) AS hr_max,
               arg_min(CASE WHEN itemid={vids['heart_rate'][0]} THEN valuenum END,
                       CASE WHEN itemid={vids['heart_rate'][0]} THEN charttime END) AS hr_first,
               max(CASE WHEN itemid={vids['temp_c'][0]} THEN valuenum
                        WHEN itemid={vids['temp_f'][0]} THEN (valuenum-32)*5.0/9.0 END) AS temp_c_max,
               max(CASE WHEN itemid={vids['resp_rate'][0]} THEN valuenum END) AS rr_max,
               min(CASE WHEN itemid={vids['spo2'][0]} THEN valuenum END) AS spo2_min,
               min(CASE WHEN itemid={vids['sysbp'][0]} THEN valuenum END) AS sysbp_min
        FROM w GROUP BY hadm_id
    """).df()

    # --- BMI (omr, hospital-wide; value nearest to admission) ---
    bmi = con.execute("""
        WITH b AS (
          SELECT c.hadm_id, TRY_CAST(o.result_value AS DOUBLE) bmi,
                 abs(date_diff('day', o.chartdate, c.admittime)) d
          FROM cohort c JOIN omr o USING (subject_id)
          WHERE o.result_name = 'BMI (kg/m2)' AND TRY_CAST(o.result_value AS DOUBLE) BETWEEN 10 AND 80)
        SELECT hadm_id, arg_min(bmi, d) AS bmi FROM b GROUP BY hadm_id
    """).df()

    # --- comorbidities (index admission; label families excluded) ---
    como = con.execute(f"""
        SELECT c.hadm_id,
           {_comorbidity_exprs(fc['comorbidities'])}
        FROM cohort c JOIN diagnoses_icd d USING (hadm_id) GROUP BY c.hadm_id
    """).df()
    src.close()

    df = (base.merge(lab_wide, on="hadm_id", how="left")
              .merge(vitals, on="hadm_id", how="left")
              .merge(bmi, on="hadm_id", how="left")
              .merge(como, on="hadm_id", how="left"))
    for c in df.columns:
        if c.startswith("como_"):
            df[c] = df[c].fillna(0).astype(int)
    df["female"] = (df.gender == "F").astype(int)
    df["has_icu_hr"] = df.hr_max.notna().astype(int)

    # --- structured ALT-70 (3 of 4 components; asymmetry pending NLP) ---
    a = fc["alt70"]
    pt = a["points"]
    df["alt70_age_pts"] = (df.age >= a["age_threshold"]).astype(int) * pt["age"]
    df["alt70_leuko_pts"] = (df.wbc_max >= a["leukocytosis_wbc"]).fillna(False).astype(int) * pt["leukocytosis"]
    df["alt70_tachy_pts"] = (df.hr_max >= a["tachycardia_hr"]).fillna(False).astype(int) * pt["tachycardia"]
    df["alt70_structured"] = df.alt70_age_pts + df.alt70_leuko_pts + df.alt70_tachy_pts  # max 4 (asymmetry=3 pending)

    Path(out_parquet).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)
    return df


if __name__ == "__main__":
    df = build()
    print(f"feature matrix: {df.shape[0]:,} patients x {df.shape[1]} cols")
    feats = [c for c in df.columns if c not in ("subject_id", "hadm_id", "admittime", "gender")]
    cov = (df[feats].notna().mean() * 100).round(1).sort_values()
    print("\n=== feature coverage (% non-null), lowest first ===")
    print(cov.head(12).to_string())
    print("\nlabel by split:")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
