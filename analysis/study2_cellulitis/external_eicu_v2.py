"""eICU-CRD external cohort and structured features, used by celtic_noculture.py.

eICU has no notes (so no asymmetry item), so the external test uses a structured model only: age, sex,
the first-48-hour laboratory panel, vital signs and comorbidities, harmonized to the MIMIC-IV feature
names in SHARED. The cohort is a broad cellulitis vs pseudocellulitis spectrum taken from the eICU
diagnosis table (first stay per patient, adults). No culture result is read.

DUA RULE: aggregate metrics only.
"""
from __future__ import annotations

from pathlib import Path

EICU = str(Path("~/eicu-crd-2.0").expanduser())

# MIMIC feature -> eICU labname / handling
SHARED = ["age", "female", "wbc_max", "creatinine_max", "sodium_min", "bicarbonate_min",
          "hemoglobin_min", "platelet_min", "glucose_max", "albumin_min",
          "hr_max", "temp_c_max", "rr_max", "spo2_min",
          "como_diabetes", "como_chf", "como_ckd", "como_malignancy_active"]


def eicu_cohort_features(con):
    con.execute(f"CREATE OR REPLACE VIEW dx AS SELECT * FROM read_csv_auto('{EICU}/diagnosis.csv.gz', ignore_errors=true)")
    con.execute(f"CREATE OR REPLACE VIEW pt AS SELECT * FROM read_csv_auto('{EICU}/patient.csv.gz')")
    # broad cohort: cellulitis vs pseudocellulitis spectrum
    con.execute("""CREATE OR REPLACE TEMP TABLE lab_flag AS
      SELECT patientunitstayid stay,
        max(CASE WHEN icd9code LIKE '%682%' OR icd9code LIKE '%L03%' OR lower(diagnosisstring) LIKE '%cellulitis%' THEN 1 ELSE 0 END) has_case,
        max(CASE WHEN icd9code LIKE '%4534%' OR icd9code LIKE '%I82.4%' OR lower(diagnosisstring) LIKE '%deep vein thromb%'
              OR lower(diagnosisstring) LIKE '%stasis%' OR lower(diagnosisstring) LIKE '%venous%' OR icd9code LIKE '%274%'
              OR lower(diagnosisstring) LIKE '%gout%' OR lower(diagnosisstring) LIKE '%dermatitis%' OR lower(diagnosisstring) LIKE '%eczema%'
              OR lower(diagnosisstring) LIKE '%lymphedema%' OR lower(diagnosisstring) LIKE '%thrombophleb%'
              OR lower(diagnosisstring) LIKE '%erythema nodosum%' OR lower(diagnosisstring) LIKE '%vasculitis%' THEN 1 ELSE 0 END) has_mimic,
        max(CASE WHEN lower(diagnosisstring) LIKE '%diabetes%' THEN 1 ELSE 0 END) diab,
        max(CASE WHEN lower(diagnosisstring) LIKE '%heart failure%' OR icd9code LIKE '%428%' THEN 1 ELSE 0 END) chf,
        max(CASE WHEN lower(diagnosisstring) LIKE '%chronic kidney%' OR lower(diagnosisstring) LIKE '%renal failure%' THEN 1 ELSE 0 END) ckd,
        max(CASE WHEN lower(diagnosisstring) LIKE '%cancer%' OR lower(diagnosisstring) LIKE '%malignan%' OR lower(diagnosisstring) LIKE '%carcinoma%' OR lower(diagnosisstring) LIKE '%metasta%' THEN 1 ELSE 0 END) malig
      FROM dx GROUP BY patientunitstayid""")
    con.execute("""CREATE OR REPLACE TEMP TABLE cohort AS
      WITH lab AS (SELECT * FROM lab_flag WHERE has_case+has_mimic=1),
      p AS (SELECT patientunitstayid stay, uniquepid,
              CASE WHEN age='> 89' THEN 90 ELSE TRY_CAST(age AS INTEGER) END age,
              CASE WHEN gender='Female' THEN 1 ELSE 0 END female, hospitaladmitoffset
            FROM pt),
      j AS (SELECT l.stay, l.has_case y, l.diab, l.chf, l.ckd, l.malig, p.age, p.female,
              row_number() OVER (PARTITION BY p.uniquepid ORDER BY p.hospitaladmitoffset) rn
            FROM lab l JOIN p USING(stay) WHERE p.age>=18)
      SELECT stay, y, age, female, diab, chf, ckd, malig FROM j WHERE rn=1""")

    labmap = {"wbc_max": ("WBC x 1000", "max"), "creatinine_max": ("creatinine", "max"),
              "sodium_min": ("sodium", "min"), "bicarbonate_min": ("bicarbonate", "min"),
              "hemoglobin_min": ("Hgb", "min"), "platelet_min": ("platelets x 1000", "min"),
              "glucose_max": ("glucose", "max"), "albumin_min": ("albumin", "min")}
    labsel = ",".join(f"{agg}(CASE WHEN labname='{ln}' THEN TRY_CAST(labresult AS DOUBLE) END) AS {feat}"
                      for feat, (ln, agg) in labmap.items())
    labs = con.execute(f"""SELECT patientunitstayid stay, {labsel}
        FROM read_csv_auto('{EICU}/lab.csv.gz') WHERE labresultoffset BETWEEN 0 AND 2880
        AND patientunitstayid IN (SELECT stay FROM cohort) GROUP BY 1""").df()
    vit = con.execute(f"""SELECT patientunitstayid stay,
          max(CASE WHEN heartrate BETWEEN 20 AND 250 THEN heartrate END) hr_max,
          max(CASE WHEN temperature BETWEEN 30 AND 45 THEN temperature END) temp_c_max,
          max(CASE WHEN respiration BETWEEN 4 AND 60 THEN respiration END) rr_max,
          min(CASE WHEN sao2 BETWEEN 40 AND 100 THEN sao2 END) spo2_min
        FROM read_csv_auto('{EICU}/vitalPeriodic.csv.gz') WHERE observationoffset BETWEEN 0 AND 2880
        AND patientunitstayid IN (SELECT stay FROM cohort) GROUP BY 1""").df()
    coh = con.execute("SELECT * FROM cohort").df()
    df = coh.merge(labs, on="stay", how="left").merge(vit, on="stay", how="left")
    df["como_diabetes"] = df.diab; df["como_chf"] = df.chf; df["como_ckd"] = df.ckd; df["como_malignancy_active"] = df.malig
    return df
