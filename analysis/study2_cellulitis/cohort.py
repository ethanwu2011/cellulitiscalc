"""Cohort builder: cellulitis vs pseudocellulitis.

Config-driven (config/default.yaml + config/cohorts.yaml). Emits a CONSORT-style
exclusion flow and patient-level splits.

Design: unit = one index (earliest) admission per patient; admissions coded for both a case and
a mimic are excluded; adults >=18; split on subject_id.

DUA RULE: the row-level cohort (subject_id/hadm_id) is written only under data/
(gitignored). The committed artifact is the aggregate CONSORT log (counts only).

Data framing: MIMIC-IV v3.1, single-center, retrospective, BIDMC, 2008-2022.
"""
from __future__ import annotations

import datetime as dt
import subprocess
from pathlib import Path

import yaml

from shared.cohorts.consort import ConsortLog
from shared.cohorts.splits import patient_level_split
from shared.mimic_io import from_config


def _codes_to_sql(spec: dict) -> str:
    """Build a SQL boolean over (icd_version, icd_code) from a config code spec."""
    c: list[str] = []
    for p in spec.get("icd9_prefix", []):
        c.append(f"(icd_version=9 AND icd_code LIKE '{p}%')")
    for e in spec.get("icd9", []):
        c.append(f"(icd_version=9 AND icd_code='{e}')")
    for p in spec.get("icd10_prefix", []):
        c.append(f"(icd_version=10 AND icd_code LIKE '{p}%')")
    for e in spec.get("icd10", []):
        c.append(f"(icd_version=10 AND icd_code='{e}')")
    if spec.get("icd10_site_rule"):
        c.append(f"(icd_version=10 AND {spec['icd10_site_rule']})")
    if not c:
        raise ValueError(f"empty code spec: {spec}")
    return "(" + " OR ".join(c) + ")"


def build(
    default_cfg: str = "config/default.yaml",
    cohorts_cfg: str = "config/cohorts.yaml",
    out_dir: str = "data/cohorts",
    seed: int | None = None,
    unit: str = "index",                         # "index" (primary) | "all" (sensitivity)
    cohort_filename: str = "study2_cohort.parquet",
    consort_path: str = "study2_cellulitis/cohort_consort.json",
):
    cfg = yaml.safe_load(Path(default_cfg).read_text())
    coh = yaml.safe_load(Path(cohorts_cfg).read_text())["study2_cellulitis"]
    seed = cfg["seed"] if seed is None else seed

    case_sql = _codes_to_sql(coh["case"])
    mimic_sql = "(" + " OR ".join(_codes_to_sql(s) for s in coh["mimics"].values()) + ")"

    src = from_config(default_cfg)
    n_all_adm, n_all_pt = src.count("admissions"), src.count("patients")
    sql = f"""
    WITH flagged AS (
      SELECT hadm_id, subject_id,
             max(CASE WHEN {case_sql}  THEN 1 ELSE 0 END) AS has_case,
             max(CASE WHEN {mimic_sql} THEN 1 ELSE 0 END) AS has_mimic
      FROM diagnoses_icd GROUP BY hadm_id, subject_id)
    SELECT f.subject_id, f.hadm_id, a.admittime,
           p.anchor_age + (year(a.admittime) - p.anchor_year) AS age,
           CASE WHEN has_case=1 AND has_mimic=0 THEN 'positive'
                WHEN has_case=0 AND has_mimic=1 THEN 'negative'
                ELSE 'ambiguous' END AS label
    FROM flagged f JOIN admissions a USING (hadm_id, subject_id)
                   JOIN patients   p USING (subject_id)
    WHERE has_case=1 OR has_mimic=1
    """
    df = src.query(sql)
    src.close()

    npt = lambda d: d.subject_id.nunique()  # noqa: E731
    log = ConsortLog(
        "CELTIC: cellulitis vs pseudocellulitis (MIMIC-IV v3.1; single-center, "
        "retrospective, BIDMC 2008-2022)"
    )
    log.add("All MIMIC-IV hospital admissions", n_all_adm, n_all_pt)
    log.add("Coded cellulitis OR a pseudocellulitis mimic", len(df), npt(df))
    df = df[df.label != "ambiguous"]
    log.add("Exclude same-admission ambiguous (both-coded)", len(df), npt(df),
            "excluded from the analysis")
    df = df[df.age >= 18]
    log.add("Adults >=18", len(df), npt(df))
    if unit == "index":
        df = (df.sort_values(["subject_id", "admittime", "hadm_id"])
                .groupby("subject_id", as_index=False).first())
        log.add("One index (earliest) admission per patient", len(df), npt(df), "analytic cohort")
    else:
        log.add("All eligible admissions (subject-grouped CV)", len(df), npt(df),
                "analytic cohort (sensitivity)")

    df = patient_level_split(
        df, test_frac=cfg["split"]["test_frac"], val_frac=cfg["split"]["val_frac"], seed=seed)

    # QC (non-negotiable)
    if unit == "index":
        assert df.subject_id.is_unique, "duplicate subject_id in final cohort"
    assert set(df.label.unique()) <= {"positive", "negative"}, "unexpected label"
    per_subj = df.groupby("subject_id")["split"].nunique()   # no patient straddles splits (both units)
    assert (per_subj == 1).all(), "subject_id leak across splits"

    git_hash = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    log.meta = {
        "seed": seed,
        "git_hash": git_hash,
        "built_at": dt.datetime.now().isoformat(timespec="seconds"),
        "unit": ("one index (earliest) admission per patient" if unit == "index"
                 else "all eligible admissions (subject-grouped CV)"),
        "ambiguous_policy": "excluded",
        "class_balance": {k: int(v) for k, v in df.label.value_counts().items()},
        "split_label_counts": {
            f"{s}/{l}": int(((df.split == s) & (df.label == l)).sum())
            for s in ("train", "val", "test") for l in ("positive", "negative")},
    }

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / cohort_filename, index=False)  # row-level -> gitignored
    log.to_json(consort_path)                            # aggregate -> committable
    return df, log


if __name__ == "__main__":
    df, log = build()
    print(log.render())
    print("\nclass balance:", log.meta["class_balance"])
    print("split x label:")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0).to_string())
