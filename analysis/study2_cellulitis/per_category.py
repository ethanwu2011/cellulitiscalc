"""Mimic categories and per-condition admission lookup, used by celtic_noculture.py for the
per-mimic-category discrimination analysis (cellulitis vs each category of mimic on the held-out test set).

Each mimic admission is assigned to its coded conditions from the definitions in config/cohorts.yaml, and the
conditions are grouped into four categories (vascular, crystal, inflammatory-dermatologic, traumatic-mechanical).
"""
from __future__ import annotations
from pathlib import Path
import duckdb, pandas as pd, yaml

DIAG = str(Path("~/mimiciv-3.1/hosp/diagnoses_icd.csv.gz").expanduser())
COH = "config/cohorts.yaml"
CATEGORY = {
    "vascular (DVT/venous)": ["dvt_lower_limb", "superficial_thrombophleb", "chronic_venous_insuff", "lymphedema"],
    "crystal (gout/pseudogout)": ["gout_acute_lowerlimb", "pseudogout"],
    "inflammatory-dermatologic": ["stasis_dermatitis", "contact_dermatitis", "eczema_other_dermatitis",
                                   "erythema_nodosum", "panniculitis", "cutaneous_vasculitis",
                                   "pyoderma_gangrenosum", "erythema_migrans"],
    "traumatic-mechanical": ["ruptured_baker_cyst", "bursitis_lower_limb", "insect_bite", "contusion_hematoma_ll"],
}


def condition_hadms(hadms):
    spec = yaml.safe_load(Path(COH).read_text())["study2_cellulitis"]["mimics"]
    con = duckdb.connect()
    dx = con.execute(f"SELECT hadm_id, icd_code, icd_version FROM read_csv_auto('{DIAG}') "
                     f"WHERE hadm_id IN ({','.join(map(str, hadms))})").fetch_df()
    dx["icd_code"] = dx.icd_code.astype(str)
    out = {}
    for name, s in spec.items():
        m = pd.Series(False, index=dx.index)
        for p in s.get("icd9_prefix", []):
            m |= (dx.icd_version == 9) & dx.icd_code.str.startswith(p.replace(".", ""))
        for p in s.get("icd10_prefix", []):
            m |= (dx.icd_version == 10) & dx.icd_code.str.startswith(p)
        for c in s.get("icd9", []):
            m |= (dx.icd_version == 9) & (dx.icd_code == c)
        for c in s.get("icd10", []):
            m |= (dx.icd_version == 10) & (dx.icd_code == c)
        if "icd10_site_rule" in s:  # gout acute lower-limb
            m |= (dx.icd_version == 10) & dx.icd_code.str.startswith("M10") & dx.icd_code.str[4:5].isin(["5", "6", "7"])
        out[name] = set(dx.loc[m, "hadm_id"])
    return out
