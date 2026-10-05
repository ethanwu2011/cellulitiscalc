"""Patient-level train/val/test splits.

Split unit = subject_id, NEVER hadm_id (non-negotiable: no patient's encounters may
straddle train/test). Stratified by each subject's label, seeded for reproducibility.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def patient_level_split(
    df: pd.DataFrame,
    *,
    subject_col: str = "subject_id",
    label_col: str = "label",
    test_frac: float = 0.2,
    val_frac: float = 0.2,
    seed: int = 0,
) -> pd.DataFrame:
    """Assign each subject to train/val/test, stratified by the subject's label.

    Returns a copy of `df` with an added 'split' column. Guarantees no subject appears
    in more than one split (asserted).
    """
    # one label per subject (majority) for stratification
    subj = (
        df.groupby(subject_col)[label_col]
        .agg(lambda s: s.value_counts().index[0])
        .reset_index()
    )
    rng = np.random.default_rng(seed)
    assignment: dict = {}
    for lab, grp in subj.groupby(label_col):
        ids = np.array(grp[subject_col].to_numpy(), copy=True)
        rng.shuffle(ids)
        n = len(ids)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        for i, sid in enumerate(ids):
            assignment[sid] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")

    out = df.copy()
    out["split"] = out[subject_col].map(assignment)
    leak = out.groupby(subject_col)["split"].nunique()
    assert (leak == 1).all(), "subject_id leak across splits"
    return out
