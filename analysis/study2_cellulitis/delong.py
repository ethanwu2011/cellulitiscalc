"""Fast DeLong test for the difference between two correlated ROC AUCs (Sun & Xu 2014).

delong_roc_test(y, s1, s2) -> {auc1, auc2, delta, z, p_delong} for AUC(s1) - AUC(s2) on the same samples.
Used as an analytic cross-check on the paired bootstrap CIs.
"""
from __future__ import annotations
import numpy as np
from scipy import stats


def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2


def _fast_delong(preds, m):
    n = preds.shape[1] - m; k = preds.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _midrank(preds[r, :m]); ty[r] = _midrank(preds[r, m:]); tz[r] = _midrank(preds[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    cov = sx / m + sy / n
    return aucs, np.atleast_2d(cov)


def delong_roc_test(y, s1, s2):
    y = np.asarray(y); s1 = np.asarray(s1, float); s2 = np.asarray(s2, float)
    order = (-y).argsort(kind="mergesort")          # positives first
    m = int(y[order].sum())
    aucs, cov = _fast_delong(np.vstack([s1[order], s2[order]]), m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else np.inf
    p = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else 0.0
    return {"auc1": round(float(aucs[0]), 3), "auc2": round(float(aucs[1]), 3),
            "delta": round(float(aucs[0] - aucs[1]), 3), "z": round(float(z), 2),
            "p_delong": p if p >= 1e-4 else "<0.0001"}
