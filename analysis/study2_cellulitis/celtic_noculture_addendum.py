"""Exact cellulitis counts per band (CELTIC Bedside probability bands and original ALT-70 bands) for Table 3,
merged into celtic_noculture_metrics.json under "band_counts". Aggregate-only. Cheap (no raw-file scans).
Run from analysis/:  PYTHONPATH=. python study2_cellulitis/celtic_noculture_addendum.py
"""
from __future__ import annotations
import json
import numpy as np
from study2_cellulitis.celtic_noculture import OUT, load_mimic, predict_binary

M = json.loads(OUT.read_text())
f, d = load_mimic()
te = d[d.split == "test"].copy()
p = predict_binary(M["bedside_model"], te)
y = te.y.to_numpy()
t_lo, t_hi = M["bands"]["thresholds"]["rule_out_below"], M["bands"]["thresholds"]["rule_in_at_or_above"]
bc = {"bedside": {}, "alt70": {}}
for key, m in [("rule_out", p < t_lo), ("indeterminate", (p >= t_lo) & (p < t_hi)), ("rule_in", p >= t_hi)]:
    bc["bedside"][key] = {"n": int(m.sum()), "cellulitis": int(y[m].sum()), "mimic": int((y[m] == 0).sum())}
alt = te.alt70_complete.to_numpy()
for name, lo, hi in [("0-2 (rule-out)", -1, 2), ("3-4 (indeterminate)", 3, 4), ("5-7 (rule-in)", 5, 7)]:
    m = (alt >= lo) & (alt <= hi)
    bc["alt70"][name] = {"n": int(m.sum()), "cellulitis": int(y[m].sum()), "mimic": int((y[m] == 0).sum())}
# agreement between probability bands and the hand score's natural bands
M["band_counts"] = bc
OUT.write_text(json.dumps(M, indent=1))
print(json.dumps(bc, indent=1))
