// CELTIC model arithmetic. No DOM, no side effects. Every number here is copied from the analysis output
// celtic_noculture_metrics.json, written by analysis/study2_cellulitis/celtic_noculture.py and its step 5 add-ons
// (no culture result, seed 20260707).
// Logistic regression: logit = intercept + sum(coef_i * x_i); p = 1 / (1 + exp(-logit)).
// Inputs are binary indicators (0/1). The page asks every item as yes or no and the label carries the cut point;
// threshold() below holds the cut points exactly as in the paper, for numeric entry.

export const BANDS = { ruleOutBelow: 0.3, ruleInAtOrAbove: 0.65 };

export const ITEMS = {
  asymmetry: { label: "Unilateral involvement", short: "Unilateral", hint: "One limb involved. Bilateral or symmetric involvement favors a mimic." },
  leuko: { label: "Leukocytosis (first WBC at or above 10\u00a0\u00d710\u00b3/\u00b5L)", short: "WBC \u226510", hint: "", threshold: { unit: "\u00d710\u00b3/\u00b5L", ge: 10, min: 0.5, max: 200 } },
  age70: { label: "Age 70 years or older", short: "Age \u226570", hint: "Counts against cellulitis. Older inpatients more often have a mimic.", threshold: { unit: "years", ge: 70, min: 18, max: 120 } },
  como_malignancy_active: { label: "Active malignancy", short: "Malignancy", hint: "Current cancer, solid or hematologic." },
  como_diabetes: { label: "Diabetes", short: "Diabetes", hint: "Type 1 or type 2." },
  crp_hi: { label: "C-reactive protein (at or above 50\u00a0mg/L)", short: "CRP \u226550", hint: "Use the first value of the admission.", threshold: { unit: "mg/L", ge: 50, min: 0, max: 800 } },
  neut_hi: { label: "Absolute neutrophils (at or above 7.5\u00a0\u00d710\u00b3/\u00b5L)", short: "Neutrophils \u22657.5", hint: "Use the first differential of the admission.", threshold: { unit: "\u00d710\u00b3/\u00b5L", ge: 7.5, min: 0, max: 150 } },
  como_chf: { label: "Heart failure", short: "Heart failure", hint: "Any diagnosis of congestive heart failure." },
  como_ckd: { label: "Chronic kidney disease", short: "CKD", hint: "Any stage." },
  como_obesity: { label: "Obesity", short: "Obesity", hint: "A diagnosis of obesity, or body mass index of 30 or more.", threshold: { unit: "kg/m\u00b2", ge: 30, min: 10, max: 100 } },
};

export const TIERS = {
  bedside: {
    name: "Bedside", inputs: 5, auroc: 0.721, aurocLo: 0.704, aurocHi: 0.737,
    intercept: -1.0763,
    terms: {
    asymmetry: { coef: 1.5301, or: 4.62, orLo: 4.26, orHi: 5.0, points: 3 },
    leuko: { coef: 0.187, or: 1.21, orLo: 1.11, orHi: 1.3, points: 0 },
    age70: { coef: -0.6539, or: 0.52, orLo: 0.48, orHi: 0.56, points: -1 },
    como_malignancy_active: { coef: -0.5615, or: 0.57, orLo: 0.5, orHi: 0.65, points: -1 },
    como_diabetes: { coef: 0.3985, or: 1.49, orLo: 1.37, orHi: 1.61, points: 1 }
    },
  },
  rapid: {
    name: "Rapid", inputs: 10, auroc: 0.727, aurocLo: 0.711, aurocHi: 0.744,
    intercept: -1.0731,
    terms: {
    asymmetry: { coef: 1.5058, or: 4.51, orLo: 4.16, orHi: 4.89 },
    leuko: { coef: 0.1629, or: 1.18, orLo: 1.09, orHi: 1.27 },
    age70: { coef: -0.5977, or: 0.55, orLo: 0.51, orHi: 0.6 },
    como_malignancy_active: { coef: -0.616, or: 0.54, orLo: 0.48, orHi: 0.61 },
    como_diabetes: { coef: 0.4874, or: 1.63, orLo: 1.5, orHi: 1.77 },
    crp_hi: { coef: 0.4688, or: 1.6, orLo: 1.33, orHi: 1.93 },
    neut_hi: { coef: 0.2157, or: 1.24, orLo: 1.14, orHi: 1.35 },
    como_chf: { coef: -0.2884, or: 0.75, orLo: 0.67, orHi: 0.83 },
    como_ckd: { coef: -0.0768, or: 0.93, orLo: 0.84, orHi: 1.03 },
    como_obesity: { coef: -0.2828, or: 0.75, orLo: 0.67, orHi: 0.84 }
    },
  },
};

export const REFERENCE = {
  standard: { inputs: 37, auroc: 0.747 },
  comprehensive: { inputs: 94, auroc: 0.756 },
  alt70: { auroc: 0.604, lo: 0.587, hi: 0.622 },
  pairedDifference: { delta: 0.116, lo: 0.097, hi: 0.135 },
  cohort: { n: 24852, withNote: 17024, test: 3446, testPrevalence: 0.454, cellulitis: 11788, mimic: 13064 },
  eicu: { celtic: 0.69, alt70: 0.54, n: 2536 },
  ed: { celtic: 0.563, alt70: 0.517, n: 3314 },
  bandYield: { ruleOut: 31.5, ruleOutMimic: 79.5, ruleOutLR: 0.31, indeterminate: 49.3, ruleIn: 19.2, ruleInCellulitis: 68.2, ruleInLR: 2.58 },
  calibration: { brier: 0.211, slope: 1.0, intercept: -0.009 },
};

export function sigmoid(z) { return 1 / (1 + Math.exp(-z)); }

/** Apply the paper's threshold to a typed value; returns 0/1, or null when the value is missing or out of range. */
export function threshold(key, value) {
  const t = ITEMS[key].threshold;
  if (!t) throw new Error(`${key} has no continuous form`);
  if (value === null || value === undefined || value === "" || Number.isNaN(Number(value))) return null;
  const v = Number(value);
  if (v < t.min || v > t.max) return null;
  return v >= t.ge ? 1 : 0;
}

/** Validate an answer set for a tier. Returns { ok, missing: [keys] }. */
export function validate(tierKey, answers) {
  const tier = TIERS[tierKey];
  const missing = Object.keys(tier.terms).filter((k) => answers[k] !== 0 && answers[k] !== 1);
  return { ok: missing.length === 0, missing };
}

/**
 * Predict for a tier. answers: { key: 0|1 } for every term of the tier.
 * Returns logit, probability, band, and the signed contribution of each term (coef * x), sorted by |size| descending.
 */
export function predict(tierKey, answers) {
  const tier = TIERS[tierKey];
  const v = validate(tierKey, answers);
  if (!v.ok) return { ok: false, missing: v.missing };
  let logit = tier.intercept;
  const contributions = [];
  for (const [key, term] of Object.entries(tier.terms)) {
    const x = answers[key];
    const c = term.coef * x;
    logit += c;
    contributions.push({ key, present: x === 1, coef: term.coef, contribution: c, or: term.or });
  }
  contributions.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
  const p = sigmoid(logit);
  const band = p < BANDS.ruleOutBelow ? "rule-out" : p >= BANDS.ruleInAtOrAbove ? "rule-in" : "indeterminate";
  const base = sigmoid(tier.intercept);
  const points = tierKey === "bedside"
    ? Object.entries(tier.terms).reduce((s, [k, t]) => s + (t.points ?? 0) * answers[k], 0)
    : null;
  return { ok: true, logit, probability: p, baseline: base, band, contributions, points };
}

/** Percent with two significant figures for display (0.2543 -> "25"), never over-precise. */
export function formatPercent(p) {
  const pct = p * 100;
  if (pct < 10) return pct.toFixed(1);
  return Math.round(pct).toString();
}
