import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { predict, threshold, validate, sigmoid, TIERS, ITEMS, BANDS, formatPercent } from "../js/model.js";

const profiles = JSON.parse(readFileSync(new URL("./paper_profiles.json", import.meta.url)));

test("every Bedside item profile from the paper (eTable 4) reproduces the published predicted probability to 0.1%", () => {
  for (const c of profiles) {
    const r = predict("bedside", c.answers);
    assert.equal(r.ok, true);
    assert.ok(Math.abs(r.probability * 100 - c.expected_pct) <= 0.051, `${JSON.stringify(c.answers)} -> ${(r.probability*100).toFixed(2)} vs ${c.expected_pct}`);
  }
  assert.ok(profiles.length >= 20);
});

test("worked example by hand: unilateral + diabetes + age 70 or older", () => {
  const r = predict("bedside", { asymmetry: 1, leuko: 0, age70: 1, como_malignancy_active: 0, como_diabetes: 1 });
  const logit = -1.0763 + 1.5301 + 0 - 0.6539 + 0 + 0.3985;
  assert.ok(Math.abs(r.logit - logit) < 1e-9);
  assert.ok(Math.abs(r.probability - 1 / (1 + Math.exp(-logit))) < 1e-12);
  assert.equal(formatPercent(r.probability), "55");
  assert.equal(r.band, "indeterminate");
  assert.equal(r.points, 3 + 0 - 1 + 0 + 1);
});

test("no findings gives the cohort baseline and rule-out band", () => {
  const r = predict("bedside", { asymmetry: 0, leuko: 0, age70: 0, como_malignancy_active: 0, como_diabetes: 0 });
  assert.ok(Math.abs(r.probability - sigmoid(TIERS.bedside.intercept)) < 1e-12);
  assert.equal(r.band, "rule-out");
  assert.equal(r.points, 0);
});

test("unilateral + diabetes reaches the rule-in band; contributions sort by size and sum to the logit", () => {
  const r = predict("bedside", { asymmetry: 1, leuko: 1, age70: 0, como_malignancy_active: 0, como_diabetes: 1 });
  assert.equal(r.band, "rule-in");
  assert.ok(r.probability >= BANDS.ruleInAtOrAbove);
  const sum = r.contributions.reduce((s, c) => s + c.contribution, 0);
  assert.ok(Math.abs(TIERS.bedside.intercept + sum - r.logit) < 1e-12);
  for (let i = 1; i < r.contributions.length; i++) assert.ok(Math.abs(r.contributions[i-1].contribution) >= Math.abs(r.contributions[i].contribution));
  assert.equal(r.contributions[0].key, "asymmetry");
});

test("Rapid tier: all ten items entered, CRP and neutrophils push up, heart failure and obesity push down", () => {
  const yes = { asymmetry: 1, leuko: 1, age70: 0, como_malignancy_active: 0, como_diabetes: 0, crp_hi: 1, neut_hi: 1, como_chf: 0, como_ckd: 0, como_obesity: 0 };
  const r = predict("rapid", yes);
  const expected = 1 / (1 + Math.exp(-(-1.0731 + 1.5058 + 0.1629 + 0.4688 + 0.2157)));
  assert.ok(Math.abs(r.probability - expected) < 1e-12);
  assert.ok(r.contributions.find((c) => c.key === "crp_hi").contribution > 0);
  const r2 = predict("rapid", { ...yes, como_chf: 1, como_obesity: 1 });
  assert.ok(r2.probability < r.probability);
});

test("thresholds match the paper and reject out-of-range or missing values", () => {
  assert.equal(threshold("leuko", 10), 1);
  assert.equal(threshold("leuko", 9.9), 0);
  assert.equal(threshold("age70", 70), 1);
  assert.equal(threshold("age70", 69), 0);
  assert.equal(threshold("crp_hi", 50), 1);
  assert.equal(threshold("neut_hi", 7.5), 1);
  assert.equal(threshold("como_obesity", 30), 1);
  assert.equal(threshold("age70", ""), null);
  assert.equal(threshold("age70", 250), null);
  assert.equal(threshold("leuko", -1), null);
});

test("missing answers are reported, never silently zeroed", () => {
  const v = validate("bedside", { asymmetry: 1 });
  assert.equal(v.ok, false);
  assert.deepEqual(v.missing, ["leuko", "age70", "como_malignancy_active", "como_diabetes"]);
  assert.equal(predict("bedside", { asymmetry: 1 }).ok, false);
});

test("age and leukocytosis are asked as yes or no, so their labels must still print the paper's thresholds", () => {
  assert.match(ITEMS.age70.label, /70 years or older/);
  assert.match(ITEMS.leuko.label, /first WBC at or above 10/);
  assert.equal(ITEMS.age70.threshold.ge, 70);
  assert.equal(ITEMS.leuko.threshold.ge, 10);
  for (const a of [0, 1]) for (const w of [0, 1]) {
    const r = predict("bedside", { asymmetry: 0, leuko: w, age70: a, como_malignancy_active: 0, como_diabetes: 0 });
    assert.ok(Math.abs(r.logit - (-1.0763 + 0.187 * w - 0.6539 * a)) < 1e-12);
  }
});
