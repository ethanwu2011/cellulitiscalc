# CELTIC calculator

Source for [cellulitiscalc.com](https://www.cellulitiscalc.com/), the web calculator for CELTIC (Cellulitis Tiered
Interpretable Classifier), a logistic regression model that estimates the probability of cellulitis versus a
noninfectious mimic in a hospitalized adult with a red, warm, swollen limb.

The site implements the two tiers that can be entered by hand: Bedside (5 yes/no findings) and Rapid (10). The Standard
(37 predictors) and Comprehensive (94 predictors) tiers read the electronic record and are shown for reference only.
Everything runs in the browser. There is no server, no analytics, and nothing entered is stored.

Research use only. The model has not been validated prospectively.

## Run it

```bash
npm run serve    # python3 -m http.server 4321, then open http://localhost:4321
npm test         # node --test tests/*.test.mjs
```

There is no build step and there are no dependencies.

## Layout

| Path | What it is |
|---|---|
| `index.html`, `styles.css`, `fonts.css`, `fonts/` | the page and its self-hosted fonts (Literata, Atkinson Hyperlegible Next; SIL Open Font License, texts in `fonts/`) |
| `js/model.js` | the model: coefficients, thresholds, the 30% and 65% bands, and the probability arithmetic |
| `js/app.js` | the interface; it only reads answers and draws the result |
| `model_source.json` | the numbers copied from the analysis output, which `js/model.js` reproduces |
| `tests/` | checks `js/model.js` against 25 patient profiles from the paper and a worked example |
| `analysis/` | the code that produced the paper's numbers, without data (see `analysis/README.md`) |

## The model

Bedside tier: `logit = -1.076 + 1.530 unilateral + 0.187 leukocytosis - 0.654 age>=70 - 0.562 malignancy + 0.399 diabetes`,
probability `= 1 / (1 + exp(-logit))`. Below 30% is the rule-out band and 65% or above is the rule-in band. A blank finding
counts as no, which is how missing values were handled in the study. No tier uses a culture result.

Derivation and testing used MIMIC-IV v3.1 with MIMIC-IV-Note v2.2, MIMIC-IV-ED v2.2, and eICU-CRD v2.0, available to
credentialed users through PhysioNet. No patient data are in this repository.

## Deploy

Any static host works. On Vercel: import the repository, framework preset "Other", no build command, output directory `.`.

## Cite

Wu E, Lee A, Youssef A, English J, Mason E, Xing EP, Pugliano-Mauro M. CELTIC, a tiered interpretable model to distinguish
cellulitis from mimics in hospitalized adults. Submitted to JAMA Dermatology, 2026.

## License

MIT (see `LICENSE`). Fonts are under the SIL Open Font License.
