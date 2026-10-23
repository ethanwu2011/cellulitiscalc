# CELTIC analysis code

This folder holds the code that produced the numbers in the CELTIC paper: Wu E, Lee A, Youssef A, English J, Mason E, Xing EP, Pugliano-Mauro M. CELTIC: A Tiered Model to Distinguish Lower Extremity Cellulitis From Mimics in Hospitalized Adults. Submitted to JAMA Dermatology, 2026.

The code builds a cohort of hospitalized adults coded for cellulitis or for one of 18 pseudocellulitis conditions, reads limb laterality from the discharge summary with a local language model, fits the four CELTIC tiers (Bedside, Rapid, Standard, Comprehensive), and compares them with the ALT-70 in MIMIC-IV inpatients. It then compares structured versions of the model with the ALT-70 in MIMIC-IV-ED and in eICU-CRD. No tier uses a culture result, and no code here reads one. No patient data are in this repository. The browser calculator in the parent folder does not need this code to run.

`config/` holds the settings, `shared/` the data loaders and the laterality extractor, and `study2_cellulitis/` the scripts.

## Data access

All four datasets come from PhysioNet under credentialed access. For each you need a PhysioNet account, the human-subjects training PhysioNet requires, and a signed data use agreement.

| Dataset | Version | Folder the code expects |
|---|---|---|
| [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/) | 3.1 | `~/mimiciv-3.1/hosp/` and `~/mimiciv-3.1/icu/` |
| [MIMIC-IV-Note](https://physionet.org/content/mimic-iv-note/2.2/) | 2.2 | `~/mimic-iv-note-2.2/note/` |
| [MIMIC-IV-ED](https://physionet.org/content/mimic-iv-ed/2.2/) | 2.2 | `~/mimic-iv-ed-2.2/ed/` |
| [eICU-CRD](https://physionet.org/content/eicu-crd/2.0/) | 2.0 | `~/eicu-crd-2.0/` (files directly in this folder) |

Tables read: MIMIC-IV `admissions`, `patients`, `diagnoses_icd`, `labevents`, `omr`, `prescriptions` and `chartevents`; MIMIC-IV-Note `discharge`; MIMIC-IV-ED `edstays`, `triage` and `diagnosis`; eICU-CRD `diagnosis`, `patient`, `lab` and `vitalPeriodic`. File names are case sensitive on Linux. The loader registers every csv.gz file it finds in the MIMIC-IV and note folders, but only these tables are queried.

Two paths are set in `config/default.yaml` (`paths.mimic_iv`, `paths.mimic_note`); they serve the cohort, feature and laterality steps and the Table 1, subgroup, temporal and antibiotic queries in `celtic_noculture.py`. The others are fixed at the top of three scripts: `H` and `ED` in `celtic_noculture.py` (also used by `celtic_noculture_ed2.py` and `celtic_noculture_ed3.py`), `DIAG` in `per_category.py`, and `EICU` in `external_eicu_v2.py`. `paths.eicu` and `paths.cache` in `config/default.yaml` are not read. The simplest route is to keep every path and link your downloads to the expected names:

```bash
ln -s /path/to/mimiciv/3.1 ~/mimiciv-3.1
ln -s /path/to/mimic-iv-note/2.2 ~/mimic-iv-note-2.2
ln -s /path/to/mimic-iv-ed/2.2 ~/mimic-iv-ed-2.2
ln -s /path/to/eicu-crd/2.0 ~/eicu-crd-2.0
```

Without MIMIC-IV-ED, step 4 fails with a file-not-found error after writing the inpatient results; without eICU-CRD it fails after writing the emergency department results. Results already written stay valid. In step 5, the two emergency department scripts need MIMIC-IV-ED and MIMIC-IV; the addendum needs neither.

## Setup

The paper's numbers were produced with Python 3.14.6 and the package versions in `requirements.txt`. Other versions are untested.

```bash
cd analysis
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

scikit-learn 1.6.1 has no prebuilt wheel for Python 3.14, so pip builds it from source, which needs a C and C++ compiler (on macOS, the Xcode command line tools and `brew install libomp`). Git must be on your `PATH`, because the scripts record the current commit in their output.

The laterality step needs [Ollama](https://ollama.com) and one model:

```bash
ollama pull qwen2.5:7b-instruct
ollama list    # confirm qwen2.5:7b-instruct is listed
```

The authors' machine runs Ollama 0.35.1 with `qwen2.5:7b-instruct` digest `845dbda0ea48`; the extraction file does not record the version used at run time. Temperature is fixed at 0, but other Ollama builds or model digests may change a small number of extractions.

Resources: `config/default.yaml` gives DuckDB 8 threads and a 24 GB memory limit, with spill files in `data/cache/duckdb_tmp/`; lower `duckdb.threads` and `duckdb.memory_limit` on a smaller machine. The emergency department and eICU sections of step 4 and the step 5 scripts open their own DuckDB connections (8 GB and 6 threads, set in the code; `per_category.py` uses DuckDB defaults). Step 2 scans `labevents` and `chartevents`, the emergency department analyses scan `labevents`, and the eICU analysis scans `lab` and `vitalPeriodic`, the largest read in the pipeline.

## Run order

Run every step from `analysis/` with `PYTHONPATH=.`; all paths in the code are relative to this folder.

**Step 1. Cohort and split.** Cases carry a cellulitis or erysipelas code (ICD-9-CM 681, 682, 035; ICD-10-CM L03, A46) and no mimic code. Mimics carry a code for one of the 18 pseudocellulitis conditions listed in `config/cohorts.yaml` and no case code. Admissions coded for both are excluded. Each adult contributes the earliest eligible admission, and patients are split 60/20/20, stratified by label. The paper's cohort is 24,852 patients, 47.4% with cellulitis.

```bash
PYTHONPATH=. python study2_cellulitis/cohort.py
```

**Step 2. Structured features.** First 48 hours of laboratory values, ICU vital signs, body mass index, comorbidity flags and the ALT-70 components that do not need a note. Steps 2 and 3 do not depend on each other.

```bash
PYTHONPATH=. python study2_cellulitis/features.py
```

**Step 3. Laterality extraction (long, resumable).** One local model call per admission with a discharge summary. Short snippets around fixed anchors (defaults in `shared/nlp_engine/retrieval.py`) are passed to the model, never the full note. The script appends one line per admission and skips admissions already in the output, so it can be stopped and restarted. A failed call is written with `ok` false and is not retried on restart; delete those lines to retry them. If the model is missing, the script says so and exits with status 0, so confirm the tag with `ollama list` first. To check the failure count afterwards (counts only):

```bash
PYTHONPATH=. python study2_cellulitis/nlp_extract_cohort.py
python -c "import json; r=[json.loads(l) for l in open('data/nlp/cohort_extractions.jsonl')]; print(len(r), sum(not x.get('ok') for x in r))"
```

**Step 4. Main analysis.** The four tiers, the ALT-70 comparison, probability bands, operating points, the stewardship simulation, Table 1, subgroups, temporal validation, and the MIMIC-IV-ED and eICU-CRD analyses. The emergency department and eICU cohorts are defined inside this script and in `external_eicu_v2.py` from their own diagnosis tables, not from `config/cohorts.yaml`. Culture columns do not exist in the feature table; the script also excludes any column named `micro*` and asserts that none reaches the Comprehensive tier. Bedside and Rapid are fixed item lists.

```bash
PYTHONPATH=. python study2_cellulitis/celtic_noculture.py
```

**Step 5. Add-on analyses.** Run one at a time, after step 4. Each reads the results file, adds keys and rewrites it, and running step 4 again erases those keys. They run on import, so run them as scripts.

```bash
PYTHONPATH=. python study2_cellulitis/celtic_noculture_addendum.py   # patients per band, by diagnosis
PYTHONPATH=. python study2_cellulitis/celtic_noculture_ed2.py        # emergency department model on the Bedside items without the examination item
PYTHONPATH=. python study2_cellulitis/celtic_noculture_ed3.py        # why discrimination is lower in the emergency department; like-for-like on admitted stays
```

`external_eicu_v2.py`, `per_category.py` and `delong.py` are helpers imported by step 4.

## What each step writes

| Step | File | Level | Contents |
|---|---|---|---|
| 1 | `data/cohorts/study2_cohort.parquet` | Row-level | one row per patient: identifiers, admission time, age, label, split |
| 1 | `study2_cellulitis/cohort_consort.json` | Aggregate | exclusion counts, class balance, split counts, seed |
| 1, 2, 3 | `data/cache/duckdb_tmp/` | Row-level scratch | DuckDB spill files, normally deleted |
| 4, 5 | `.tmp/` | Row-level scratch | DuckDB spill files for connections without a set folder, normally deleted |
| 2 | `data/cohorts/study2_features.parquet` | Row-level | first-48-hour feature table |
| 3 | `data/nlp/cohort_extractions.jsonl` | Row-level | per admission: laterality, impression label, confidence, skin findings, consults, `ok`; no note text |
| 4 | `study2_cellulitis/celtic_noculture_metrics.json` | Aggregate | every model result, interval, band and operating point |
| 4 | `study2_cellulitis/celtic_table1.json` | Aggregate | Table 1 |
| 5 | `study2_cellulitis/celtic_noculture_metrics.json` | Aggregate | adds `band_counts`, `ed.bedside_items`, `ed.bedside_items_plus_vitals`, `ed.comorbidity_prevalence_pct`, `ed.admitted_like_for_like` |

The results file can contain `NaN` values, which Python's `json` module reads and JavaScript's `JSON.parse` does not.

## Where the paper's numbers come from

Key paths are in `study2_cellulitis/celtic_noculture_metrics.json`. AUROC is the area under the receiver operating characteristic curve; intervals are 95% bootstrap intervals. Every tier, band, operating point and stewardship figure uses the 17,024 patients whose discharge summary was processed by the laterality step (68.5% of the cohort), so the inpatient test set is 3,446 of the 4,971 test patients (1,566 with cellulitis); `inpatient_full_test` holds a structured-only reference on all 4,971.

| Number | Value | Key path | Script |
|---|---|---|---|
| Bedside AUROC, inpatient test | 0.721 (0.704 to 0.737) | `inpatient_nlp_test.aurocs.bedside_prob` | `celtic_noculture.py` |
| ALT-70 AUROC, same patients | 0.604 (0.587 to 0.622) | `inpatient_nlp_test.aurocs.alt70_complete` | `celtic_noculture.py` |
| Bedside minus ALT-70 | 0.116 (0.097 to 0.135) | `inpatient_nlp_test.paired_differences.bedside_prob_minus_alt70_complete` | `celtic_noculture.py` |
| Rapid, Standard, Comprehensive AUROC | 0.727, 0.747, 0.756 | `inpatient_nlp_test.aurocs.rapid_prob`, `.standard_prob`, `.comprehensive_prob` | `celtic_noculture.py` |
| Emergency department, Bedside items without examination, vs ALT-70 (3,314 stays) | 0.563 vs 0.517 | `ed.bedside_items.aurocs_test.celtic`, `.alt70` | `celtic_noculture_ed2.py` |
| eICU external, 18-feature structured model vs structured ALT-70 (2,536 patients) | 0.690 vs 0.540 | `eicu.aurocs.celtic_structured`, `.alt70_structured` | `celtic_noculture.py` |
| Sensitivity at the cut point closest in specificity: Bedside points 3 or more vs ALT-70 5 or more | 64.2% vs 28.0% (specificity 69.4% vs 75.8%) | `operating_inpatient.key_points.bedside_matched_spec`, `.alt70_ge5` | `celtic_noculture.py` |
| Probability bands: below 30% and 65% or above | 31.5% and 19.2% of test patients; cellulitis 20.5% and 68.2% | `bands.bedside_test` | `celtic_noculture.py` |
| Mimic patients given a systemic antibiotic (test set) | 58.4% | `stewardship_nlp_test.pct_mimic_any_abx_test` | `celtic_noculture.py` |
| At 95% sensitivity: mimic antibiotic days avoided, cellulitis missed | 23.2%, 4.7% | `stewardship_nlp_test.p_bed_at_sens_95` | `celtic_noculture.py` |

P values for AUROC differences are from the DeLong test (`delong` fields). The operating-point row uses the integer points score (AUROC 0.717). The eICU model uses age, sex, eight laboratory values, four vital signs and four comorbidities (`SHARED` in `external_eicu_v2.py`); eICU has no notes, so it has no examination item. The calculator's coefficients in `../model_source.json` and `../js/model.js` were copied from `bedside_model` and `rapid_model` in the results file, and `npm test` in the parent folder checks them against 25 patient profiles from the paper.

## Reproducibility

From this folder, steps 1, 2, 4 and 5 were rerun against the PhysioNet files with the laterality output of step 3 reused, and the output was compared with the run reported in the paper (16 September 2026). The cohort table came out identical, the feature table identical in every column it contains, and every inpatient number was reproduced exactly, including Table 1. Emergency department and eICU AUROCs, AUROC differences and DeLong P values were also identical. Their bootstrap intervals can move in the third decimal between runs, because those tables come back from DuckDB in no fixed row order and the bootstrap resamples by position. Separately, each eICU patient's index stay is chosen by admission offset with no tie-breaker, so a patient with two stays at the same offset can be represented by a different stay; between runs this moved one eICU operating point by one patient (sensitivity 53.9% in the paper, 54.0% on a rerun), while the eICU AUROCs, stored at two decimals, did not change.

## Fixed settings

| Setting | Value | Where |
|---|---|---|
| Random seed | 20260707 | `seed` in `config/default.yaml` and `SEED` in `celtic_noculture.py` |
| Split | by patient, never by admission; 60/20/20 stratified by label | `shared/cohorts/splits.py`, `split` in `config/default.yaml` |
| Final fits | train plus validation (80%), evaluated on the 20% test split | `celtic_noculture.py` |
| Inpatient feature window | first 48 hours after admission | `lab_window_hours` in `config/features_study2.yaml` |
| Emergency department items | triage vital signs; white cell count from 6 hours before to 24 hours after arrival; comorbidities from hospital admissions up to 2 days after the visit | `celtic_noculture.py`, `celtic_noculture_ed2.py`, `celtic_noculture_ed3.py` |
| eICU items | laboratory values and vital signs in the first 48 hours after ICU admission; comorbidities from the eICU diagnosis table | `external_eicu_v2.py` |
| Bootstrap | 2,000 stratified resamples, paired across scores | `NBOOT` in `celtic_noculture.py` |
| Probability bands | below 0.30 rule-out, 0.65 or above rule-in | `celtic_noculture.py` |
| Emergency department split | 70/30 by patient | `celtic_noculture.py`, `celtic_noculture_ed2.py`, `celtic_noculture_ed3.py` |
| Temporal check | by each patient's MIMIC `anchor_year_group`: fit 2008 to 2016, test 2017 to 2022, and the reverse | `celtic_noculture.py` |
| eICU | model fit on MIMIC-IV train plus validation, applied unchanged to the whole eICU cohort | `celtic_noculture.py` |
| Laterality model | `qwen2.5:7b-instruct` through Ollama on 127.0.0.1, temperature 0, JSON output | model name in `config/nlp.yaml`; temperature, output format, anchors and window sizes in `shared/nlp_engine/extract.py` and `retrieval.py` |

## Data use

The PhysioNet data use agreements limit what can leave your machine. The laterality step sends note snippets only to Ollama at `http://localhost:11434` and refuses any other host (`shared/nlp_engine/extract.py`); keep Ollama bound to 127.0.0.1 and never point this step at a hosted model. The extraction file stores no note text. Row-level files are written under `data/`, which the repository's `.gitignore` (in the parent folder) excludes along with the results files, spill folders and virtual environments. Do not commit, upload or send row-level files. Only aggregate results may leave the machine, and even those can contain small counts (per-condition mimic counts, confusion-matrix cells at extreme thresholds), so check for small cells before sharing; the paper and the calculator suppress cells under 20 patients.

## License and citation

MIT (see `../LICENSE`). Cite as: Wu E, Lee A, Youssef A, English J, Mason E, Xing EP, Pugliano-Mauro M. CELTIC: A Tiered Model to Distinguish Lower Extremity Cellulitis From Mimics in Hospitalized Adults. Submitted to JAMA Dermatology, 2026.
