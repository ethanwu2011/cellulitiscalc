"""Laterality extraction across the whole cohort (admissions with a discharge summary).

Batched, resumable, checkpointed. Appends one JSON line per admission to
data/nlp/cohort_extractions.jsonl so partial progress is usable and the job can resume.
Parallelizes extraction across a small thread pool (Ollama queues requests).

DUA RULE: note text -> local retrieval + local Ollama only. Output (structured extractions
keyed by hadm_id) -> data/nlp/ (gitignored). Only progress counts go to stdout.
"""
from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from shared.mimic_io import from_config
from shared.nlp_engine.extract import extract_case, model_available
from shared.nlp_engine.retrieval import retrieve_snippets

OUT = Path("data/nlp/cohort_extractions.jsonl")
BATCH = 400
WORKERS = 3
_lock = threading.Lock()


def _done_hadms():
    if not OUT.exists():
        return set()
    done = set()
    for line in OUT.read_text().splitlines():
        try:
            done.add(int(json.loads(line)["hadm_id"]))
        except (json.JSONDecodeError, KeyError, ValueError):
            continue
    return done


def _append(rec):
    with _lock:
        with OUT.open("a") as f:
            f.write(json.dumps(rec) + "\n")


def main(default_cfg="config/default.yaml", nlp_cfg="config/nlp.yaml"):
    model = yaml.safe_load(Path(nlp_cfg).read_text())["model"]["name"]
    if not model_available(model):
        print(f"model {model} unavailable, aborting"); return
    OUT.parent.mkdir(parents=True, exist_ok=True)

    src = from_config(default_cfg)
    con = src.con
    con.execute("CREATE OR REPLACE VIEW cohort AS SELECT * FROM read_parquet('data/cohorts/study2_cohort.parquet')")
    todo = con.execute("""
        SELECT DISTINCT CAST(c.hadm_id AS BIGINT) hadm_id, c.label
        FROM cohort c WHERE CAST(c.hadm_id AS BIGINT) IN (SELECT DISTINCT hadm_id FROM discharge)
        ORDER BY hadm_id""").df()

    done = _done_hadms()
    todo = todo[~todo.hadm_id.isin(done)]
    total = len(todo)
    print(f"[nlp-cohort] model={model} to-do={total} (already done {len(done)})", flush=True)

    t0 = time.time()
    processed = 0
    hadms = todo.hadm_id.tolist()
    labels = dict(zip(todo.hadm_id, todo.label))
    for b in range(0, total, BATCH):
        chunk = hadms[b:b + BATCH]
        idlist = ",".join(str(h) for h in chunk)
        notes = con.execute(
            f"SELECT hadm_id, string_agg(text, chr(10)) AS text FROM discharge "
            f"WHERE hadm_id IN ({idlist}) GROUP BY hadm_id").df()
        text_by = dict(zip(notes.hadm_id, notes.text))

        def work(hadm):
            ext = extract_case(retrieve_snippets(text_by.get(hadm, "")), model=model)
            rec = {"hadm_id": int(hadm), "label": labels.get(hadm)}
            if ext.get("_ok"):
                sf = ext["skin_findings"]
                rec.update({
                    "laterality": ext["laterality"],
                    "final_impression_label": ext["final_impression_label"],
                    "extraction_confidence": ext["extraction_confidence"],
                    **{f"skin_{k}": sf.get(k) for k in sf},
                    "consults": ",".join(ext["consult_services_mentioned"]),
                    "ok": True,
                })
            else:
                rec.update({"ok": False, "error": ext.get("_error", "")})
            _append(rec)

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            list(ex.map(work, chunk))
        processed += len(chunk)
        rate = processed / (time.time() - t0)
        eta_h = (total - processed) / rate / 3600 if rate else 0
        print(f"[nlp-cohort] {processed}/{total} ({100*processed/total:.0f}%) "
              f"{rate:.2f}/s ETA {eta_h:.1f}h", flush=True)
    src.close()
    print(f"[nlp-cohort] DONE {processed} in {(time.time()-t0)/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
