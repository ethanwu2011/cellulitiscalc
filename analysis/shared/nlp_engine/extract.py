"""shared.nlp_engine.extract - structured extraction from retrieved snippets via a LOCAL
Ollama model. No note text ever leaves this machine (localhost only).

Output schema (config/nlp.yaml): laterality, skin_findings{warmth,induration,fluctuance,
purulence,bullae}, consult_services_mentioned, final_impression_label,
extraction_confidence, source_sentence_span.

DUA RULE: posts snippets to http://localhost:11434 ONLY. Never a remote endpoint.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

HOST = "http://localhost:11434"

SYSTEM = (
    "You are a clinical information-extraction tool. From the provided discharge-summary "
    "SNIPPETS about a patient admitted with a possible lower-extremity skin/soft-tissue "
    "problem, extract ONLY what is explicitly documented. Do not infer beyond the text. "
    "Return a single JSON object with EXACTLY these keys:\n"
    '  "laterality": one of "unilateral_left","unilateral_right","bilateral","not_documented"\n'
    '  "skin_findings": object with boolean-or-null keys warmth, induration, fluctuance, purulence, bullae\n'
    '  "consult_services_mentioned": array of strings (e.g. "dermatology","infectious_disease"); [] if none\n'
    '  "final_impression_label": one of "cellulitis","stasis_dermatitis","dvt_suspected","gout","other_mimic","indeterminate"\n'
    '  "extraction_confidence": number 0.0-1.0\n'
    '  "source_sentence_span": the single verbatim sentence best supporting final_impression_label\n'
    "Use not_documented / null when the text does not say. Output JSON only."
)

_LAT = {"unilateral_left", "unilateral_right", "bilateral", "not_documented"}
_IMP = {"cellulitis", "stasis_dermatitis", "dvt_suspected", "gout", "other_mimic", "indeterminate"}
_SKIN = ("warmth", "induration", "fluctuance", "purulence", "bullae")


def extract_case(snippets, model, host=HOST, timeout=180) -> dict:
    """Call the local model on the snippets; return the validated/normalized dict.

    Adds `_ok` (bool) and, on failure, `_error`. Never raises on model/HTTP errors.
    """
    assert host.startswith(("http://localhost", "http://127.0.0.1")), "laterality extraction must use a local model"
    user = "SNIPPETS:\n" + "\n---\n".join(snippets) if snippets else "SNIPPETS: (none found)"
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        "stream": False, "format": "json", "options": {"temperature": 0},
    }
    req = urllib.request.Request(
        host + "/api/chat", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            resp = json.loads(r.read())
        raw = json.loads(resp["message"]["content"])
    except (urllib.error.URLError, json.JSONDecodeError, KeyError, TimeoutError) as e:
        return {"_ok": False, "_error": f"{type(e).__name__}: {e}"}
    return _normalize(raw)


def _normalize(raw: dict) -> dict:
    sf = raw.get("skin_findings") or {}
    out = {
        "laterality": raw.get("laterality") if raw.get("laterality") in _LAT else "not_documented",
        "skin_findings": {k: (bool(sf[k]) if isinstance(sf.get(k), bool) else None) for k in _SKIN},
        "consult_services_mentioned": raw.get("consult_services_mentioned") or [],
        "final_impression_label": raw.get("final_impression_label") if raw.get("final_impression_label") in _IMP else "indeterminate",
        "source_sentence_span": (raw.get("source_sentence_span") or "")[:400],
        "_ok": True,
    }
    try:
        out["extraction_confidence"] = max(0.0, min(1.0, float(raw.get("extraction_confidence", 0.0))))
    except (TypeError, ValueError):
        out["extraction_confidence"] = 0.0
    return out


def model_available(model, host=HOST) -> bool:
    try:
        with urllib.request.urlopen(host + "/api/tags", timeout=10) as r:
            tags = json.loads(r.read())
        names = {m["name"] for m in tags.get("models", [])}
        return model in names or any(n.split(":")[0] == model.split(":")[0] for n in names)
    except (urllib.error.URLError, json.JSONDecodeError, KeyError, TimeoutError):
        return False
