"""shared.nlp_engine.retrieval - SPELL-style snippet retrieval.

Regex/keyword anchors locate the small spans of a discharge summary that carry the
diagnostic signal (laterality/asymmetry, skin findings, impression, consults). Only these
bounded snippets are ever passed to the local LLM - never the full note.

DUA RULE: operates on note text locally; returns snippets that stay on local/approved
compute. Callers must not export raw snippets off-machine.
"""
from __future__ import annotations

import re

# anchor -> regex. Case-insensitive. Kept broad; windows are merged so overlap is fine.
DEFAULT_ANCHORS = {
    "laterality": r"\b(left|right|bilateral|unilateral|asymmetr|symmetr)\b",
    "leg": r"\b(cellulitis|erysipelas|erythema|swelling|edema|extremit|\bleg\b|calf|shin|ankle|foot)\b",
    "skin_findings": r"\b(warm|warmth|indurat|fluctuan|purulen|drainage|bulla|bullae|tender|erythema)\b",
    "impression": r"(impression|assessment and plan|assessment/plan|a&p|diagnosis:|discharge diagnosis)",
    "consult": r"\b(dermatology|infectious disease|derm consult|id consult|consulted)\b",
}


def retrieve_snippets(text, anchors=None, window_chars=1200, max_total_chars=6000):
    """Return merged text windows around anchor hits (bounded, deduplicated).

    window_chars: total width centered on each hit. max_total_chars: hard cap on the
    concatenated snippet length handed to the LLM.
    """
    if not text:
        return []
    anchors = anchors or DEFAULT_ANCHORS
    low = text.lower()
    half = window_chars // 2
    spans = []
    for rx in anchors.values():
        for m in re.finditer(rx, low):
            spans.append((max(0, m.start() - half), min(len(text), m.end() + half)))
    if not spans:
        return []
    # merge overlapping spans
    spans.sort()
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    snippets, total = [], 0
    for s, e in merged:
        chunk = re.sub(r"\s+", " ", text[s:e]).strip()
        if total + len(chunk) > max_total_chars:
            chunk = chunk[: max_total_chars - total]
        snippets.append(chunk)
        total += len(chunk)
        if total >= max_total_chars:
            break
    return snippets


def coverage_flags(text):
    """Aggregate-safe booleans about what a note contains (no text returned)."""
    low = (text or "").lower()
    return {
        "has_laterality": bool(re.search(DEFAULT_ANCHORS["laterality"], low)),
        "has_skin_findings": bool(re.search(DEFAULT_ANCHORS["skin_findings"], low)),
        "has_impression": bool(re.search(DEFAULT_ANCHORS["impression"], low)),
        "mentions_consult": bool(re.search(DEFAULT_ANCHORS["consult"], low)),
    }
