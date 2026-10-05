"""shared.nlp_engine - snippet retrieval + local-LLM extraction.

DUA RULE (non-negotiable): PHI never leaves local/approved compute. Note text is
processed ONLY by a local open-weight LLM (Ollama) on this machine. No note text is
ever sent to an external API. Retrieval returns a bounded snippet; the LLM sees the
snippet, never the full note, unless a step explicitly requires it. Aggregates out.

Data framing: MIMIC-IV-Note v2.2 - single-center, retrospective, BIDMC, 2008-2022.
"""
