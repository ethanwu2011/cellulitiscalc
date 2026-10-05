"""CONSORT-style exclusion-flow logging. Aggregate counts only - DUA-safe to commit.

Data framing lives in the log title: single-center, retrospective, BIDMC, 2008-2022.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ConsortStep:
    step: str
    n_admissions: int
    n_patients: int
    note: str = ""


@dataclass
class ConsortLog:
    title: str
    steps: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def add(self, step: str, n_admissions: int, n_patients: int, note: str = "") -> "ConsortLog":
        self.steps.append(ConsortStep(step, int(n_admissions), int(n_patients), note))
        return self

    def to_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(
            {"title": self.title, "meta": self.meta,
             "steps": [asdict(s) for s in self.steps]}, indent=2))

    def render(self) -> str:
        lines = [self.title]
        for s in self.steps:
            tail = f"   ({s.note})" if s.note else ""
            lines.append(f"  {s.n_admissions:>8,} adm | {s.n_patients:>8,} pts  - {s.step}{tail}")
        return "\n".join(lines)
