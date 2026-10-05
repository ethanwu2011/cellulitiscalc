"""shared.mimic_io.duckdb_source - DuckDB adapter over the raw MIMIC-IV /
MIMIC-IV-Note flat-file distributions.

No PostgreSQL server is required: DuckDB queries the gzipped CSVs directly and pushes
down filters, so multi-GB tables (labevents 2.4 GB, discharge notes 1.1 GB) are never
fully loaded into memory. Each `hosp/*.csv.gz`, `icu/*.csv.gz` (from mimic_iv_root) and
`note/*.csv.gz` (from mimic_note_root) is registered as a lazily-evaluated VIEW named by
its filename stem (e.g. `diagnoses_icd`, `labevents`, `discharge`, `radiology`).

DUA RULE (non-negotiable): `query()` may return row-level data - including note text.
Callers MUST aggregate before any display or save, and note text must never leave
local/approved compute (PhysioNet data use agreement).

Data framing: MIMIC-IV v3.1 + MIMIC-IV-Note v2.2 - single-center, retrospective,
Beth Israel Deaconess Medical Center EHR, 2008-2022.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


class DuckDBSource:
    name = "duckdb"

    def __init__(
        self,
        mimic_iv_root: str | Path,
        mimic_note_root: str | Path | None = None,
        *,
        threads: int | None = None,
        memory_limit: str | None = None,
        temp_directory: str | Path | None = None,
        force_varchar_tables: tuple[str, ...] | list[str] = (),
    ):
        self.iv_root = Path(mimic_iv_root).expanduser()
        if not self.iv_root.exists():
            raise FileNotFoundError(f"mimic_iv_root does not exist: {self.iv_root}")
        self.note_root = Path(mimic_note_root).expanduser() if mimic_note_root else None
        if self.note_root and not self.note_root.exists():
            raise FileNotFoundError(f"mimic_note_root does not exist: {self.note_root}")
        self.force_varchar = set(force_varchar_tables)
        self.con = duckdb.connect()
        if threads:
            self.con.execute(f"PRAGMA threads={int(threads)}")
        if memory_limit:
            self.con.execute(f"PRAGMA memory_limit='{memory_limit}'")
        if temp_directory:
            tmp = Path(temp_directory).expanduser()
            tmp.mkdir(parents=True, exist_ok=True)
            self.con.execute(f"PRAGMA temp_directory='{tmp}'")
        self._paths = self._discover()
        self._register()

    def _discover(self) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for sub in ("hosp", "icu"):
            for f in sorted((self.iv_root / sub).glob("*.csv.gz")):
                paths[f.name[: -len(".csv.gz")]] = f
        if self.note_root:
            for f in sorted((self.note_root / "note").glob("*.csv.gz")):
                paths[f.name[: -len(".csv.gz")]] = f
        if not paths:
            raise FileNotFoundError(f"No *.csv.gz under {self.iv_root}/(hosp|icu)")
        return paths

    def _reader(self, table: str, path: Path) -> str:
        p = str(path).replace("'", "''")
        # icd_code and other zero-padded/alphanumeric keys must stay VARCHAR, so
        # read_csv_auto does not silently coerce e.g. '0010' -> 10.
        if table in self.force_varchar:
            return f"read_csv('{p}', auto_detect=true, all_varchar=true)"
        return f"read_csv_auto('{p}')"

    def _register(self) -> None:
        for table, path in self._paths.items():
            self.con.execute(
                f'CREATE OR REPLACE VIEW "{table}" AS '
                f"SELECT * FROM {self._reader(table, path)}"
            )

    def list_tables(self) -> list[str]:
        """Available table names (schema metadata - DUA-safe)."""
        return sorted(self._paths)

    def table_columns(self, table: str) -> pd.DataFrame:
        if table not in self._paths:
            raise KeyError(f"Unknown table '{table}'. Available: {self.list_tables()}")
        return self.con.execute(f'DESCRIBE "{table}"').df()[["column_name", "column_type"]]

    def count(self, table: str, where: str | None = None, params: Any = None) -> int:
        """Aggregate row count (DUA-safe)."""
        if table not in self._paths:
            raise KeyError(f"Unknown table '{table}'.")
        sql = f'SELECT count(*) AS n FROM "{table}"'
        if where:
            sql += f" WHERE {where}"
        return int(self.con.execute(sql, params or []).fetchone()[0])

    def query(self, sql: str, params: Any = None) -> pd.DataFrame:
        """INTERNAL - output may be row-level (incl. note text). Aggregate before display."""
        return self.con.execute(sql, params or []).df()

    def health(self) -> dict:
        """Aggregate-only connectivity summary for diagnostics/dry-run."""
        return {
            "source": self.name,
            "iv_root": str(self.iv_root),
            "note_root": str(self.note_root) if self.note_root else None,
            "n_tables": len(self._paths),
            "modules": sorted({p.parent.name for p in self._paths.values()}),
            "tables": self.list_tables(),
        }

    def close(self) -> None:
        self.con.close()

    def __enter__(self) -> "DuckDBSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def from_config(config_path: str | Path = "config/default.yaml") -> DuckDBSource:
    """Instantiate a DuckDBSource from config/default.yaml (paths + duckdb settings)."""
    import yaml

    cfg = yaml.safe_load(Path(config_path).read_text())
    paths, d = cfg["paths"], cfg.get("duckdb", {})
    return DuckDBSource(
        paths["mimic_iv"],
        paths.get("mimic_note"),
        threads=d.get("threads"),
        memory_limit=d.get("memory_limit"),
        temp_directory=d.get("temp_directory"),
        force_varchar_tables=d.get("force_varchar", ()),
    )
