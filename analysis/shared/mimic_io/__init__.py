"""shared.mimic_io - DuckDB loaders over the raw MIMIC-IV / eICU flat files.

No PostgreSQL: DuckDB queries the gzipped CSVs directly.
Each hosp/*.csv.gz, icu/*.csv.gz and
note/*.csv.gz is a lazily-evaluated view named by its filename stem.

DUA RULE: queries may return row-level data - aggregate before display; nothing
leaves local/approved compute.

Data framing: MIMIC-IV v3.1 - single-center, retrospective, BIDMC, 2008-2022.
"""
from .duckdb_source import DuckDBSource, from_config

__all__ = ["DuckDBSource", "from_config"]
