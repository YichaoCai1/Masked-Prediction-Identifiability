"""Tidy-table output with provenance stamping.

Parquet is the primary format (spec section 6); if ``pyarrow`` is unavailable
the same frame is written as CSV next to it and a note is printed.  Every table
carries the config hash, git hash and branch threshold ``T`` so that a figure
can always be traced back to the run that produced it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import RESULTS_DIR, RunConfig, provenance

__all__ = ["write_table", "read_table", "stamp"]


def stamp(df: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    prov = provenance(cfg)
    out = df.copy()
    for key, value in prov.items():
        out[key] = value
    return out


def write_table(df: pd.DataFrame, name: str, cfg: RunConfig,
                results_dir: Path | None = None) -> Path:
    """Write ``<name>.parquet`` (or ``.csv``) into the results directory."""
    results_dir = Path(results_dir or RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = stamp(df, cfg)
    target = results_dir / f"{name}.parquet"
    try:
        out.to_parquet(target, index=False)
    except Exception as exc:  # pragma: no cover - depends on optional pyarrow
        target = results_dir / f"{name}.csv"
        out.to_csv(target, index=False)
        print(f"  [warn] parquet unavailable ({exc.__class__.__name__}); wrote {target.name}")
    return target


def read_table(name: str, results_dir: Path | None = None) -> pd.DataFrame:
    """Read ``<name>`` written by :func:`write_table`, parquet or CSV."""
    results_dir = Path(results_dir or RESULTS_DIR)
    pq = results_dir / f"{name}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    csv = results_dir / f"{name}.csv"
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(
        f"no artifact {name!r} in {results_dir}; run `python scripts/run_all.py` first"
    )
