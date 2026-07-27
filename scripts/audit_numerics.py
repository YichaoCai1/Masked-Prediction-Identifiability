#!/usr/bin/env python
"""Leg (c) of the numerical CI: float64 against mpmath at 50 digits.

This is separated from ``run_all.py`` because arbitrary-precision arithmetic
costs ``O(m(N-m))`` mpf operations per cell -- seconds to minutes for the
largest ``N`` -- whereas the whole float64 run matrix takes under a minute.

    python scripts/audit_numerics.py                 # 24 cells, N <= 255
    python scripts/audit_numerics.py --max-N 1023 --n-cells 40

Writes ``results/audit.parquet``, which feeds appendix figure A7.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mpi.config import (  # noqa: E402
    RESULTS_DIR,
    RunConfig,
    mode_weight_interval,
    visible_sizes,
)
from mpi.core import visible_law  # noqa: E402
from mpi.estimands import log_D, log_U  # noqa: E402
from mpi.highprec import log_D_mp, log_U_mp  # noqa: E402
from mpi.tableio import write_table  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    p.add_argument("--n-cells", type=int, default=24)
    p.add_argument("--max-N", type=int, default=255)
    p.add_argument("--dps", type=int, default=50)
    p.add_argument("--seed", type=int, default=20260725)
    args = p.parse_args(argv)

    cfg = RunConfig()
    specs = cfg.law_specs()
    Ns = [n for n in cfg.N_grid if n <= args.max_N]
    rng = np.random.default_rng(args.seed)

    rows = []
    t0 = time.time()
    for i in range(args.n_cells):
        spec = specs[int(rng.integers(len(specs)))]
        N = int(rng.choice(Ns))
        w = float(rng.choice(cfg.w_grid))
        m = int(rng.choice(visible_sizes(N)))
        a, b = mode_weight_interval(w)
        lam = float(rng.uniform(a, b))
        if abs(lam - w) < 1e-3:
            lam = b

        ref = log_D_mp(spec, N, m, w, lam, dps=args.dps)
        ref_u = log_U_mp(spec, N, m, w, dps=args.dps)
        vl = visible_law(spec, N, m, w)
        got_u = log_U(vl)

        for impl in ("stable", "branched"):
            got = log_D(vl, lam, cfg.branch_T, impl)
            rows.append(
                {
                    "law": spec.law, "param": spec.param, "N": N, "m": m,
                    "w": w, "lambda": lam, "impl": impl, "dps": args.dps,
                    "logD_float64": got, "logD_mp": ref,
                    "abs_err_log": got - ref,
                    "logU_float64": got_u, "logU_mp": ref_u,
                    "abs_err_logU": got_u - ref_u,
                }
            )
        print(f"  [{i+1:>3d}/{args.n_cells}] {spec.key:>12s} N={N:<5d} m={m:<4d} "
              f"w={w:<4g}  |dlogD| stable="
              f"{abs(rows[-2]['abs_err_log']):.2e} branched="
              f"{abs(rows[-1]['abs_err_log']):.2e}")

    audit = pd.DataFrame(rows)
    write_table(audit, "audit", cfg, Path(args.results_dir))

    print(f"\nDone in {time.time()-t0:.1f}s")
    for impl, g in audit.groupby("impl"):
        print(f"  {impl:>9s}: max |d log D| = {g.abs_err_log.abs().max():.3e}")
    print(f"  {'logU':>9s}: max |d log U| = {audit.abs_err_logU.abs().max():.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
