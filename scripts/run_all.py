#!/usr/bin/env python
"""Run the experiments and write the result tables into ``results/``.

    python scripts/run_all.py                 # E1 + E2 (exact, no network)
    python scripts/run_all.py --e3            # also E3 on the offline corpus
    python scripts/run_all.py --quick         # smaller grid, for a smoke test

See the README for the full option list.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mpi.config import (  # noqa: E402
    ALL_LAWS,
    LAWS_C_SWEEP,
    RESULTS_DIR,
    RunConfig,
    provenance,
)
from mpi.e1 import run_e1  # noqa: E402
from mpi.e2 import run_e2  # noqa: E402
from mpi.e4 import run_e4  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    p.add_argument("--quick", action="store_true",
                   help="reduced grid (Law P only, N in {127,255}, w=0.5)")
    p.add_argument("--laws", nargs="*", default=None,
                   help="law keys, e.g. product-0.8 cw-1.5 (default: all four)")
    p.add_argument("--N", nargs="*", type=int, default=None)
    p.add_argument("--w", nargs="*", type=float, default=None)
    p.add_argument("--cw-sweep", action="store_true",
                   help="add the optional Curie-Weiss sweep beta in {0.9, 1.1}")
    p.add_argument("--kl-impl", choices=["stable", "branched"], default=None,
                   help="binary-KL evaluation scheme (default: stable)")
    p.add_argument("--branch-T", type=float, default=None,
                   help="branch threshold, only used by --kl-impl branched")

    p.add_argument("--skip-e1", action="store_true")
    p.add_argument("--skip-e2", action="store_true")
    p.add_argument("--skip-e4", action="store_true")

    p.add_argument("--e3", action="store_true", help="also run E3 (real corpora)")
    p.add_argument("--corpora", nargs="*", default=["code_prose", "bilingual"],
                   choices=["code_prose", "bilingual", "markov"],
                   help="code_prose carries the 'nonempty window' claim, "
                        "bilingual the numerical kappa-hat; markov is an "
                        "offline fixture, never reported")
    p.add_argument("--tokenizer", default="hf:gpt2",
                   help="shared neutral tokenizer: 'hf:<name>' or 'bytes' (offline)")
    p.add_argument("--n-docs", type=int, default=20000,
                   help="packed documents per side")
    p.add_argument("--doc-tokens", type=int, default=1152,
                   help="tokens per packed document (must exceed max |V|)")
    p.add_argument("--n-windows", type=int, default=20000,
                   help="windows drawn per side, per split, per |V|")
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--calibration", default="isotonic", choices=["isotonic", "platt"])
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    laws = args.laws or [s.key for s in ALL_LAWS]
    if args.cw_sweep:
        laws = list(laws) + [s.key for s in LAWS_C_SWEEP]
    kw = {}
    if args.quick:
        kw.update(laws=("product-0.6", "product-0.8"), N_grid=(127, 255), w_grid=(0.5,))
    else:
        kw.update(laws=tuple(laws))
        if args.N:
            kw["N_grid"] = tuple(args.N)
        if args.w:
            kw["w_grid"] = tuple(args.w)
    if args.kl_impl:
        kw["kl_impl"] = args.kl_impl
    if args.branch_T is not None:
        kw["branch_T"] = args.branch_T

    cfg = RunConfig(**kw)
    results_dir = Path(args.results_dir)

    print("Masked-prediction identifiability -- experiment run")
    for k, v in provenance(cfg).items():
        print(f"  {k:>12s}: {v}")
    print(f"  {'laws':>12s}: {', '.join(cfg.laws)}")
    print(f"  {'N':>12s}: {cfg.N_grid}")
    print(f"  {'w':>12s}: {cfg.w_grid}")
    print()

    t0 = time.time()
    if not args.skip_e1:
        print("E1 -- mode blindness")
        run_e1(cfg, results_dir)
    if not args.skip_e2:
        print("E2 -- low-visibility intervention")
        run_e2(cfg, results_dir)
    if not args.skip_e4:
        print("E4 -- sampling cost of low-visibility mass")
        run_e4(cfg, results_dir)
    if args.e3:
        print("E3 -- residual mode MMSE on labelled corpora")
        from mpi.realdata import run_e3

        run_e3(
            cfg,
            corpora=tuple(args.corpora),
            tokenizer=args.tokenizer,
            n_docs=args.n_docs,
            doc_tokens=args.doc_tokens,
            n_windows=args.n_windows,
            n_bootstrap=args.bootstrap,
            calibration=args.calibration,
            results_dir=results_dir,
        )

    print(f"\nTotal {time.time()-t0:.1f}s.  Artifacts in {results_dir}")
    print("Next: python scripts/make_figures.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
