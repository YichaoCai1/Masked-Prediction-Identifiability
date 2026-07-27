#!/usr/bin/env python
"""Regenerate every figure and Table M from the frozen artifacts in ``results/``.

    python scripts/make_figures.py                # all figures + Table M
    python scripts/make_figures.py --only M A2    # just those
    python scripts/make_figures.py --png          # also emit PNG previews

Each figure is written twice: composed as ``figures/figure_<name>.pdf`` and
panel by panel as ``figures/figure_<name>/panel_<key>.pdf``, so a single panel
can be dropped straight into a paper subfigure slot.

No figure recomputes anything: they are pure functions of the result tables,
so a figure can always be traced back to the run that produced it via the
``config_hash`` / ``git_hash`` columns those tables carry.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mpi.config import FIGURES_DIR, RESULTS_DIR  # noqa: E402
from mpi.figures import make_all_figures  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-dir", default=str(RESULTS_DIR))
    p.add_argument("--out-dir", default=str(FIGURES_DIR))
    p.add_argument("--only", nargs="*", default=None,
                   help="subset of {M, A1..A8}")
    p.add_argument("--png", action="store_true",
                   help="also emit PNG previews alongside the PDFs")
    args = p.parse_args(argv)

    print(f"Reading artifacts from {args.results_dir}")
    make_all_figures(Path(args.results_dir), Path(args.out_dir), args.only, args.png)
    print(f"Figures in {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
