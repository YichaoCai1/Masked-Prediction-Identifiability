#!/usr/bin/env python
"""Run one or more named studies and write machine-readable result tables.

Examples:

    python scripts/run_studies.py
    python scripts/run_studies.py --studies mode-weight-optimization
    python scripts/run_studies.py --studies corpus-mode-uncertainty
    python scripts/run_studies.py --quick --results-dir smoke_results

See the README for complete reproduction commands.
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
from mpi.low_visibility_intervention import run_low_visibility_intervention  # noqa: E402
from mpi.low_visibility_sampling_cost import (  # noqa: E402
    run_low_visibility_sampling_cost,
)
from mpi.mode_blindness import run_mode_blindness  # noqa: E402


STUDIES = (
    "mode-blindness",
    "low-visibility-intervention",
    "mode-weight-optimization",
    "corpus-mode-uncertainty",
    "low-visibility-sampling-cost",
)
DEFAULT_STUDIES = (
    "mode-blindness",
    "low-visibility-intervention",
    "low-visibility-sampling-cost",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--studies",
        nargs="+",
        choices=STUDIES,
        default=list(DEFAULT_STUDIES),
        help="named studies to run (default: the three exact offline studies)",
    )
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use smoke-test defaults; never use quick outputs for paper claims",
    )

    exact = parser.add_argument_group("exact synthetic-law grid")
    exact.add_argument("--laws", nargs="*", default=None,
                       help="law keys, e.g. product-0.8 cw-1.5")
    exact.add_argument("--dimensions", nargs="*", type=int, default=None)
    exact.add_argument("--mode-weights", nargs="*", type=float, default=None)
    exact.add_argument("--cw-sweep", action="store_true",
                       help="add the optional Curie-Weiss beta={0.9,1.1} sweep")
    exact.add_argument("--kl-impl", choices=["stable", "branched"], default=None)
    exact.add_argument("--branch-threshold", type=float, default=None,
                       help="threshold used only by the branched binary-KL evaluator")

    mode = parser.add_argument_group("direct mode-weight optimization")
    mode.add_argument("--mode-weight-dimensions", nargs="*", type=int, default=None,
                      help="candidate odd dimensions (default: 31 63 127)")
    mode.add_argument("--mode-weight-schedule-masses", nargs="*", type=float,
                      default=None, help="low-visibility schedule masses")
    mode.add_argument("--mode-weight-boost-sizes", nargs="*", type=int, default=None,
                      help="low-visibility sizes (default: 0 1 4)")
    mode.add_argument("--mode-weight-learning-rate", type=float, default=0.1,
                      help="largest rate in the one-time {eta/4,eta/2,eta} stability sweep")
    mode.add_argument("--mode-weight-batch-size", type=int, default=None)
    mode.add_argument("--mode-weight-max-steps", type=int, default=None)
    mode.add_argument("--mode-weight-checkpoint-interval", type=int, default=None)
    mode.add_argument("--mode-weight-validation-size", type=int, default=None)
    mode.add_argument("--mode-weight-seeds", type=int, default=None)
    mode.add_argument("--mode-weight-population-inits", nargs="*", type=float,
                      default=(0.1, 0.3, 0.4, 0.6, 0.7, 0.9))
    mode.add_argument("--mode-weight-sgd-inits", nargs="*", type=float,
                      default=(0.4, 0.6))
    mode.add_argument("--mode-weight-consecutive-checkpoints", type=int, default=3)
    mode.add_argument("--mode-weight-bootstrap-reps", type=int, default=1000)
    mode.add_argument("--mode-weight-bootstrap-seed", type=int, default=2027)
    mode.add_argument("--mode-weight-population-only", action="store_true")

    corpus = parser.add_argument_group("corpus mode uncertainty")
    corpus.add_argument("--corpora", nargs="*", default=["code_prose", "bilingual"],
                        choices=["code_prose", "bilingual", "markov"])
    corpus.add_argument("--tokenizer", default="hf:gpt2",
                        help="shared tokenizer: 'hf:<name>' or 'bytes' (offline)")
    corpus.add_argument("--n-docs", type=int, default=None,
                        help="packed documents per side")
    corpus.add_argument("--doc-tokens", type=int, default=1152,
                        help="tokens per packed document (must exceed max visible size)")
    corpus.add_argument("--n-windows", type=int, default=None,
                        help="windows per side, split, and visible size")
    corpus.add_argument("--bootstrap", type=int, default=None)
    corpus.add_argument("--calibration", default="isotonic",
                        choices=["isotonic", "platt"])

    return parser


def _run_config(args: argparse.Namespace) -> RunConfig:
    laws = args.laws or [spec.key for spec in ALL_LAWS]
    if args.cw_sweep:
        laws = list(laws) + [spec.key for spec in LAWS_C_SWEEP]
    values: dict = {}
    if args.quick:
        values.update(
            laws=("product-0.6", "product-0.8"),
            N_grid=(127, 255),
            w_grid=(0.5,),
        )
    else:
        values["laws"] = tuple(laws)
        if args.dimensions:
            values["N_grid"] = tuple(args.dimensions)
        if args.mode_weights:
            values["w_grid"] = tuple(args.mode_weights)
    if args.kl_impl:
        values["kl_impl"] = args.kl_impl
    if args.branch_threshold is not None:
        values["branch_T"] = args.branch_threshold
    return RunConfig(**values)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _run_config(args)
    results_dir = Path(args.results_dir)
    studies = tuple(dict.fromkeys(args.studies))

    print("Masked-prediction identifiability -- named study run")
    for key, value in provenance(cfg).items():
        print(f"  {key:>12s}: {value}")
    print(f"  {'studies':>12s}: {', '.join(studies)}")
    print(f"  {'laws':>12s}: {', '.join(cfg.laws)}")
    print(f"  {'N':>12s}: {cfg.N_grid}")
    print(f"  {'w':>12s}: {cfg.w_grid}")
    print()

    started = time.time()
    if "mode-blindness" in studies:
        print("Mode blindness and sensitivity")
        run_mode_blindness(cfg, results_dir)
    if "low-visibility-intervention" in studies:
        print("Low-visibility intervention")
        run_low_visibility_intervention(cfg, results_dir)
    if "low-visibility-sampling-cost" in studies:
        print("Sampling cost of low-visibility mass")
        run_low_visibility_sampling_cost(cfg, results_dir)
    if "mode-weight-optimization" in studies:
        from mpi.mode_weight_optimization import (  # noqa: E402
            DEFAULT_N_GRID,
            DEFAULT_PI_GRID,
            DEFAULT_S_GRID,
            run_mode_weight_optimization,
        )

        print("Direct mode-weight optimization within q_lambda")
        run_mode_weight_optimization(
            cfg,
            results_dir=results_dir,
            theta=0.8,
            w=0.5,
            N_grid=tuple(
                args.mode_weight_dimensions
                or ((31,) if args.quick else DEFAULT_N_GRID)
            ),
            pi_grid=tuple(
                args.mode_weight_schedule_masses
                or ((0.0, 0.01, 0.03, 0.1) if args.quick else DEFAULT_PI_GRID)
            ),
            s_grid=tuple(args.mode_weight_boost_sizes or DEFAULT_S_GRID),
            learning_rate=args.mode_weight_learning_rate,
            batch_size=(args.mode_weight_batch_size or (128 if args.quick else 512)),
            max_steps=(args.mode_weight_max_steps or (1_000 if args.quick else 100_000)),
            checkpoint_every=(
                args.mode_weight_checkpoint_interval or (20 if args.quick else 100)
            ),
            validation_size=(
                args.mode_weight_validation_size or (512 if args.quick else 10_000)
            ),
            n_seeds=(args.mode_weight_seeds or (2 if args.quick else 5)),
            run_sgd=not args.mode_weight_population_only,
            quick=args.quick,
            population_inits=tuple(args.mode_weight_population_inits),
            stochastic_inits=tuple(args.mode_weight_sgd_inits),
            consecutive_successes=args.mode_weight_consecutive_checkpoints,
            bootstrap_reps=args.mode_weight_bootstrap_reps,
            bootstrap_seed=args.mode_weight_bootstrap_seed,
        )
    if "corpus-mode-uncertainty" in studies:
        from mpi.corpus_mode_uncertainty import run_corpus_mode_uncertainty  # noqa: E402

        print("Residual mode uncertainty on labelled corpora")
        run_corpus_mode_uncertainty(
            cfg,
            corpora=tuple(args.corpora),
            tokenizer=args.tokenizer,
            n_docs=args.n_docs or (2_000 if args.quick else 20_000),
            doc_tokens=args.doc_tokens,
            n_windows=args.n_windows or (2_000 if args.quick else 20_000),
            n_bootstrap=args.bootstrap or (100 if args.quick else 1_000),
            calibration=args.calibration,
            results_dir=results_dir,
        )
    print(f"\nTotal {time.time()-started:.1f}s. Artifacts in {results_dir}")
    print(
        "Next: python scripts/make_figures.py "
        f"--results-dir {results_dir} --out-dir reproduced_figures"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
