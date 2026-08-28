# An Identifiability Theory of Masked Prediction: Mode Blindness and Mask Schedules

Reference implementation and reproducibility artifacts for [An Identifiability Theory of Masked Prediction: Mode Blindness and Mask Schedules](https://arxiv.org/abs/2608.01383).

The repository contains exact-enumeration experiments, a numerical audit, an optimization experiment, a real-corpus experiment, and the scripts used to produce every table and figure in the paper. The committed `results/` tables are sufficient to rebuild all figures without rerunning the experiments.

Python 3.10 or newer is required.

## Install

```bash
python -m pip install -r requirements.txt
```

The full requirements include PyTorch and the Hugging Face dependencies used by the optimization and real-corpus experiments. Rebuilding figures from the committed artifacts only needs NumPy, pandas, PyArrow, and Matplotlib.

## Rebuild the figures from committed artifacts

```bash
python scripts/make_figures.py
```

This writes the composed PDFs and standalone panel PDFs to `figures/`. To rebuild selected figures or also write PNG previews:

```bash
python scripts/make_figures.py --only M A8
python scripts/make_figures.py --png
```

No experiment is recomputed by the figure script.

## Run the experiments

First run the test suite:

```bash
python -m pytest tests -q
```

Run the exact experiments and the high-precision numerical audit:

```bash
python scripts/run_all.py
python scripts/audit_numerics.py
```

Run the optimization experiment:

```bash
python scripts/run_all.py --skip-e1 --skip-e2 --skip-e4 --e5
```

Run the real-corpus experiment (network access is required):

```bash
python scripts/run_all.py --skip-e1 --skip-e2 --skip-e4 --e3 --corpora code_prose bilingual --tokenizer hf:gpt2 --n-docs 16000 --n-windows 15000 --bootstrap 1000
```

Finally, rebuild all figures and tables:

```bash
python scripts/make_figures.py
```

The experiment numbering in the code predates the paper's final ordering:

| Paper experiment | What it runs | Code entry point | Default |
|---|---|---|---|
| 1 | Exactly enumerates masked discrepancies for the product and Curie-Weiss laws to measure mode blindness. | `e1.py` via `run_all.py` | yes |
| 2 | Combines exact cells under low-visibility mask schedules to measure curvature and recovery radius. | `e2.py` via `run_all.py` | yes |
| 3 | Trains a small MLP across mask schedules and seeds to test whether optimization reproduces blindness and recovery. | `e5.py` via `run_all.py --e5` | no |
| 4 | Measures residual mode uncertainty on code-versus-prose and German-versus-English corpora using posterior and calibrated-classifier estimators. | `realdata.py` via `run_all.py --e3` | no |
| 5 | Computes exact per-example moments and the sample complexity of detecting the low-visibility intervention. | `e4.py` via `run_all.py` | yes |

Each stage writes separate tables, so the commands may be run independently or in any order. Use `python scripts/run_all.py --help` for the complete set of grid, corpus, tokenizer, calibration, seed, step, and device options.

## Smoke tests

A reduced exact run:

```bash
python scripts/run_all.py --quick
```

A shorter optimization run:

```bash
python scripts/run_all.py --skip-e1 --skip-e2 --skip-e4 --e5 --e5-steps 5000 --e5-seeds 2
```

An offline test of the real-data pipeline using the synthetic Markov fixture:

```bash
python scripts/run_all.py --skip-e1 --skip-e2 --skip-e4 --e3 --corpora markov --tokenizer bytes --n-docs 2000 --n-windows 2000
```

## Data

Experiments 1, 2, 3, and 5 generate or enumerate their data internally. Experiment 4 streams and caches its corpora on first use:

| Corpus key | Sources | License |
|---|---|---|
| `code_prose` | `codeparrot/github-code-clean` (Python) and `allenai/c4` (`en`) | per-file permissive licenses / ODC-BY |
| `bilingual` | `Helsinki-NLP/opus-100` (`de-en`) | CC-BY-4.0 |
| `markov` | synthetic offline fixture | not applicable |

The cache is stored under `.cache/e3/`. Both classes use one shared tokenizer; the reported runs use GPT-2 BPE (`--tokenizer hf:gpt2`), while `--tokenizer bytes` is available for offline checks.

## Repository layout

```text
src/mpi/                 experiment and plotting modules
scripts/run_all.py       experiment runner
scripts/audit_numerics.py
scripts/make_figures.py  figure/table builder from saved results
tests/test_exact.py      exact and brute-force checks
results/                 committed result tables
figures/                 committed composed and panel PDFs
```

Result tables include configuration, implementation, library, platform, and Git provenance where available. Parquet is the primary format; table readers also support the CSV fallback used when PyArrow is unavailable.

## Citation

```bibtex
@article{cai2026identifiability,
  title   = {An Identifiability Theory of Masked Prediction: Mode Blindness and Mask Schedules},
  author  = {Cai, Yichao and Shi, Javen Qinfeng},
  journal = {arXiv preprint arXiv:2608.01383},
  year    = {2026}
}
```
