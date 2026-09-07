# An Identifiability Theory of Masked Prediction

Reference implementation and reproducibility package for **“An Identifiability Theory of Masked Prediction: Mode Blindness and Mask Schedules.”** ([Paper](https://arxiv.org/abs/2608.01383))

---

## 🚀 Overview

Masked prediction can accurately model local conditional distributions while remaining nearly insensitive to errors in the global frequencies of separated data modes. This repository studies that phenomenon and the role of the mask schedule through three complementary components:

1. **Exact controlled calculations** measure mode blindness and show how low-visibility masks restore sensitivity.
2. **Direct mode-weight optimization** tests whether residual mode uncertainty predicts optimization speed inside a coherent joint family.
3. **Natural-text measurements** explore whether the required uncertainty patterns occur for code versus prose and German versus English.

The controlled studies use exact sums or direct scalar optimization. The natural-text study is a preliminary appendix exploration. All study names, output tables, and figures describe the quantity they contain. Generated tables also record configuration and Git hashes so that figures can be traced to their source run.

---

## 📦 1. Installation

Python **3.10 or newer** is required. A GPU is not required.

```bash
git clone https://github.com/YichaoCai1/Masked-Prediction-Identifiability.git
cd Masked-Prediction-Identifiability
python -m venv .venv
```

Activate the environment on Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The exact and optimization studies use NumPy, SciPy, and pandas. The natural-text study additionally uses Hugging Face Datasets, Transformers, and scikit-learn and requires network access on its first run.

---

## ⚡ 2. Quick start

Run the test suite and a reduced offline workflow:

```bash
python -m pytest tests -q
python scripts/run_studies.py --quick --results-dir smoke_results
python scripts/make_figures.py --results-dir smoke_results --out-dir smoke_figures --only mode_blindness_and_recovery --png
```

The files in `smoke_results/` verify that the pipeline works. They use reduced grids and must not be used for numerical claims in the paper.

---

## 🧪 3. Reproducing the experiments

The commands below write to `reproduced_results/`. Use a different value of `--results-dir` if you want to keep the included rerun tables untouched.

### 3.1 Exact controlled studies

The default command runs all three offline studies:

```bash
python scripts/run_studies.py --results-dir reproduced_results
python scripts/audit_numerics.py --results-dir reproduced_results
```

| Study | Question | Main tables |
|---|---|---|
| `mode-blindness` | How quickly does masked discrepancy vanish as more coordinates become visible? | `profiles`, `summary`, `fits`, `numerics`, `constants` |
| `low-visibility-intervention` | How does assigning probability to low-visibility masks restore curvature and reduce the recovery radius? | `boost`, `boostfits`, `uscale` |
| `low-visibility-sampling-cost` | How many masked examples are required for the restored signal to match one standard error, and what raw-loss cost is incurred? | `sampling` |

The main setting is the symmetric conditioned-product mixture with `theta=0.8` and mode weight `w=0.5`. Additional exact evaluations use `theta=0.6`, `w=0.9`, and the Curie–Weiss model at inverse temperatures `beta=1.5` and `beta=0.5`. The system-size grid is `N = {127, 255, 511, 1023}`.

To run only selected studies, pass their semantic names:

```bash
python scripts/run_studies.py \
  --studies mode-blindness low-visibility-intervention \
  --results-dir reproduced_results
```

Use `python scripts/run_studies.py --help` to inspect the exact-distribution, dimension, mode-weight, and numerical-evaluation options.

### 3.2 Direct mode-weight optimization

Run the reported population gradient-descent and minibatch-SGD experiment with:

```bash
python scripts/run_studies.py \
  --studies mode-weight-optimization \
  --results-dir reproduced_results
```

The trainable family is

$$
q_\lambda=\lambda p^+ + (1-\lambda)p^-,
\qquad
\lambda=\operatorname{sigmoid}(a).
$$

Only the scalar logit `a` is optimized. Every masked conditional is induced by the same joint distribution $q_\lambda$, and the primary objective comparison uses exact population excess risk.

The reported protocol uses:

- conditioned-product parameter `theta=0.8` and true weight `w=0.5`;
- candidate dimensions `N = {31, 63, 127}` and baseline visibility `m_base = floor(N/2)`;
- low-visibility sizes `s = {0, 1, 4}` and schedule masses `pi = {0, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1}`;
- population initial weights `{0.1, 0.3, 0.4, 0.6, 0.7, 0.9}` and stochastic initial weights `{0.4, 0.6}`;
- batch size 512, five SGD seeds, checkpoints every 100 steps, a fixed 10,000-example validation stream, and a maximum of 100,000 steps;
- three consecutive successful checkpoints before a stochastic half-time is accepted, with non-crossing runs retained as right-censored;
- 1,000 bootstrap replicates for the median stochastic half-time interval.

Before the trajectories are inspected, exact calibration selects `N=63` and `m_base=31` using the predeclared selection rule. A fixed-seed stability sweep tests learning rates `{0.025, 0.05, 0.1}` on the highest-curvature configuration; `0.1` is selected once and then held fixed across every reported run. See the complete[machine-readable protocol](reproduced_results/mode_weight_config.json) and [acceptance summary](reproduced_results/mode_weight_RESULTS.md).

### 3.3 Natural-text exploration

The reported natural-text run is:

```bash
python scripts/run_studies.py \
  --studies corpus-mode-uncertainty \
  --results-dir reproduced_results \
  --corpora code_prose bilingual \
  --tokenizer hf:gpt2 \
  --n-docs 16000 \
  --n-windows 15000 \
  --bootstrap 1000
```

The first run downloads and caches the datasets and GPT-2 tokenizer. Short and long visible contexts are evaluated with separate estimators because the two parts of the analysis require different directions of control: a backoff posterior estimator is used at short contexts, while a calibrated classifier is used at long contexts. Cells that fail the predeclared support or calibration checks are marked as censored rather than used in the fitted rates.

For a fully offline pipeline check, use the synthetic Markov fixture:

```bash
python scripts/run_studies.py \
  --quick \
  --studies corpus-mode-uncertainty \
  --results-dir smoke_corpus_results \
  --corpora markov \
  --tokenizer bytes
```

---

## 📊 4. Generating figures

Figures are rendered only from saved tables; figure generation never reruns an experiment.

Generate every figure supported by the available artifacts:

```bash
python scripts/make_figures.py \
  --results-dir reproduced_results \
  --out-dir reproduced_figures \
  --png
```

Generate only the two main experiment figures:

```bash
python scripts/make_figures.py \
  --results-dir reproduced_results \
  --out-dir reproduced_figures \
  --only mode_blindness_and_recovery mode_weight_optimization \
  --png
```

PDF is the primary output format; `--png` also writes convenient previews. Multi-panel figures are saved both as assembled figures and as individually named panels under `figure_<name>/`. The natural-text result facets are saved only as `panel_code_prose.pdf` and `panel_german_english.pdf` in the corpus diagnostics directory.

An all-figure build reports and skips optional figures when their source study has not been run. A figure explicitly requested with `--only` instead exits with a nonzero status when a required table is missing.

<details>
<summary>Available semantic figure selectors</summary>

- `mode_blindness_and_recovery`
- `masked_discrepancy_profiles`
- `blindness_decay_rates`
- `low_visibility_intervention`
- `residual_uncertainty_by_visibility`
- `boundary_flow_mechanism`
- `asymmetric_mixture_weights`
- `numerical_stability_audit`
- `corpus_mode_uncertainty_diagnostics`
- `low_visibility_detection_cost`
- `low_visibility_loss_cost`
- `mode_weight_optimization`

</details>

---

## 📁 5. Outputs and repository structure

```text
src/mpi/
  mode_blindness.py                    exact mode-blindness calculations
  low_visibility_intervention.py       mask-schedule intervention
  low_visibility_sampling_cost.py      signal and sample-cost calculations
  mode_weight_optimization.py          direct scalar optimization
  corpus_mode_uncertainty.py           natural-text exploration
  figures.py                           figure and table builders
scripts/
  run_studies.py                       semantic study runner
  make_figures.py                      rendering from saved tables
  audit_numerics.py                    independent high-precision audit
tests/                                 exact and reproducibility tests
reproduced_results/                    current rerun artifacts
figures/                               current paper figures
```

Result tables are written as Parquet files. If PyArrow is unavailable, the table writer falls back to CSV. Every table includes the repository configuration hash, Git commit, numerical branch settings, Python and NumPy versions, and platform information.

The important output groups are:

- `profiles`, `summary`, `fits`, `numerics`, and `constants` for exact mode-blindness geometry;
- `boost`, `boostfits`, and `uscale` for low-visibility interventions;
- `sampling` for signal, variance, and sample-cost calculations;
- `mode_weight_*` plus `mode_weight_config.json` for direct optimization;
- `corpus_uncertainty*` for the natural-text exploration;
- `audit` for the independent high-precision comparison.

---

## 🗂️ 6. Data sources and cache

The controlled studies generate or enumerate their distributions internally. The natural-text study streams the following sources:

| Corpus key | Two labelled sources | Upstream license |
|---|---|---|
| `code_prose` | Python files from `codeparrot/github-code-clean`; English web prose from `allenai/c4` | per-file MIT, Apache-2.0, BSD-2-Clause, or BSD-3-Clause; ODC-BY |
| `bilingual` | German and English from `Helsinki-NLP/opus-100` (`de-en`) | CC-BY-4.0 |
| `markov` | Synthetic offline fixture | not applicable |

Downloaded and tokenized corpus data are stored under `.cache/corpus_mode_uncertainty/`. Both classes in a comparison use the same neutral tokenizer so that the tokenizer itself does not reveal the class label. The reported run uses GPT-2 BPE; byte tokenization is provided only for offline checks.

---

## ✅ 7. Validation

Run the default test suite:

```bash
python -m pytest tests -q
```

The suite checks the exact identities, brute-force small-system comparisons, optimization invariants, censoring rules, deterministic provenance, and figure interfaces. Tests marked `slow` are excluded by the project’s default pytest configuration.

The independent arbitrary-precision audit can be rerun with:

```bash
python scripts/audit_numerics.py --results-dir reproduced_results
```

---

## 🧾 Citation

If you find this work useful, please cite:

```bibtex
@article{cai2026identifiability,
  title   = {An Identifiability Theory of Masked Prediction: Mode Blindness and Mask Schedules},
  author  = {Cai, Yichao and Shi, Javen Qinfeng},
  journal = {arXiv preprint arXiv:2608.01383},
  year    = {2026}
}
```

---

## License

The code is released under the [MIT License](LICENSE). The external corpora remain subject to their respective upstream licenses.
