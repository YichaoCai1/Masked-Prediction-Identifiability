# Masked-Prediction Identifiability

Reference implementation of the five experiments accompanying *On the
Identifiability of Masked Prediction: Mode Blindness and Mask Schedules*.

The paper asks when a masked-prediction objective can identify the **mode
weight** of a multi-modal data law — and shows that under ordinary mask
schedules it cannot, while a vanishing amount of near-blank mask restores it.
Three of the five experiments compute the relevant quantities *exactly*, one
trains a model to check the prediction survives optimisation, and one measures
the single quantity that is a property of real data.

| # in paper | What it does | Kind | CLI | Module |
|---|---|---|---|---|
| **1** | mode blindness | exact enumeration | on by default | `e1.py` |
| **2** | low-visibility intervention | exact, by schedule linearity | on by default | `e2.py` |
| **3** | does SGD reproduce it? | trained, 25 runs | `--e5` | `e5.py` |
| **4** | residual mode MMSE on corpora | measured, needs network | `--e3` | `realdata.py` |
| **5** | what the intervention costs | exact | on by default | `e4.py` |

> **The module names do not match the paper's numbering.** They predate the
> paper's final ordering and are kept so that the committed artifacts and
> `config_hash` values stay valid. Use the table above; the CLI flags are the
> ones in the `CLI` column, not the paper's numbers.

Python ≥ 3.10. The three exact experiments reproduce from a clean checkout in
about 25 seconds. The trained one adds ~30 min on a CPU, the corpus one ~12 min
plus a download.

---

## The experiments

The data law is a two-mode mixture $p = w\,p^+ + (1-w)\,p^-$. A *witness*
$q_\lambda = \lambda\,p^+ + (1-\lambda)\,p^-$ differs from $p$ only in the
mixing weight, so its total variation distance from $p$ is exactly
$|w - \lambda|$ and never shrinks. Whether a masked objective can *see* that
difference is measured by the **masked log discrepancy**

$$
\mathfrak{D}_\mu(p \,\|\, q) \;=\; \mathbb{E}_{K \sim \mu}\ \mathbb{E}_{X_V \sim p_V}\
D_\mathrm{KL}\big( p(X_K \mid X_V) \,\|\, q(X_K \mid X_V) \big)
$$

over mask sets $K$ drawn from a schedule $\mu$, with $V$ the visible
complement. The mechanism is that $\mathfrak{D}$ collapses exponentially as the
visible block grows while $|w-\lambda|$ stays put — two laws a masked objective
cannot tell apart, though they are far away from each other. The rate is
governed by the **residual mode uncertainty**
$\mathrm{mmse}(Z \mid X_V) = \mathbb{E}[\beta(1-\beta)]$, where
$\beta = P(\text{mode is } {+} \mid X_V)$.

**1 — mode blindness** (`e1.py`). Two synthetic laws on $\{-1,+1\}^N$, with $N$
odd so the magnetisation $S(x)$ is never zero and the modes partition the cube
cleanly:

| Law | Definition | Role |
|---|---|---|
| **P** (product) | $\nu^+$ i.i.d. with $P(+1)=(1+\theta)/2$, restricted to $S(x)>0$; $\theta \in \{0.6,\,0.8\}$ | primary — every constant is closed-form, so results are prediction-vs-measurement, never fitted |
| **C** (Curie–Weiss) | $p(x) \propto \exp(\beta\, S(x)^2 / 2N)$, $\beta = 1.5$ | robustness — two modes, but no closed form |
| **C** (null control) | same, $\beta = 0.5$ | **mechanism absent**: one mode, so no blindness should appear |

Both are exchangeable and spin-flip symmetric, so the computation reduces to an
exact enumeration over the visible magnetisation rather than over $2^N$ states,
carried in the log domain so values down to $\mathfrak{D} \approx 10^{-308}$
stay representable — that is what makes the decay measurable over 170+ decades
instead of hitting an underflow floor. Grid: $N \in \{127, 255, 511, 1023\}$,
$w \in \{0.5,\,0.9\}$, visible size $m$ dyadic $\cup\ \{0,1,2,4\} \cup \{N/4,
N/2, 3N/4\}$, and $\lambda$ on 201 points.

**2 — low-visibility intervention** (`e2.py`). The remedy the paper proposes:
keep the schedule, but put a small probability $\pi$ on a nearly blank mask — a
two-point schedule $\mu = (1-\pi)\,\delta_m + \pi\,\delta_s$ with $s \ll m$,
where $s = 0$ is the full mask. Because $\mathfrak{D}_\mu$ is *linear* in $\mu$,
every quantity is an exact convex combination of two cells from experiment 1, so
this stage adds no enumeration at all. It reports the curvature at $\lambda = w$
and the **recovery radius** $r(\varepsilon)$, the largest mode-weight error still
resolvable at tolerance $\varepsilon$.

**3 — does the prediction survive optimisation?** (`e5.py`, `--e5`). Everything
above is a statement about the population objective. This stage trains a small
model by SGD on samples from Law P and checks that the blindness and the
recovery both show up in what optimisation actually converges to. The law is
exchangeable, so the mode posterior depends on the visible block only through
the sufficient statistics $(r/N, m/N)$ — a $2\!\to\!128\!\to\!128\!\to\!1$ MLP
predicting $\mathrm{logit}\,\hat\beta$ suffices, and the implied mode weight is
read off the *trivial fibre*, $\hat w = \sigma(\mathrm{logit}\,\hat\beta(0,0))$,
i.e. what the model believes with nothing visible. Five schedules
(the blind one, plus $\pi \in \{10^{-4},10^{-3},10^{-2},10^{-1}\}$ of full mask)
× 5 seeds × 50 000 steps, batch 512, Adam at $10^{-3}$. CPU by default: at this
size a GPU is measurably *not* faster (18.5 s vs 17.5 s over 3000 steps), so
`pick_device` returns CPU unless `--e5-device` says otherwise.

**4 — does the shape occur in real data?** (`realdata.py`, `--e3`). The
theory's structural conditions are statements about $\mathrm{mmse}(Z \mid X_V)$
— a property of the **data law**, not of any trained model. This stage measures
it on corpora where the "mode" is a real regime with a ground-truth label:
Python code vs English prose, and German vs English. The design is governed by a
**direction problem**: for any predictor,
$\mathbb{E}[(Z-\hat\beta)^2] \geq \mathrm{mmse}(Z \mid X_V)$, so a classifier
yields an *upper* bound and never a lower one. Hence two arms — a smoothed
backoff $n$-gram posterior at $|V| \leq 8$, and a calibrated linear classifier
at $|V| \geq 12$ on two **nested** feature sets, so that a flattening curve can
be attributed to the corpus rather than to the model class saturating. Cells
failing the calibration check are censored and *reported*, not silently dropped.

**5 — what the intervention costs** (`e4.py`). The others are population or
asymptotic statements; this one asks how many masked examples it takes to *see*
a discrepancy they certify is nonzero. It samples nothing either. The
per-example excess loss
$S = \log p(X_K \mid X_V) - \log q(X_K \mid X_V)$ is **two-point** on each
fibre — given the visible block, the masked block decides the mode — so its
second moment is exactly as computable as its mean, and
$\mathbb{E}[S] = \mathfrak{D}_\mu$ exactly. It reports
$n^\star = (\mathrm{sd}[S]/\mathbb{E}[S])^2$, the examples needed for a
one-sigma detection, and the raw per-example loss $H(X_K \mid X_V)$ with its
variance decomposed.

### What comes out

- **1** — exponential decay in the visible size at the predicted Chernoff rate
  (to 0.27 % at $N = 1023$) while total variation stays flat; the null control
  shows no decay.
- **2** — curvature $\propto \pi$, recovery radius $\propto \pi^{1/2}$ and
  $\varepsilon^{1/2}$. On Law P at $w = 1/2$ the measured exponents are
  $+0.9965$ to $+1.0000$, $-0.5000$ to $-0.4894$, and $+0.4947$ to $+0.4998$.
  Off that grid the curvature exponent falls as low as $0.47$ — the prediction
  is that $\varkappa$ is *affine* in $\pi$, and the slope is $1$ only where the
  boost term dominates the base term.
- **3** — the dichotomy transfers to SGD. Under the blind schedule the five
  seeds do not agree on $w$ at all, ending at
  $\hat w \in \{0.0002, 0.0003, 0.0022, 0.6129, 0.6921\}$ — three runs give the
  $+$ mode essentially no mass, two settle at interior weights — for a mean
  error of $0.36$. Any positive full-mask mass removes the failure; the error
  then falls as $\pi^{-0.44 \pm 0.09}$, reaching $9.5 \times 10^{-4}$ at
  $\pi = 10^{-1}$, a factor of 380 below the blind baseline.
- **4** — the required shape is present (91× suppression on the bilingual pair,
  183× on code vs prose), but its functional form is a **power law**, not the
  exponential the theory assumes ($R^2 = 0.977$ vs $0.939$ on the bilingual
  pair). A strictly richer classifier leaves that exponent essentially
  untouched there ($-1.150 \to -1.149$) and steepens it by 18 % on code vs
  prose ($-0.942 \to -1.109$), so the law belongs mostly to the corpus rather
  than to the estimator's order-blindness.
- **5** — the cost is exactly $1/\pi$: at the mass an untruncated schedule
  provides ($\pi \approx 1/N$), detection needs $\sim 1.6 \times 10^4$
  examples; with no low-visibility mass at all, $\sim 10^{116}$.

---

## Install

```bash
pip install -r requirements.txt
```

Only `numpy`, `scipy`, `pandas`, `matplotlib` and `mpmath` are needed for the
three exact experiments. Experiment 3 needs `torch` (CPU build is enough);
experiment 4 needs `scikit-learn`, `datasets`, `transformers` and network
access. Experiment 4 can be run fully offline against a synthetic fixture
(`--corpora markov`) if you only want to exercise the pipeline.

---

## Quickstart

```bash
python -m pytest tests -q          # 177 tests, ~10 s  -- run before any figure
python scripts/run_all.py          # experiments 1, 2, 5: exact enumeration, ~25 s
python scripts/audit_numerics.py   # mpmath cross-check at 50 digits, ~3 s
python scripts/make_figures.py     # figure M, figure M_E3, table M, A1-A10
```

Artifacts land in `results/` as Parquet; figures in `figures/` as PDF, both
composed (`figure_M.pdf`) and panel by panel (`figure_M/panel_a.pdf`, …);
Table M in `figures/table_M.csv` and `figures/table_M.tex`.

To reproduce everything as committed, including the trained and real-corpus
stages:

```bash
python scripts/run_all.py                                    # 1, 2, 5   ~25 s
python scripts/audit_numerics.py                             #           ~3 s
python scripts/run_all.py --skip-e1 --skip-e2 --skip-e4 --e5 # 3        ~30 min
python scripts/run_all.py --skip-e1 --skip-e2 --skip-e4 --e3 \
    --corpora code_prose bilingual --tokenizer hf:gpt2 \
    --n-docs 16000 --n-windows 15000 --bootstrap 1000        # 4        ~12 min
python scripts/make_figures.py
```

Each stage writes its own tables, so they can be run in any order or separately.
For a fast smoke test of experiment 3, `--e5-steps 5000 --e5-seeds 2` finishes in
about a minute (the trend survives; the seed spread does not tighten).

Without network access, skip the `--e3` line. Figures degrade gracefully: any
figure whose tables are missing prints a warning and is skipped; everything else
is unaffected.

---

## Data for experiment 4

Experiments 1, 2, 3 and 5 need no data — they enumerate or sample their laws.
Experiment 4 streams two labelled corpora from HuggingFace; **nothing needs
downloading by hand**, as `run_all.py --e3` streams, tokenises, packs and caches
on first use.

| Corpus key | `Z = 1` | `Z = 0` | Licence |
|---|---|---|---|
| `code_prose` | `codeparrot/github-code-clean`, Python, MIT / Apache-2.0 / BSD | `allenai/c4`, config `en` | per-file / ODC-BY |
| `bilingual` | `Helsinki-NLP/opus-100` `de-en`, German side | same, English side | CC-BY-4.0 |
| `markov` | synthetic order-1 Markov source | second synthetic source | n/a (offline) |

- **Tokenizer.** One shared, neutral tokenizer for both sides — GPT-2 BPE by
  default. Do not substitute a code-specialised one; it leaks the label through
  the tokenisation itself. `--tokenizer bytes` is an offline fallback.
- **Cache.** `.cache/e3/`, keyed by `(corpus, side, tokenizer, n_docs,
  doc_tokens)`; ~300 MB at default size. Delete it to force a re-download.
- **Source exhaustion is handled.** OPUS-100's English side is shorter than its
  German side; packing truncates both to the common minimum so `w = 1/2` stays
  exact, reporting `balanced to N docs/side (source exhausted)`.

Offline smoke test of the whole corpus pipeline:

```bash
python scripts/run_all.py --skip-e1 --skip-e2 --skip-e4 --e3 \
    --corpora markov --tokenizer bytes --n-docs 2000 --n-windows 2000
```

---

## Options

`python scripts/run_all.py --help` lists everything; these are the ones worth
knowing.

| Option | Default | Meaning |
|---|---|---|
| `--quick` | off | reduced grid (Law P only, `N ∈ {127,255}`, `w = 0.5`) — smoke test |
| `--laws KEY ...` | all four | `product-0.6`, `product-0.8`, `cw-1.5`, `cw-0.5` |
| `--N N ...` | `127 255 511 1023` | sequence lengths (must be odd) |
| `--w W ...` | `0.5 0.9` | true mode weights |
| `--kl-impl {stable,branched}` | `stable` | binary-KL evaluation scheme |
| `--skip-e1`, `--skip-e2`, `--skip-e4` | off | skip experiment 1, 2 or 5 |
| `--results-dir DIR` | `results` | where to write the tables |
| `--e5` | off | run experiment 3 (trained; needs `torch`) |
| `--e5-seeds`, `--e5-steps` | `5`, `50000` | seeds per schedule, steps per run |
| `--e5-device` | `cpu` | `cuda` is *slower* at this model size — see above |
| `--e3` | off | run experiment 4 (needs network) |
| `--corpora ...` | `code_prose bilingual` | any of `code_prose`, `bilingual`, `markov` |
| `--n-docs`, `--n-windows` | `20000`, `20000` | sample sizes; cost scales with `n_windows` × `\|V\|` |
| `--bootstrap` | `1000` | bootstrap resamples, over documents |
| `--calibration` | `isotonic` | `isotonic` or `platt` |

`make_figures.py` takes `--only M A9 ...` to rebuild a subset, `--png` to emit
PNG previews alongside the PDFs, and `--results-dir` / `--out-dir`.

---

## Outputs

Sixteen Parquet tables in `results/` (CSV fallback), each stamped with
`config_hash`, `git_hash`, `kl_impl`, library versions and platform. Every
figure is a **pure function of those tables** — nothing is recomputed at plot
time, so any figure traces back to the run that produced it.

Twelve composed figures and 38 standalone panels in `figures/`, PDF only:

```
figures/figure_M.pdf              composed contact sheet, panels titled (a), (b), …
figures/figure_M/panel_a.pdf      each panel standalone and *untitled*, at exactly
figures/figure_M/panel_b.pdf      its declared size, so panels line up when placed
figures/figure_M/panel_c.pdf      side by side in a \subfigure row
```

Panel titles are drawn only on the composed sheet — on a standalone panel they
would duplicate the `\subcaption`. Panels keep their run context (`N`, `m_base`,
the witness) as muted in-axes notes.

---

## Code layout

```
src/mpi/
  config.py     grids, run matrix, closed-form predicted constants, band constants
  laws.py       the only law-dependent line: unnormalised log-weight of (a, b)
  core.py       log h^tau, ell, pooled law, the two binary-KL schemes  [exact]
  estimands.py  D, Delta, U, F, rho, boosted-schedule linearity        [exact]
  fits.py       window OLS, anchored exponential fit, curvature, bisection radius
  brute.py      explicit 2^N reference implementation (tests only)
  highprec.py   mpmath at 50 digits (T3 and the audit only)
  e1.py         experiment 1 driver
  e2.py         experiment 2 driver
  e5.py         experiment 3: sampler, MLP, training loop, w-hat readout  [torch]
  realdata.py   experiment 4: corpora, tokenizer, both arms, calibration, bootstrap
  e4.py         experiment 5: per-example moments, detection cost, raw loss [exact]
  tableio.py    Parquet/CSV writing with provenance stamping
  style.py      colour and mark decisions for every figure
  figures.py    figure M, table M, A1-A10
scripts/
  run_all.py        experiments 1, 2, 5 (+ 3 with --e5, + 4 with --e3)
  audit_numerics.py float64 vs mpmath
  make_figures.py   all figures from frozen artifacts
tests/
  test_exact.py     T1-T15 + brute-force end-to-end
```

A law is *only* a `weight_fn` returning the unnormalised log-weight of a
visible/hidden count pair; everything downstream is law-agnostic. Adding a third
law means one function in [src/mpi/laws.py](src/mpi/laws.py) and one entry in
[src/mpi/config.py](src/mpi/config.py).

---

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `no E1 artifacts in results/` when plotting | run `python scripts/run_all.py` first |
| A7 / A8 / A10 say "not run", or `figure_M_E3` is missing | run `scripts/audit_numerics.py` / `run_all.py --e3` / `run_all.py --e5` |
| `parquet unavailable … wrote *.csv` | `pyarrow` missing; readers fall back to CSV automatically |
| experiment 3 seems to hang | it is 25 runs × 50 000 steps, ~30 min; lower `--e5-steps` to check progress |
| `source exhausted after only N/M documents` | the stream ran out; lower `--n-docs` or `--doc-tokens` |
| `datasets` refuses `trust_remote_code` | pin `datasets<3`, or edit `CORPORA` in [src/mpi/realdata.py](src/mpi/realdata.py) |
| experiment 4 too slow / out of memory at large `\|V\|` | lower `--n-windows` |
| want to re-download the corpora | delete `.cache/e3/` |
| `N must be odd` | the construction needs odd `N` so `S(x) ≠ 0` |
