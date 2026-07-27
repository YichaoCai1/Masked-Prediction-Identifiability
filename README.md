# Masked-Prediction Identifiability

Reference implementation of the four experiments accompanying *On the
Identifiability of Masked Prediction: Mode Blindness and Mask Schedules*.

The paper asks when a masked-prediction objective can identify the **mode
weight** of a multi-modal data law — and shows that under ordinary mask
schedules it cannot, while a vanishing amount of near-blank mask restores it.
This repository computes the relevant quantities *exactly* (E1, E2, E4) and
measures the one that is a property of real data (E3).

- **E1** — mode blindness. Exact enumeration; no sampling, no optimisation.
- **E2** — low-visibility intervention. Exact, by schedule linearity.
- **E3** — residual mode MMSE on labelled corpora. The only stage with
  statistical error, and the only one needing network access.
- **E4** — the sampling cost of that intervention. Exact.

Python ≥ 3.10. Everything reproduces from a clean checkout in about 25 seconds,
plus ~12 minutes for E3 if you want the real-corpus stage.

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

**E1 — mode blindness.** Two synthetic laws on $\{-1,+1\}^N$, with $N$ odd so
the magnetisation $S(x)$ is never zero and the modes partition the cube
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

**E2 — low-visibility intervention.** The remedy the paper proposes: keep the
schedule, but put a small probability $\pi$ on a nearly blank mask — a
two-point schedule $\mu = (1-\pi)\,\delta_m + \pi\,\delta_s$ with $s \ll m$,
where $s = 0$ is the full mask. Because $\mathfrak{D}_\mu$ is *linear* in $\mu$,
every quantity is an exact convex combination of two E1 cells, so E2 adds no
enumeration at all. It reports the curvature at $\lambda = w$ and the
**recovery radius** $r(\varepsilon)$, the largest mode-weight error still
resolvable at tolerance $\varepsilon$.

**E3 — does the shape occur in real data?** The theory's structural conditions
are statements about $\mathrm{mmse}(Z \mid X_V)$ — a property of the **data
law**, not of any trained model. E3 measures it on corpora where the "mode" is
a real regime with a ground-truth label: Python code vs English prose, and
German vs English. The design is governed by a **direction problem**: for any
predictor, $\mathbb{E}[(Z-\hat\beta)^2] \geq \mathrm{mmse}(Z \mid X_V)$, so a
classifier yields an *upper* bound and never a lower one. Hence two arms — a
smoothed backoff $n$-gram posterior at $|V| \leq 8$, and a calibrated linear
classifier at $|V| \geq 12$ on two **nested** feature sets, so that a
flattening curve can be attributed to the corpus rather than to the model class
saturating. Cells failing the calibration check are censored and *reported*,
not silently dropped.

**E4 — what the intervention costs.** E1–E3 are population statements; E4 asks
how many masked examples it takes to *see* a discrepancy they certify is
nonzero. It samples nothing either. The per-example excess loss
$S = \log p(X_K \mid X_V) - \log q(X_K \mid X_V)$ is **two-point** on each
fibre — given the visible block, the masked block decides the mode — so its
second moment is exactly as computable as its mean, and
$\mathbb{E}[S] = \mathfrak{D}_\mu$ exactly. It reports
$n^\star = (\mathrm{sd}[S]/\mathbb{E}[S])^2$, the examples needed for a
one-sigma detection, and the raw per-example loss $H(X_K \mid X_V)$ with its
variance decomposed.

### What comes out

- **E1** — exponential decay in the visible size at the predicted Chernoff rate
  (0.27 % at $N = 1023$) while total variation stays flat; the null control
  shows no decay.
- **E2** — curvature $\propto \pi$, recovery radius $\propto \pi^{1/2}$ and
  $\varepsilon^{1/2}$; measured exponents $+1.0000$, $-0.4955$, $+0.4968$.
- **E3** — the required shape is present (91× suppression on the bilingual
  pair, 183× on code vs prose), but its functional form is a **power law**, not
  the exponential the theory assumes — and a strictly richer classifier moves
  the level, not the exponent, so it is a property of the corpus rather than of
  the estimator.
- **E4** — the cost is exactly $1/\pi$: at the mass an untruncated schedule
  provides, detection needs $\sim 1.6 \times 10^4$ examples; with no
  low-visibility mass at all, $\sim 10^{116}$.

---

## Install

```bash
pip install -r requirements.txt
```

E3 additionally needs `scikit-learn`, `datasets`, `transformers` and network
access. It can be run fully offline against a synthetic fixture
(`--corpora markov`) if you only want to exercise the pipeline.

---

## Quickstart

```bash
python -m pytest tests -q          # 177 tests, ~10 s  -- run before any figure
python scripts/run_all.py          # E1 + E2 + E4 exact enumeration, ~25 s
python scripts/audit_numerics.py   # mpmath cross-check at 50 digits, ~3 s
python scripts/make_figures.py     # figure M, figure M_E3, table M, A1-A9
```

Artifacts land in `results/` as Parquet; figures in `figures/` as PDF, both
composed (`figure_M.pdf`) and panel by panel (`figure_M/panel_a.pdf`, …);
Table M in `figures/table_M.csv` and `figures/table_M.tex`.

To reproduce everything as committed, including the real-corpus stage:

```bash
python scripts/run_all.py
python scripts/audit_numerics.py
python scripts/run_all.py --skip-e1 --skip-e2 --skip-e4 --e3 \
    --corpora code_prose bilingual --tokenizer hf:gpt2 \
    --n-docs 16000 --n-windows 15000 --bootstrap 1000
python scripts/make_figures.py
```

Without network access, skip the E3 line. Figures degrade gracefully: A7, A8
and `figure_M_E3` print a warning and are skipped; everything else is
unaffected.

---

## Data for E3

E1, E2 and E4 need no data — they enumerate their laws. E3 streams two labelled
corpora from HuggingFace; **nothing needs downloading by hand**, as
`run_all.py --e3` streams, tokenises, packs and caches on first use.

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

Offline smoke test of the whole E3 pipeline:

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
| `--skip-e1`, `--skip-e2`, `--skip-e4` | off | skip a stage |
| `--results-dir DIR` | `results` | where to write the tables |
| `--e3` | off | run E3 (needs network) |
| `--corpora ...` | `code_prose bilingual` | any of `code_prose`, `bilingual`, `markov` |
| `--n-docs`, `--n-windows` | `20000`, `20000` | E3 sample sizes; cost scales with `n_windows` × `\|V\|` |
| `--bootstrap` | `1000` | bootstrap resamples, over documents |
| `--calibration` | `isotonic` | `isotonic` or `platt` |

`make_figures.py` takes `--only M A9 ...` to rebuild a subset, `--png` to emit
PNG previews alongside the PDFs, and `--results-dir` / `--out-dir`.

---

## Outputs

Fourteen Parquet tables in `results/` (CSV fallback), each stamped with
`config_hash`, `git_hash`, `kl_impl`, library versions and platform. Every
figure is a **pure function of those tables** — nothing is recomputed at plot
time, so any figure traces back to the run that produced it.

Eleven composed figures and 35 standalone panels in `figures/`, PDF only:

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
  e1.py         E1 driver
  e2.py         E2 driver
  e4.py         E4: per-example moments, detection cost, raw-loss cost  [exact]
  realdata.py   E3: corpora, tokenizer, both estimator arms, calibration, bootstrap
  tableio.py    Parquet/CSV writing with provenance stamping
  style.py      colour and mark decisions for every figure
  figures.py    figure M, table M, A1-A9
scripts/
  run_all.py        E1 + E2 + E4 (+ E3)
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
| A7 / A8 say "not run", or `figure_M_E3` is missing | run `scripts/audit_numerics.py` / `run_all.py --e3` |
| `parquet unavailable … wrote *.csv` | `pyarrow` missing; readers fall back to CSV automatically |
| `source exhausted after only N/M documents` | the stream ran out; lower `--n-docs` or `--doc-tokens` |
| `datasets` refuses `trust_remote_code` | pin `datasets<3`, or edit `CORPORA` in [src/mpi/realdata.py](src/mpi/realdata.py) |
| E3 too slow / out of memory at large `\|V\|` | lower `--n-windows` |
| want to re-download the corpora | delete `.cache/e3/` |
| `N must be odd` | the construction needs odd `N` so `S(x) ≠ 0` |
