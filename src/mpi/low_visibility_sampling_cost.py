"""Exact sampling cost of adding low-visibility mass.

The mode-blindness and intervention calculations are population statements:
they evaluate ``D_mu`` exactly, as though an infinite sample were available.
This stage asks the finite-sample question that
Prop. `full-mask` raises but does not answer -- *how many masked examples does
it take to see* a discrepancy that those calculations certify is nonzero? This is the
quantitative content of Sec. 5.2, "The cost of the full mask".

Nothing here samples either.  For the mode-reweighting witness the per-example
excess loss is a **two-point** random variable on every fibre, so its second
moment is exactly as computable as its mean.  With

    S = log p_K(X_K | X_V) - log q_K(X_K | X_V),     K ~ mu,   X ~ p

the masked block determines the mode (``X_+`` and ``X_-`` partition the cube
and ``N`` is odd), and ``p`` and ``q_lambda`` share their mode-conditionals, so
conditionally on ``(K, X_V)``

    S = log(beta / beta_lambda)              with probability beta
      = log((1-beta) / (1-beta_lambda))      with probability 1 - beta.

Its conditional mean is the fibrewise ``kl(beta || beta_lambda)`` of
Prop. `info-decomp`, so ``E[S] = D_mu`` exactly; its second moment is one more
``logsumexp`` over the same ``r`` grid as the mode-blindness calculation.

Two costs are reported, because Sec. 5.2 names two and they behave differently:

*Detection cost.*  ``cv = sd[S] / E[S]`` and ``n_star = cv^2``, the number of
masked examples at which the sample mean of ``S`` first clears one standard
error (a ``z``-sigma detection needs ``z^2 n_star``).  For a two-point schedule
``mu = (1-pi) delta_m + pi delta_s`` the signal grows like ``pi`` while the
noise grows like ``sqrt(pi)``, so ``n_star`` scales as ``1/pi`` -- the
Monte-Carlo counterpart of the ``pi_0^{-1/2}`` in Prop. `full-mask`.

*Raw-loss cost.*  ``H(X_K | X_V)``, the per-example loss that actually enters
the training objective, together with its exact variance.  The full mask sits
at the joint entropy by construction (``H(X_K | X_V) = H(X)`` at ``m = 0``).
Its schedule-induced variance contribution ``pi (1-pi) (H_s - H_base)^2`` is
the "high-variance term" of Sec. 5.2.

The sampling-cost stage was added after the exact population stages were
frozen, and it touches nothing they own: it
defines its own grids here rather than extending
:class:`~mpi.config.RunConfig`, so re-running the earlier stages reproduces their existing
artifacts byte for byte.

Produces ``sampling``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp

from .config import LawSpec, RunConfig, mode_weight_interval
from .core import VisibleLaw, softplus, visible_law
from .estimands import delta_logit, log_D, log_mix, log_U
from .laws import make_weight_fn
from .tableio import write_table

__all__ = [
    "LAMBDA_FRACTIONS",
    "log_second_moment",
    "SamplingCost",
    "sampling_cost",
    "raw_loss_moments",
    "run_low_visibility_sampling_cost",
]

#: Mode-weight errors probed, as fractions of the half-width of ``I``.  ``1.0``
#: is the interval endpoint -- the largest error identifiability is asked to
#: resolve, and the one ``Delta_{N,m}`` itself is defined by -- and the rest
#: trace the ``(lambda - w)^{-2}`` blow-up as the error shrinks.
LAMBDA_FRACTIONS: tuple[float, ...] = (1.0, 0.6, 0.4, 0.2, 0.1, 0.05, 0.02)


def _base_sizes(N: int) -> tuple[int, ...]:
    """Use the intervention stage's base sizes so the two tables line up."""
    return (N // 4, N // 2)


def _log_binom(n: int, k: np.ndarray) -> np.ndarray:
    k = np.asarray(k, dtype=float)
    return gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)


def _safe_exp(x: float) -> float:
    """``exp`` that saturates instead of raising; ``n_star`` legitimately
    reaches ``e^266`` at ``pi = 0``, and larger cells must not abort a run."""
    try:
        return math.exp(x)
    except OverflowError:
        return math.inf


# --------------------------------------------------------------------------
# Detection cost: the second moment of the per-example excess loss
# --------------------------------------------------------------------------


def log_second_moment(vl: VisibleLaw, lam: float) -> float:
    r"""``log E[S^2]`` at a single visible size, exactly.

    ``S`` is two-point on each fibre, so

        E[S^2] = sum_r h(r) [ beta (log beta/beta_lam)^2
                            + (1-beta) (log (1-beta)/(1-beta_lam))^2 ].

    Both arms are non-negative, so the whole sum goes through one
    ``logsumexp`` with no sign bookkeeping -- unlike the mean, where the arms
    cancel down to ``kl``.  Fibres with ``beta in {0, 1}`` carry ``S == 0`` and
    drop out, exactly as they do in :func:`~mpi.estimands.log_D`.
    """
    d = delta_logit(lam, vl.w)
    if d == 0.0:
        return -math.inf

    finite = np.isfinite(vl.logit_b)
    x = vl.logit_b[finite]
    y = x + d

    # |up| and |dn| are both bounded by |d| (mean value theorem on softplus),
    # so neither arm can overflow; underflow to 0 is the correct limit.
    up = softplus(-y) - softplus(-x)          # log(beta / beta_lambda)
    dn = softplus(y) - softplus(x)            # log((1-beta) / (1-beta_lambda))

    with np.errstate(divide="ignore"):
        terms = np.logaddexp(
            -softplus(-x) + 2.0 * np.log(np.abs(up)),
            -softplus(x) + 2.0 * np.log(np.abs(dn)),
        )
    return float(logsumexp(vl.log_h[finite] + terms))


def _log_diff(la: float, lb: float) -> float:
    """``log(e^la - e^lb)`` for ``la >= lb``; ``-inf`` when they coincide."""
    if not math.isfinite(lb):
        return la
    if la <= lb:
        return -math.inf
    return la + math.log1p(-math.exp(lb - la))


@dataclass(frozen=True)
class SamplingCost:
    """Detection cost of one schedule against one witness."""

    log_mean: float      # log E[S] = log D_mu
    log_second: float    # log E[S^2]
    log_var: float
    cv: float            # sd / mean
    log_n_star: float    # log cv^2

    @property
    def n_star(self) -> float:
        return _safe_exp(self.log_n_star)


def sampling_cost(log_mean: float, log_second: float) -> SamplingCost:
    """Assemble the cost summary from the two exact log-moments.

    ``E[S]^2 <= E[S^2]`` by Jensen, so the difference is taken in the stable
    order and never underflows to a negative variance.
    """
    log_var = _log_diff(log_second, 2.0 * log_mean)
    log_n = log_var - 2.0 * log_mean
    return SamplingCost(log_mean, log_second, log_var,
                        _safe_exp(0.5 * log_n), log_n)


# --------------------------------------------------------------------------
# Raw-loss cost: the per-example training loss and its variance
# --------------------------------------------------------------------------


def _joint_count_law(spec: LawSpec, N: int, m: int,
                     w: float) -> tuple[np.ndarray, np.ndarray]:
    """Pooled pmf of the count pair ``(a, b)``, and ``log p(x)`` per configuration.

    ``N`` odd means ``S = 2(a+b) - N`` is never zero, so each *count* cell lies
    wholly in ``X_+`` or wholly in ``X_-`` and exactly one mode charges it.
    The negative mode is obtained by the spin flip ``(a, b) -> (m-a, N-m-b)``
    rather than re-enumerated, the same symmetry :func:`~mpi.core.visible_law`
    uses.
    """
    if N % 2 == 0:
        raise ValueError("N must be odd")
    weight_fn = make_weight_fn(spec)
    A = np.arange(m + 1, dtype=float)[:, None]
    B = np.arange(N - m + 1, dtype=float)[None, :]

    LW = weight_fn(N, m, A, B)
    plus = (A + B) >= (N + 1) // 2

    LW_plus = np.where(plus, LW, -np.inf)
    log_P_plus = LW_plus - logsumexp(LW_plus)
    log_P_minus = log_P_plus[::-1, ::-1]

    log_P = np.logaddexp(math.log(w) + log_P_plus,
                         math.log1p(-w) + log_P_minus)
    log_multiplicity = _log_binom(m, A) + _log_binom(N - m, B)
    return log_P, log_P - log_multiplicity


def raw_loss_moments(spec: LawSpec, N: int, m: int,
                     w: float) -> tuple[float, float]:
    """``(E[L], Var[L])`` in nats for ``L = -log p_K(X_K | X_V)``.

    This is the loss a learner actually pays on one example at visible size
    ``m``, not the excess loss the identifiability statement tracks.  At
    ``m = 0`` it reduces to the joint entropy ``H(X)``, which is the sense in
    which the full-mask channel "prices in the entire joint law".
    """
    log_P, log_px = _joint_count_law(spec, N, m, w)
    vl = visible_law(spec, N, m, w)
    log_pv = vl.log_h - _log_binom(m, np.arange(m + 1))

    loss = -log_px + log_pv[:, None]
    P = np.exp(log_P)
    keep = np.isfinite(loss) & (P > 0.0)
    first = float(np.sum(P[keep] * loss[keep]))
    second = float(np.sum(P[keep] * loss[keep] ** 2))
    return first, max(second - first * first, 0.0)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def _cell_rows(spec: LawSpec, N: int, w: float, cfg: RunConfig) -> list[dict]:
    a_end, b_end = mode_weight_interval(w)
    lambdas = [w + f * (b_end - w) for f in LAMBDA_FRACTIONS]

    sizes = sorted(set(_base_sizes(N)) | set(cfg.s_boost_grid))
    comp: dict[int, dict] = {}
    for m in sizes:
        vl = visible_law(spec, N, int(m), w)
        raw_mean, raw_var = raw_loss_moments(spec, N, int(m), w)
        comp[int(m)] = {
            "logU": log_U(vl),
            "logD": {f: log_D(vl, lam, cfg.branch_T, cfg.kl_impl)
                     for f, lam in zip(LAMBDA_FRACTIONS, lambdas)},
            "logM2": {f: log_second_moment(vl, lam)
                      for f, lam in zip(LAMBDA_FRACTIONS, lambdas)},
            "raw_mean": raw_mean,
            "raw_var": raw_var,
        }

    # H(X) itself: the m = 0 fibre is the whole joint law, by definition.
    h_joint = comp[0]["raw_mean"] if 0 in comp else float("nan")

    rows: list[dict] = []
    for m_base in _base_sizes(N):
        cb = comp[int(m_base)]
        for s in cfg.s_boost_grid:
            cs = comp[int(s)]
            for pi in cfg.pi_grid:
                # raw-loss moments of the mixture, by the law of total variance
                raw_mean = (1.0 - pi) * cb["raw_mean"] + pi * cs["raw_mean"]
                raw_var_within = (1.0 - pi) * cb["raw_var"] + pi * cs["raw_var"]
                gap = cs["raw_mean"] - cb["raw_mean"]
                raw_var_between = pi * (1.0 - pi) * gap * gap

                for frac, lam in zip(LAMBDA_FRACTIONS, lambdas):
                    cost = sampling_cost(
                        float(log_mix(cb["logD"][frac], cs["logD"][frac], pi)),
                        float(log_mix(cb["logM2"][frac], cs["logM2"][frac], pi)),
                    )
                    rows.append(
                        {
                            "law": spec.law, "param": spec.param, "N": N, "w": w,
                            "m_base": int(m_base), "s": int(s), "pi": pi,
                            "lambda": lam, "lambda_frac": frac,
                            "lambda_err": abs(lam - w),
                            "logD": cost.log_mean,
                            "logM2": cost.log_second,
                            "logVar": cost.log_var,
                            "sd": _safe_exp(0.5 * cost.log_var),
                            "cv": cost.cv,
                            "n_star": cost.n_star,
                            "log10_n_star": cost.log_n_star / math.log(10.0),
                            "raw_mean": raw_mean,
                            "raw_sd": math.sqrt(raw_var_within + raw_var_between),
                            "raw_var_within": raw_var_within,
                            "raw_var_between": raw_var_between,
                            "raw_var_schedule_frac": (
                                raw_var_between / (raw_var_within + raw_var_between)
                                if raw_var_within + raw_var_between > 0 else 0.0
                            ),
                            "H_base": cb["raw_mean"],
                            "H_boost": cs["raw_mean"],
                            "H_joint": h_joint,
                            "H_boost_over_joint": (
                                cs["raw_mean"] / h_joint if h_joint else float("nan")
                            ),
                            "logU_base": cb["logU"],
                            "logU_boost": cs["logU"],
                        }
                    )
    return rows


def run_low_visibility_sampling_cost(
    cfg: RunConfig, results_dir: Path | None = None, verbose: bool = True
):
    """Compute the ``sampling`` table over the intervention grid."""
    t0 = time.time()
    rows: list[dict] = []
    for spec in cfg.law_specs():
        if not spec.pinned:
            continue  # the witness family is defined by the two-mode structure
        for w in cfg.w_grid:
            for N in cfg.N_grid:
                tic = time.time()
                rows.extend(_cell_rows(spec, N, w, cfg))
                if verbose:
                    print(f"  sampling cost {spec.key:>12s}  N={N:<5d} w={w:<4g}"
                          f"  [{time.time()-tic:5.1f}s]")

    sampling = pd.DataFrame(rows)
    write_table(sampling, "sampling", cfg, results_dir)
    if verbose:
        print(
            f"  low-visibility sampling cost done in {time.time()-t0:.1f}s "
            f"({len(sampling)} cells)"
        )
    return {"sampling": sampling}
