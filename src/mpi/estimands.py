"""Exact estimands: D, Delta, U, F, rho, and boosted-schedule quantities.

All returns are natural logarithms unless the name says otherwise, because the
interesting range spans hundreds of decades.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .config import BRANCH_T_DEFAULT, KL_IMPL_DEFAULT, LawSpec, band_constants, mode_weight_interval
from .core import (
    VisibleLaw,
    binary_kl,
    log_binary_kl,
    log_kl_from_logit,
    softplus,
    visible_law,
)

__all__ = [
    "delta_logit",
    "log_D",
    "log_D_grid",
    "log_U",
    "log_flows",
    "log_Delta",
    "rho_grid",
    "CellSummary",
    "summarise_cell",
    "log_mix",
    "boosted_log_D",
    "boosted_log_U",
    "curvature_exact",
]


def delta_logit(lam: float, w: float) -> float:
    """``delta = logit(lambda) - logit(w)``, constant in ``r``."""
    return math.log(lam / (1.0 - lam)) - math.log(w / (1.0 - w))


# --------------------------------------------------------------------------
# Per-cell estimands
# --------------------------------------------------------------------------


def log_D(
    vl: VisibleLaw,
    lam: float,
    T: float = BRANCH_T_DEFAULT,
    impl: str = KL_IMPL_DEFAULT,
) -> float:
    r"""``log D_{N,m}(lambda)`` for the mode-reweighting witness ``q_lambda``.

    Implements ``log D = logsumexp_r [ log h(r) + log kl_r ]`` where
    ``kl_r = kl(beta_V(r) || beta_{V,lambda}(r))`` -- Prop. `info-decomp`,
    Eq. (decomp-binary).
    """
    d = delta_logit(lam, vl.w)
    log_kl = log_kl_from_logit(vl.logit_b, d, T, impl)
    return float(logsumexp(vl.log_h + log_kl))


def log_D_grid(
    vl: VisibleLaw,
    lambdas: np.ndarray,
    T: float = BRANCH_T_DEFAULT,
    impl: str = KL_IMPL_DEFAULT,
) -> np.ndarray:
    """``log D_{N,m}`` over a whole lambda grid, reusing the single ``ell`` array."""
    return np.array([log_D(vl, float(lam), T, impl) for lam in lambdas])


def log_U(vl: VisibleLaw) -> float:
    r"""``log mmse_p(Z | X_V)`` = ``logsumexp_r [log h + log b + log(1-b)]``."""
    return float(logsumexp(vl.log_h + vl.log_b + vl.log_1mb))


def log_flows(vl: VisibleLaw) -> tuple[float, float]:
    r"""``(log F^+, log F^-)`` -- the one-step cross-mode flows of Rem. `mode-pinning`.

    ``F^tau = E_{X_V ~ p^tau_V}[ p(X_{-tau} | X_V) ]`` is exactly the quantity
    bounded by Lem. `cross-mode-transition`(i).
    """
    lf_plus = float(logsumexp(vl.log_h_plus + vl.log_1mb))
    lf_minus = float(logsumexp(vl.log_h_minus + vl.log_b))
    return lf_plus, lf_minus


def log_Delta(
    vl: VisibleLaw,
    T: float = BRANCH_T_DEFAULT,
    impl: str = KL_IMPL_DEFAULT,
) -> tuple[float, float]:
    r"""``(log Delta_{N,m}, argmax lambda)`` via the endpoint shortcut.

    ``f(delta) = kl(b || b_delta)`` has ``f'(delta) = b_delta - b``, so
    ``D_{N,m}`` is decreasing on ``(0, w]`` and increasing on ``[w, 1)``.  The
    supremum over ``I`` is therefore attained at an endpoint -- two
    evaluations, no grid (spec section 2.5).
    """
    a, b = mode_weight_interval(vl.w)
    d_lo = log_D(vl, a, T, impl)
    d_hi = log_D(vl, b, T, impl)
    return (d_hi, b) if d_hi >= d_lo else (d_lo, a)


def rho_grid(
    log_D_vals: np.ndarray,
    log_U_val: float,
    lambdas: np.ndarray,
    w: float,
    exclude_radius: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    r"""``rho_{N,m}(lambda) = D / (U * kl(w || lambda))`` off a neighbourhood of ``w``.

    Returns ``(lambdas_kept, rho)``.  Lem. `sensitivity` asserts
    ``c_I <= rho <= C_I`` for every cell -- test T7.
    """
    keep = np.abs(lambdas - w) >= exclude_radius
    lam_keep = lambdas[keep]
    log_kl_prior = np.array([log_binary_kl(w, float(l)) for l in lam_keep])
    return lam_keep, np.exp(log_D_vals[keep] - log_U_val - log_kl_prior)


# --------------------------------------------------------------------------
# Cell summary
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CellSummary:
    law: str
    param: float
    N: int
    w: float
    m: int
    log_Delta: float
    lambda_star: float
    log_U: float
    log_Fplus: float
    log_Fminus: float


def summarise_cell(
    spec: LawSpec,
    N: int,
    m: int,
    w: float,
    T: float = BRANCH_T_DEFAULT,
    impl: str = KL_IMPL_DEFAULT,
) -> tuple[CellSummary, VisibleLaw]:
    vl = visible_law(spec, N, m, w)
    ld, lam_star = log_Delta(vl, T, impl)
    lfp, lfm = log_flows(vl)
    return (
        CellSummary(
            law=spec.law, param=spec.param, N=N, w=w, m=m,
            log_Delta=ld, lambda_star=lam_star,
            log_U=log_U(vl), log_Fplus=lfp, log_Fminus=lfm,
        ),
        vl,
    )


# --------------------------------------------------------------------------
# Boosted schedules  (spec section 2.5, "Boosted schedule linearity")
# --------------------------------------------------------------------------


def log_mix(log_a: float | np.ndarray, log_b: float | np.ndarray,
            pi: float) -> float | np.ndarray:
    """``log[(1-pi) * exp(log_a) + pi * exp(log_b)]``, exact at ``pi in {0, 1}``."""
    if pi <= 0.0:
        return log_a
    if pi >= 1.0:
        return log_b
    return np.logaddexp(math.log1p(-pi) + np.asarray(log_a),
                        math.log(pi) + np.asarray(log_b))


def boosted_log_D(log_D_base: float | np.ndarray, log_D_boost: float | np.ndarray,
                  pi: float) -> float | np.ndarray:
    r"""``D_mu`` for ``mu = (1-pi) delta_m + pi delta_s`` -- exactly linear."""
    return log_mix(log_D_base, log_D_boost, pi)


def boosted_log_U(log_U_base: float, log_U_boost: float, pi: float) -> float:
    r"""``E_{K~mu}[mmse]`` for the same two-point schedule."""
    return float(log_mix(log_U_base, log_U_boost, pi))


def curvature_exact(log_U_eff: float, w: float) -> float:
    r"""``kappa(pi, s) = d^2 D / d lambda^2 |_{lambda=w} = U_eff / (w(1-w))^2``.

    Exact, parameter-free (spec section 2.5, "Exact curvature").
    """
    return math.exp(log_U_eff - 2.0 * math.log(w * (1.0 - w)))
