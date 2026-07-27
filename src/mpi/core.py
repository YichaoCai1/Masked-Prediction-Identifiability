"""Exact log-domain machinery.

Nothing here samples, optimises or approximates: every quantity is an exact
enumeration over the visible magnetisation ``r``, carried out in the log
domain so that values down to ``log D ~ -1e308`` remain representable.  That
is what makes the exponential decay measurable over dozens of decades
(spec section 2.3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.special import logsumexp

from .config import BRANCH_T_DEFAULT, KL_IMPL_DEFAULT, LawSpec
from .laws import WeightFn, make_weight_fn

__all__ = [
    "softplus",
    "log_g",
    "log_kl_branched",
    "log_kl_stable",
    "log_kl_from_logit",
    "binary_kl",
    "log_binary_kl",
    "log_h_plus",
    "log_h_mode",
    "VisibleLaw",
    "visible_law",
    "clear_cache",
]


# --------------------------------------------------------------------------
# Scalar helpers
# --------------------------------------------------------------------------


def softplus(t: np.ndarray | float) -> np.ndarray:
    """``log(1 + e^t)``, overflow-free and exact at ``t = +-inf``."""
    t = np.asarray(t, dtype=float)
    with np.errstate(over="ignore"):
        return np.maximum(t, 0.0) + np.log1p(np.exp(-np.abs(t)))


def _g(d: float) -> float:
    """``g(d) = e^d - 1 - d >= 0``, accurate for small ``|d|``."""
    if abs(d) < 1e-2:
        # d^2/2 * (1 + d/3 + d^2/12 + d^3/60 + d^4/360); tail < 1e-14 relative.
        return 0.5 * d * d * (
            1.0 + d * (1.0 / 3.0 + d * (1.0 / 12.0 + d * (1.0 / 60.0 + d / 360.0)))
        )
    return math.expm1(d) - d


def log_g(d: float) -> float:
    """``log g(d)``; ``-inf`` at ``d = 0``."""
    if d == 0.0:
        return -math.inf
    return math.log(_g(d))


def binary_kl(w: float, lam: float) -> float:
    """``kl(w || lambda)`` for scalars in ``(0, 1)``; exact closed form."""
    if w == lam:
        return 0.0
    return w * math.log(w / lam) + (1.0 - w) * math.log((1.0 - w) / (1.0 - lam))


def log_binary_kl(w: float, lam: float) -> float:
    kl = binary_kl(w, lam)
    return math.log(kl) if kl > 0.0 else -math.inf


# --------------------------------------------------------------------------
# Stable binary KL from logits  (spec section 2.3)
# --------------------------------------------------------------------------


def log_kl_branched(x: np.ndarray, delta: float, T: float = BRANCH_T_DEFAULT) -> np.ndarray:
    r"""``log kl(sigma(x) || sigma(x + delta))``, three-branch scheme.

    This is the scheme prescribed by spec section 2.3 and is the default used
    for every reported number; the branch threshold ``T`` is one of the three
    knobs varied to produce the *numerical CI* (spec section 3.1).

    * ``|x| <= T``  : ``kl = sigma(x)(x-y) + softplus(y) - softplus(x)``
    * ``x >  T``    : ``log kl = -x + log g(-delta)``
    * ``x < -T``    : ``log kl =  x + log g( delta)``
    """
    x = np.asarray(x, dtype=float)
    if delta == 0.0:
        return np.full(x.shape, -np.inf)

    out = np.empty(x.shape, dtype=float)

    mid = np.abs(x) <= T
    hi = x > T
    lo = x < -T

    if np.any(mid):
        xm = x[mid]
        y = xm + delta
        kl = -delta / (1.0 + np.exp(-xm)) + softplus(y) - softplus(xm)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[mid] = np.where(kl > 0.0, np.log(np.maximum(kl, 0.0)), -np.inf)

    if np.any(hi):
        out[hi] = -x[hi] + log_g(-delta)
    if np.any(lo):
        out[lo] = x[lo] + log_g(delta)

    # x = +-inf are boundary fibers: beta in {0,1}, the divergence vanishes.
    out[~np.isfinite(x)] = -np.inf
    return out


def log_kl_stable(x: np.ndarray, delta: float, T: float = BRANCH_T_DEFAULT) -> np.ndarray:
    r"""Threshold-free reference implementation of the same quantity.

    Uses the algebraically equivalent forms

    ``kl = -b*delta + log1p(b*expm1(delta))``            (``x <= 0``, ``b = sigma(x)``)
    ``kl =  e*delta + log1p(e*expm1(-delta))``           (``x >  0``, ``e = sigma(-x)``)

    Both are well conditioned at their own tail, so no branch threshold is
    needed.  Kept as an independent cross-check of
    :func:`log_kl_branched` (test T3b); ``T`` is accepted and ignored so the
    two are drop-in interchangeable.
    """
    del T
    x = np.asarray(x, dtype=float)
    if delta == 0.0:
        return np.full(x.shape, -np.inf)

    out = np.empty(x.shape, dtype=float)
    neg = x <= 0.0
    pos = ~neg

    with np.errstate(over="ignore"):
        if np.any(neg):
            b = np.exp(x[neg]) / (1.0 + np.exp(x[neg]))
            kl = -b * delta + np.log1p(b * math.expm1(delta))
            with np.errstate(divide="ignore"):
                out[neg] = np.where(kl > 0.0, np.log(np.maximum(kl, 0.0)), -np.inf)
        if np.any(pos):
            e = np.exp(-x[pos]) / (1.0 + np.exp(-x[pos]))
            kl = e * delta + np.log1p(e * math.expm1(-delta))
            with np.errstate(divide="ignore"):
                out[pos] = np.where(kl > 0.0, np.log(np.maximum(kl, 0.0)), -np.inf)

    out[~np.isfinite(x)] = -np.inf
    return out


#: Registry so drivers can switch implementations from the command line.
KL_IMPLS = {"branched": log_kl_branched, "stable": log_kl_stable}


def log_kl_from_logit(
    x: np.ndarray, delta: float, T: float = BRANCH_T_DEFAULT, impl: str = KL_IMPL_DEFAULT
) -> np.ndarray:
    return KL_IMPLS[impl](x, delta, T)


# --------------------------------------------------------------------------
# Visible-magnetisation laws  (spec section 2.1)
# --------------------------------------------------------------------------


def _log_h_tau_uncached(weight_fn: WeightFn, N: int, m: int, tau: int) -> np.ndarray:
    """Normalised ``log h^tau_{N,m}(r)``, indexed by ``a = (r+m)/2 in {0..m}``.

    ``h^tau`` is the visible-count marginal of ``p^tau = nu_tau(. | X_tau)``.
    Because ``N`` is odd, ``S = 2(a+b) - N`` is odd and never zero, so the
    restriction to ``X_+ = {S > 0}`` is ``a + b >= (N+1)/2`` and the
    restriction to ``X_- = {S < 0}`` is ``a + b <= (N-1)/2``.
    """
    if N % 2 == 0:
        raise ValueError("N must be odd")
    if not 0 <= m <= N:
        raise ValueError("m must lie in [0, N]")

    A = np.arange(m + 1, dtype=float)[:, None]        # (m+1, 1)
    B = np.arange(N - m + 1, dtype=float)[None, :]    # (1, N-m+1)

    LW = weight_fn(N, m, A, B)
    keep = (A + B >= (N + 1) // 2) if tau > 0 else (A + B <= (N - 1) // 2)
    LW = np.where(keep, LW, -np.inf)

    log_h = logsumexp(LW, axis=1)
    return log_h - logsumexp(log_h)


@dataclass(frozen=True)
class VisibleLaw:
    """Everything about a ``(law, N, m, w)`` cell that downstream code needs.

    Arrays are indexed by ``a in {0..m}`` i.e. by ``r = 2a - m``.
    """

    N: int
    m: int
    w: float
    r: np.ndarray            # visible magnetisation grid
    log_h_plus: np.ndarray   # log h^+(r)
    log_h_minus: np.ndarray  # log h^-(r) = log h^+(-r)
    ell: np.ndarray          # log h^+(r) - log h^-(r)
    log_h: np.ndarray        # pooled log h(r) under p = w p^+ + (1-w) p^-
    logit_b: np.ndarray      # logit beta_V(r) = logit(w) + ell(r)

    @property
    def log_b(self) -> np.ndarray:
        """``log beta_V(r)``, stable at both tails."""
        return -softplus(-self.logit_b)

    @property
    def log_1mb(self) -> np.ndarray:
        """``log(1 - beta_V(r))``, stable at both tails."""
        return -softplus(self.logit_b)


def log_h_mode(spec: LawSpec, N: int, m: int, tau: int = +1) -> np.ndarray:
    r"""Cached ``log h^tau_{N,m}``.  Independent of ``w``, hence shared across it.

    ``tau = -1`` enumerates the negative mode from scratch rather than by
    reflection; the main path never needs it, but test T1 compares the two.
    """
    return _log_h_mode_cached(spec.law, float(spec.param), int(N), int(m), int(tau)).copy()


def log_h_plus(spec: LawSpec, N: int, m: int) -> np.ndarray:
    return log_h_mode(spec, N, m, +1)


@lru_cache(maxsize=1024)
def _log_h_mode_cached(law: str, param: float, N: int, m: int, tau: int) -> np.ndarray:
    # nu_- is nu_+ with every coordinate flipped; for the product law that is
    # theta -> -theta, and the Curie-Weiss weight is already flip-invariant.
    param_tau = -param if (law == "product" and tau < 0) else param
    weight_fn = make_weight_fn(LawSpec(law, param_tau, True, ""))
    arr = _log_h_tau_uncached(weight_fn, N, m, tau)
    arr.flags.writeable = False
    return arr


def clear_cache() -> None:
    _log_h_mode_cached.cache_clear()


def visible_law(spec: LawSpec, N: int, m: int, w: float) -> VisibleLaw:
    """Assemble the ``(law, N, m, w)`` cell.

    Uses the spin-flip symmetry ``h^-(r) = h^+(-r)`` (test T1) so that only
    ``h^+`` is enumerated.
    """
    lhp = log_h_plus(spec, N, m)
    lhm = lhp[::-1].copy()                     # h^-(r) = h^+(-r)

    with np.errstate(invalid="ignore"):
        ell = lhp - lhm
    # (-inf) - (-inf) is an impossible visible pattern under both modes; it
    # cannot occur because at least one mode always supports every r that the
    # pooled law charges.  Guard anyway so a bug surfaces as NaN, not silence.
    ell = np.where(np.isnan(ell), 0.0, ell)

    log_h = np.logaddexp(math.log(w) + lhp, math.log1p(-w) + lhm)
    logit_b = math.log(w / (1.0 - w)) + ell

    r = 2.0 * np.arange(m + 1) - m
    return VisibleLaw(
        N=N, m=m, w=w, r=r,
        log_h_plus=lhp, log_h_minus=lhm, ell=ell, log_h=log_h, logit_b=logit_b,
    )
