"""Fitting utilities: window OLS, curvature fit, recovery-radius bisection.

There is no sampling noise in the exact synthetic studies, so every
"uncertainty" reported here is a *numerical* CI: the spread of an estimate
under fit-window shifts,
branch-threshold changes, and float64-vs-mpmath recomputation
(spec section 3.1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "OLSFit",
    "ols",
    "best_window_ols",
    "window_shift_spread",
    "anchored_exponential_fit",
    "quadratic_curvature_fit",
    "curvature_step",
    "curvature_richardson",
    "RecoveryRadius",
    "recovery_radius",
]


# --------------------------------------------------------------------------
# Ordinary least squares
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class OLSFit:
    slope: float
    intercept: float
    r2: float
    n: int
    x_lo: float
    x_hi: float
    idx_lo: int = -1
    idx_hi: int = -1

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.intercept + self.slope * np.asarray(x, dtype=float)


def ols(x: Sequence[float], y: Sequence[float]) -> OLSFit:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2:
        raise ValueError("need at least two points")
    xbar, ybar = x.mean(), y.mean()
    sxx = float(((x - xbar) ** 2).sum())
    if sxx == 0.0:
        raise ValueError("degenerate design (all x equal)")
    slope = float(((x - xbar) * (y - ybar)).sum() / sxx)
    intercept = float(ybar - slope * xbar)
    resid = y - (intercept + slope * x)
    ss_tot = float(((y - ybar) ** 2).sum())
    ss_res = float((resid**2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return OLSFit(slope, intercept, r2, int(x.size), float(x.min()), float(x.max()))


def best_window_ols(
    x: Sequence[float],
    y: Sequence[float],
    min_points: int = 5,
    tie_tol: float = 1e-12,
) -> OLSFit | None:
    """OLS over the contiguous window of ``>= min_points`` maximising ``R^2``.

    ``x`` must be sorted.  Non-finite ``y`` entries are dropped first.  Exact
    ties in ``R^2`` (within ``tie_tol``) are broken toward the *longer* window,
    which matters because with zero sampling noise the shortest admissible
    window can otherwise win by an amount below machine precision.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    n = x.size
    if n < min_points:
        return None

    best: OLSFit | None = None
    for lo in range(0, n - min_points + 1):
        for hi in range(lo + min_points, n + 1):
            try:
                fit = ols(x[lo:hi], y[lo:hi])
            except ValueError:
                continue
            if not math.isfinite(fit.r2):
                continue
            if (
                best is None
                or fit.r2 > best.r2 + tie_tol
                or (abs(fit.r2 - best.r2) <= tie_tol and fit.n > best.n)
            ):
                best = OLSFit(
                    fit.slope, fit.intercept, fit.r2, fit.n,
                    float(x[lo]), float(x[hi - 1]), lo, hi - 1,
                )
    return best


def window_shift_spread(
    x: Sequence[float],
    y: Sequence[float],
    base: OLSFit,
    shifts: Sequence[int] = (-1, 0, 1),
) -> tuple[float, float, list[float]]:
    """Slopes obtained by shifting each window endpoint by ``shifts`` grid points.

    Returns ``(min_slope, max_slope, all_slopes)``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    n = x.size

    slopes: list[float] = []
    for dlo in shifts:
        for dhi in shifts:
            lo = base.idx_lo + dlo
            hi = base.idx_hi + dhi
            if lo < 0 or hi >= n or hi - lo + 1 < 3:
                continue
            slopes.append(ols(x[lo : hi + 1], y[lo : hi + 1]).slope)
    if not slopes:
        return base.slope, base.slope, [base.slope]
    return min(slopes), max(slopes), slopes


def anchored_exponential_fit(
    s: Sequence[int], log_u: Sequence[float], log_u0_anchor: float
) -> tuple[float, float]:
    r"""Fit ``log U_s = log u0 - L s`` with the intercept pinned to the exact anchor.

    Spec section 3.2 asks for the fit to use "the three interior points plus the
    exact ``s = 0`` anchor".  The ``s = 0`` value is exact
    (``U_{N,0} = w(1-w)``), so it enters as a *constraint*, not as a fourth
    noisy observation; the slope is then the least-squares solution through
    that anchor.  Returns ``(u0_hat, L_hat)``.
    """
    s = np.asarray([v for v in s], dtype=float)
    log_u = np.asarray(log_u, dtype=float)
    interior = s > 0
    ss = s[interior]
    dy = log_u[interior] - log_u0_anchor
    L_hat = float(-(ss * dy).sum() / (ss**2).sum())
    return math.exp(log_u0_anchor), L_hat


def quadratic_curvature_fit(
    lambdas: np.ndarray, D_vals: np.ndarray, w: float, half_width: float = 0.02
) -> float:
    r"""Local quadratic estimate of ``d^2 D / d lambda^2`` at ``lambda = w``.

    The spec-literal cross-check: a one-parameter least-squares solve of
    ``D = (kappa/2)(lambda - w)^2`` on the ``Lambda_I`` points within
    ``half_width`` of ``w`` (``D(w) = 0`` and ``D'(w) = 0``, so no lower-order
    terms appear).  Unbiased at ``w = 1/2``, where the odd Taylor terms vanish
    by symmetry; biased at asymmetric ``w``, which is why
    :func:`curvature_richardson` is what T4 asserts against.
    """
    lambdas = np.asarray(lambdas, dtype=float)
    D_vals = np.asarray(D_vals, dtype=float)
    sel = (np.abs(lambdas - w) <= half_width) & (lambdas != w) & np.isfinite(D_vals)
    if sel.sum() < 3:
        return float("nan")
    t = 0.5 * (lambdas[sel] - w) ** 2
    return float((t * D_vals[sel]).sum() / (t * t).sum())


def curvature_step(w: float, base: float = 0.02) -> float:
    """Step size for :func:`curvature_richardson`, scaled to the distance to 0/1.

    The Taylor series of ``D`` about ``w`` has radius ``min(w, 1-w)``, so a
    fixed ``0.02`` that is comfortable at ``w = 1/2`` is far too coarse at
    ``w = 0.9``.  Scaling keeps the truncation error uniform across ``w``.
    """
    return base * min(1.0, 4.0 * min(w, 1.0 - w))


def curvature_richardson(
    log_D_fn: Callable[[float], float], w: float, h: float | None = None
) -> float:
    r"""Richardson-extrapolated central estimate of ``d^2 D/d lambda^2`` at ``w``.

    Since ``D(w) = 0`` and ``D'(w) = 0``,
    ``[D(w+h) + D(w-h)] / h^2 = kappa + (c4/12) h^2 + O(h^4)``,
    so combining the estimate at ``h`` and at ``h/2`` cancels the leading
    truncation term and leaves ``O(h^4)``.  This is the estimator T4 asserts
    the exact identity against.
    """
    h = curvature_step(w) if h is None else h

    def kap(step: float) -> float:
        s = math.exp(log_D_fn(w + step)) + math.exp(log_D_fn(w - step))
        return s / (step * step)

    k_h, k_h2 = kap(h), kap(h / 2.0)
    return (4.0 * k_h2 - k_h) / 3.0


# --------------------------------------------------------------------------
# Recovery radius
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RecoveryRadius:
    eps: float
    radius: float
    censored: bool
    side: str = ""


def recovery_radius(
    log_D_fn: Callable[[float], float],
    w: float,
    interval: tuple[float, float],
    eps: float,
    tol: float = 1e-10,
) -> RecoveryRadius:
    r"""``r(eps) = max{ |w - lambda| : lambda in I, D(lambda) <= eps }``.

    ``D`` is decreasing on ``[a, w]`` and increasing on ``[w, b]``, so a plain
    bisection on each side is exact.  A side whose interval endpoint already
    satisfies ``D <= eps`` is *censored*: the true radius is cut off by ``I``,
    not by the tolerance, and such cells are excluded from scaling fits.
    """
    a, b = interval
    log_eps = math.log(eps)

    def solve(lo: float, hi: float, decreasing: bool) -> tuple[float, bool]:
        """Return (boundary lambda, censored) for one side; ``w`` is the
        ``D = 0`` end and (lo, hi) brackets the side with ``lo < hi``."""
        outer = lo if decreasing else hi
        if log_D_fn(outer) <= log_eps:
            return outer, True
        left, right = lo, hi
        for _ in range(200):
            mid = 0.5 * (left + right)
            inside = log_D_fn(mid) <= log_eps
            if decreasing:
                # D decreasing in lambda: inside means we may move left.
                if inside:
                    right = mid
                else:
                    left = mid
            else:
                if inside:
                    left = mid
                else:
                    right = mid
            if right - left < tol:
                break
        return (right if decreasing else left), False

    lam_left, cens_left = solve(a, w, decreasing=True)
    lam_right, cens_right = solve(w, b, decreasing=False)

    r_left, r_right = w - lam_left, lam_right - w
    if r_left >= r_right:
        return RecoveryRadius(eps, r_left, cens_left, "left")
    return RecoveryRadius(eps, r_right, cens_right, "right")
