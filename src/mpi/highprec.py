"""mpmath cross-checks (test T3 and the precision leg of the numerical CI).

Everything here recomputes a cell from scratch at 50 significant digits with no
float64 intermediates: log-factorials, the restriction to ``X_+``, the pooled
law, the posterior logit, and the binary divergence.  mpmath has an unbounded
exponent range, so no branch scheme is needed -- which is precisely what makes
it a useful check of the float64 branch scheme.

Cost is ``O(m(N-m))`` arbitrary-precision operations per cell, so this is spot
checks only, never the main path.
"""

from __future__ import annotations

import math
from functools import lru_cache

import mpmath as mp

from .config import LawSpec

__all__ = ["log_D_mp", "log_U_mp", "log_h_plus_mp"]


@lru_cache(maxsize=8)
def _log_factorials(N: int, dps: int) -> tuple:
    with mp.workdps(dps):
        out = [mp.mpf(0)]
        for k in range(1, N + 1):
            out.append(out[-1] + mp.log(k))
        return tuple(out)


def _log_weight(spec: LawSpec, N: int, m: int, a: int, b: int, lf: tuple):
    lb = lf[m] - lf[a] - lf[m - a] + lf[N - m] - lf[b] - lf[N - m - b]
    if spec.law == "product":
        th = mp.mpf(spec.param)
        n_plus = a + b
        return (
            lb
            + n_plus * mp.log((1 + th) / 2)
            + (N - n_plus) * mp.log((1 - th) / 2)
        )
    if spec.law == "cw":
        beta = mp.mpf(spec.param)
        S = mp.mpf(2 * (a + b) - N)
        return lb + beta * S**2 / (2 * N)
    raise ValueError(f"unknown law {spec.law!r}")


def log_h_plus_mp(spec: LawSpec, N: int, m: int, dps: int = 50) -> list:
    """``log h^+_{N,m}(r)`` at ``dps`` digits, indexed by ``a``."""
    with mp.workdps(dps):
        lf = _log_factorials(N, dps)
        half = (N + 1) // 2
        rows = []
        for a in range(m + 1):
            b_min = max(0, half - a)
            if b_min > N - m:
                rows.append(mp.mpf(0))
                continue
            acc = mp.mpf(0)
            for b in range(b_min, N - m + 1):
                acc += mp.e ** _log_weight(spec, N, m, a, b, lf)
            rows.append(acc)
        total = mp.fsum(rows)
        return [
            (mp.log(v) - mp.log(total)) if v > 0 else mp.mpf("-inf") for v in rows
        ]


def _cell_mp(spec: LawSpec, N: int, m: int, w: float, dps: int):
    lhp = log_h_plus_mp(spec, N, m, dps)
    lhm = list(reversed(lhp))
    lw, l1mw = mp.log(mp.mpf(w)), mp.log(1 - mp.mpf(w))
    log_h = []
    logit_b = []
    for hp, hm in zip(lhp, lhm):
        # log-sum-exp of two terms, robust to -inf
        t1, t2 = lw + hp, l1mw + hm
        big = t1 if t1 > t2 else t2
        log_h.append(big + mp.log(mp.e ** (t1 - big) + mp.e ** (t2 - big)))
        if hp == mp.mpf("-inf"):
            logit_b.append(mp.mpf("-inf"))
        elif hm == mp.mpf("-inf"):
            logit_b.append(mp.mpf("+inf"))
        else:
            logit_b.append(mp.log(mp.mpf(w) / (1 - mp.mpf(w))) + hp - hm)
    return log_h, logit_b


def log_D_mp(spec: LawSpec, N: int, m: int, w: float, lam: float, dps: int = 50) -> float:
    r"""``log D_{N,m}(lambda)`` at ``dps`` digits."""
    with mp.workdps(dps):
        log_h, logit_b = _cell_mp(spec, N, m, w, dps)
        d = mp.log(mp.mpf(lam) / (1 - mp.mpf(lam))) - mp.log(mp.mpf(w) / (1 - mp.mpf(w)))
        acc = mp.mpf(0)
        for lh, x in zip(log_h, logit_b):
            if not mp.isfinite(x):
                continue
            b = 1 / (1 + mp.e ** (-x))
            y = x + d
            b2 = 1 / (1 + mp.e ** (-y))
            if b <= 0 or b >= 1:
                continue
            kl = b * mp.log(b / b2) + (1 - b) * mp.log((1 - b) / (1 - b2))
            if kl > 0:
                acc += mp.e**lh * kl
        return float(mp.log(acc)) if acc > 0 else float("-inf")


def log_U_mp(spec: LawSpec, N: int, m: int, w: float, dps: int = 50) -> float:
    r"""``log mmse_p(Z | X_V)`` at ``dps`` digits."""
    with mp.workdps(dps):
        log_h, logit_b = _cell_mp(spec, N, m, w, dps)
        acc = mp.mpf(0)
        for lh, x in zip(log_h, logit_b):
            if not mp.isfinite(x):
                continue
            b = 1 / (1 + mp.e ** (-x))
            acc += mp.e**lh * b * (1 - b)
        return float(mp.log(acc)) if acc > 0 else float("-inf")
