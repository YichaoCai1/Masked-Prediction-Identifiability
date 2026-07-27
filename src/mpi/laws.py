"""Data laws.

The *only* line that differs between laws is the unnormalised log-weight of a
visible/hidden count pair ``(a, b)`` (spec section 2.1).  Everything downstream
is law-agnostic, so a law is fully described by a ``weight_fn``.

Conventions (fixed throughout the package)
------------------------------------------
``N``      odd sequence length, alphabet ``{-1, +1}``
``m``      number of visible coordinates, ``|V| = m``
``a``      number of ``+1`` among the visible coordinates, ``a in {0..m}``
``r``      visible magnetisation ``2a - m``
``b``      number of ``+1`` among the hidden coordinates, ``b in {0..N-m}``
``u``      hidden magnetisation ``2b - (N-m)``
``S``      total magnetisation ``r + u``; odd, hence never zero
``X_tau``  ``{x : tau * S(x) > 0}``

Both laws are exchangeable, so the visible marginal depends on ``x_V`` only
through ``r``.  Both are spin-flip symmetric, which gives
``h^-(r) = h^+(-r)`` exactly (test T1) and halves the compute.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.special import gammaln

from .config import LawSpec

#: A weight function maps ``(N, m, A, B)`` -- with ``A`` an ``(m+1, 1)`` column
#: of visible ``+1`` counts and ``B`` a ``(1, N-m+1)`` row of hidden ``+1``
#: counts -- to the ``(m+1, N-m+1)`` matrix of unnormalised log-weights.
WeightFn = Callable[[int, int, np.ndarray, np.ndarray], np.ndarray]


def _log_binom(n: int, k: np.ndarray) -> np.ndarray:
    """``log C(n, k)`` via ``gammaln``; ``k`` may be an array."""
    k = np.asarray(k, dtype=float)
    return gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)


def product_weight_fn(theta: float) -> WeightFn:
    r"""Law P: ``nu_+`` is the product law with ``nu_+(X_i = +1) = (1+theta)/2``.

    The mode law is ``p^+ = nu_+(. | X_+)``; the restriction to ``X_+`` is
    applied by :func:`mpi.core.log_h_mode`, not here.

    A negative ``theta`` yields ``nu_-`` (every coordinate flipped), which is
    how the opposite mode is enumerated from scratch for test T1.
    """
    if not -1.0 < theta < 1.0 or theta == 0.0:
        raise ValueError("theta must lie in (-1, 0) u (0, 1)")
    log_p = math.log((1.0 + theta) / 2.0)
    log_q = math.log((1.0 - theta) / 2.0)

    def weight_fn(N: int, m: int, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        n_plus = A + B                       # total number of +1 coordinates
        n_minus = N - n_plus
        return (
            _log_binom(m, A)
            + _log_binom(N - m, B)
            + n_plus * log_p
            + n_minus * log_q
        )

    return weight_fn


def curie_weiss_weight_fn(beta: float) -> WeightFn:
    r"""Law C: ``p(x) \propto exp{ beta S(x)^2 / (2N) }`` on ``{-1,+1}^N``."""
    if beta <= 0.0:
        raise ValueError("beta must be positive")

    def weight_fn(N: int, m: int, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        r = 2.0 * A - m
        u = 2.0 * B - (N - m)
        S = r + u
        return _log_binom(m, A) + _log_binom(N - m, B) + (beta / (2.0 * N)) * S**2

    return weight_fn


def make_weight_fn(spec: LawSpec) -> WeightFn:
    if spec.law == "product":
        return product_weight_fn(spec.param)
    if spec.law == "cw":
        return curie_weiss_weight_fn(spec.param)
    raise ValueError(f"unknown law {spec.law!r}")
