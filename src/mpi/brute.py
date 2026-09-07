"""Brute-force reference implementation over the explicit state space.

Only usable for tiny ``N`` (``2^N`` configurations), but it invokes *none* of
the machinery in :mod:`mpi.core` -- no exchangeability reduction, no binary
reduction of the block KL, no spin-flip symmetry.  It therefore provides a
genuinely independent check of the whole chain: the count representation, the
posterior-shift lemma, the information decomposition, and the boosted-schedule
linearity (tests T2, T6, and the small-``N`` end-to-end test).
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

from .config import LawSpec

__all__ = ["BruteLaw", "build_brute_law"]


def _all_states(N: int) -> np.ndarray:
    return np.array(list(itertools.product((-1, 1), repeat=N)), dtype=int)


@dataclass
class BruteLaw:
    """Explicit ``p^+``, ``p^-``, ``p`` over ``{-1,+1}^N``."""

    N: int
    w: float
    states: np.ndarray   # (2^N, N)
    p_plus: np.ndarray   # (2^N,)
    p_minus: np.ndarray  # (2^N,)

    @property
    def p(self) -> np.ndarray:
        return self.w * self.p_plus + (1.0 - self.w) * self.p_minus

    def q(self, lam: float) -> np.ndarray:
        return lam * self.p_plus + (1.0 - lam) * self.p_minus

    # -- visible-set machinery ---------------------------------------------

    def _visible_keys(self, V: tuple[int, ...]) -> np.ndarray:
        """Integer code of the visible pattern of every state."""
        if not V:
            return np.zeros(self.states.shape[0], dtype=np.int64)
        bits = (self.states[:, list(V)] > 0).astype(np.int64)
        weights = (1 << np.arange(len(V), dtype=np.int64))
        return bits @ weights

    def marginal(self, dist: np.ndarray, V: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        keys = self._visible_keys(V)
        uniq, inv = np.unique(keys, return_inverse=True)
        out = np.zeros(uniq.size)
        np.add.at(out, inv, dist)
        return out, inv

    def beta_V(self, V: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        """``(beta_V per visible pattern, p_V per visible pattern)`` by direct Bayes."""
        pv_plus, _ = self.marginal(self.p_plus, V)
        pv_minus, _ = self.marginal(self.p_minus, V)
        num = self.w * pv_plus
        den = num + (1.0 - self.w) * pv_minus
        with np.errstate(invalid="ignore", divide="ignore"):
            beta = np.where(den > 0.0, num / den, 0.0)
        return beta, den

    def mmse(self, V: tuple[int, ...]) -> float:
        beta, pv = self.beta_V(V)
        return float((pv * beta * (1.0 - beta)).sum())

    def flow_plus(self, V: tuple[int, ...]) -> float:
        beta, _ = self.beta_V(V)
        pv_plus, _ = self.marginal(self.p_plus, V)
        return float((pv_plus * (1.0 - beta)).sum())

    def D_K(self, V: tuple[int, ...], lam: float) -> float:
        r"""``D_K(p || q_lambda)`` straight from Def. `masked-log-discrepancy`.

        ``D_K = E_{x_V ~ p_V} KL( p_K(. | x_V) || q_K(. | x_V) )`` evaluated by
        explicit grouping over visible patterns -- no reduction of any kind.
        """
        p = self.p
        q = self.q(lam)
        keys = self._visible_keys(V)
        uniq, inv = np.unique(keys, return_inverse=True)

        pv = np.zeros(uniq.size)
        qv = np.zeros(uniq.size)
        np.add.at(pv, inv, p)
        np.add.at(qv, inv, q)

        mask = p > 0.0
        # KL(p_K(.|x_V) || q_K(.|x_V)) weighted by p_V and summed:
        #   sum_x p(x) [ log(p(x)/p_V) - log(q(x)/q_V) ]
        terms = p[mask] * (
            np.log(p[mask]) - np.log(pv[inv][mask])
            - np.log(q[mask]) + np.log(qv[inv][mask])
        )
        return float(terms.sum())

    def D_schedule(self, schedule: dict[tuple[int, ...], float], lam: float) -> float:
        return sum(prob * self.D_K(V, lam) for V, prob in schedule.items())

    # -- Per-example sampling-cost moments, with no reduction of any kind ----

    def excess_loss_moments(self, V: tuple[int, ...], lam: float) -> tuple[float, float]:
        r"""``(E[S], E[S^2])`` for ``S = log p_K(X_K|X_V) - log q_K(X_K|X_V)``.

        The first moment is ``D_K`` by definition; the second is what
        :func:`mpi.low_visibility_sampling_cost.log_second_moment` computes
        from the two-point structure of ``S`` on each fibre.
        """
        p, q = self.p, self.q(lam)
        keys = self._visible_keys(V)
        uniq, inv = np.unique(keys, return_inverse=True)
        pv, qv = np.zeros(uniq.size), np.zeros(uniq.size)
        np.add.at(pv, inv, p)
        np.add.at(qv, inv, q)

        mask = p > 0.0
        S = (np.log(p[mask]) - np.log(pv[inv][mask])
             - np.log(q[mask]) + np.log(qv[inv][mask]))
        return float((p[mask] * S).sum()), float((p[mask] * S**2).sum())

    def raw_loss_moments(self, V: tuple[int, ...]) -> tuple[float, float]:
        r"""``(E[L], Var[L])`` for the raw per-example loss ``L = -log p_K(X_K|X_V)``."""
        p = self.p
        keys = self._visible_keys(V)
        uniq, inv = np.unique(keys, return_inverse=True)
        pv = np.zeros(uniq.size)
        np.add.at(pv, inv, p)

        mask = p > 0.0
        L = -(np.log(p[mask]) - np.log(pv[inv][mask]))
        first = float((p[mask] * L).sum())
        return first, float((p[mask] * L**2).sum()) - first * first


def build_brute_law(spec: LawSpec, N: int, w: float) -> BruteLaw:
    states = _all_states(N)
    S = states.sum(axis=1)
    if np.any(S == 0):
        raise ValueError("N must be odd so that S != 0")

    if spec.law == "product":
        theta = spec.param
        # nu_tau(x) = prod_i (1 + tau*theta*x_i)/2
        log_nu_plus = np.log((1.0 + theta * states) / 2.0).sum(axis=1)
        log_nu_minus = np.log((1.0 - theta * states) / 2.0).sum(axis=1)
        base_plus = np.exp(log_nu_plus)
        base_minus = np.exp(log_nu_minus)
    elif spec.law == "cw":
        beta = spec.param
        lw = (beta / (2.0 * N)) * S.astype(float) ** 2
        lw -= lw.max()
        base_plus = np.exp(lw)
        base_minus = base_plus.copy()
    else:
        raise ValueError(f"unknown law {spec.law!r}")

    p_plus = np.where(S > 0, base_plus, 0.0)
    p_minus = np.where(S < 0, base_minus, 0.0)
    p_plus /= p_plus.sum()
    p_minus /= p_minus.sum()
    return BruteLaw(N=N, w=w, states=states, p_plus=p_plus, p_minus=p_minus)
