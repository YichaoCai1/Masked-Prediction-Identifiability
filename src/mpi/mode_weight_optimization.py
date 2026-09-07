"""Direct mode-weight optimisation inside the coherent family ``q_lambda``.

This module deliberately has a single trainable quantity:

``a = logit(lambda)`` with ``q_lambda = lambda p+ + (1-lambda) p-``.

Every posterior used by the loss is consequently induced by the same joint
law.  In particular, no estimate is obtained by evaluating a predictor at an
unseen mask size: the estimate is always exactly ``sigmoid(a)``.

The implementation is NumPy/float64 throughout.  Population quantities reuse
the exact visible-statistic laws in :mod:`mpi.core`; stochastic batches draw
from those conditioned laws rather than from the unconditioned binomial law.
This distinction matters for Law P because ``p+`` and ``p-`` are product laws
*conditioned on the sign of the full sequence*.
"""

from __future__ import annotations

import math
import hashlib
import json
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp

from .config import (
    BRANCH_T_DEFAULT,
    KL_IMPL_DEFAULT,
    RESULTS_DIR,
    LawSpec,
    RunConfig,
    provenance,
)
from .core import VisibleLaw, log_kl_from_logit, softplus, visible_law
from .estimands import log_U
from .tableio import write_table

__all__ = [
    "DEFAULT_N_GRID",
    "DEFAULT_PI_GRID",
    "DEFAULT_S_GRID",
    "QLambdaSchedule",
    "QLambdaConfig",
    "SampleBatch",
    "ExactVisibilityMetrics",
    "ExactScheduleMetrics",
    "TrainingCheckpoint",
    "TrainingResult",
    "CalibrationTables",
    "ExactLawPScheduleSampler",
    "base_posterior_logit",
    "tilted_posterior_logit",
    "lambda_from_a",
    "mode_weight_loss",
    "mode_weight_gradients",
    "paired_excess_losses",
    "paired_excess_risk",
    "sample_schedule_batch",
    "exact_visibility_metrics",
    "exact_schedule_metrics",
    "exact_schedule_metrics_from_a",
    "exact_population_risk",
    "calibrate_schedule_grid",
    "population_gradient_descent",
    "train_mode_weight",
    "run_mode_weight_optimization",
]


DEFAULT_N_GRID: tuple[int, ...] = (31, 63, 127)
DEFAULT_PI_GRID: tuple[float, ...] = (
    0.0,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
)
DEFAULT_S_GRID: tuple[int, ...] = (0, 1, 4)


# ---------------------------------------------------------------------------
# Configuration and plain result records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QLambdaSchedule:
    """A two-point visible-size schedule.

    It assigns mass ``1-pi`` to ``m_base`` and mass ``pi`` to ``s``.  Zero
    weight components are removed, and equal support points are merged, by
    :meth:`support`.
    """

    name: str = "blind"
    m_base: int = 31
    s: int = 1
    pi: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.m_base, (int, np.integer)):
            raise TypeError("m_base must be an integer")
        if not isinstance(self.s, (int, np.integer)):
            raise TypeError("s must be an integer")
        if not math.isfinite(float(self.pi)) or not 0.0 <= self.pi <= 1.0:
            raise ValueError("pi must lie in [0, 1]")

    @property
    def label(self) -> str:
        if self.pi == 0.0:
            return f"blind (m={self.m_base})"
        return f"s={self.s}, pi={self.pi:g}"

    def support(self) -> tuple[tuple[int, float], ...]:
        """Return the positive-mass support, merging coincident sizes."""
        weights: dict[int, float] = {}
        for m, weight in ((int(self.m_base), 1.0 - self.pi), (int(self.s), self.pi)):
            if weight > 0.0:
                weights[m] = weights.get(m, 0.0) + float(weight)
        return tuple(sorted(weights.items()))


@dataclass(frozen=True)
class QLambdaConfig:
    """All mathematical and optimisation settings for one schedule cell."""

    N: int = 63
    theta: float = 0.8
    w: float = 0.5
    schedule: QLambdaSchedule = field(default_factory=QLambdaSchedule)
    learning_rate: float = 0.1
    batch_size: int = 512
    max_steps: int = 100_000
    checkpoint_every: int = 100
    validation_size: int = 10_000
    consecutive_successes: int = 3
    early_stop: bool = False
    branch_T: float = BRANCH_T_DEFAULT
    kl_impl: str = KL_IMPL_DEFAULT

    def __post_init__(self) -> None:
        if not isinstance(self.N, (int, np.integer)) or self.N <= 0 or self.N % 2 == 0:
            raise ValueError("N must be a positive odd integer")
        if not math.isfinite(self.theta) or not 0.0 < self.theta < 1.0:
            raise ValueError("theta must lie in (0, 1) for Law P")
        if not math.isfinite(self.w) or not 0.0 < self.w < 1.0:
            raise ValueError("w must lie in (0, 1)")
        for name, m in (("m_base", self.schedule.m_base), ("s", self.schedule.s)):
            if not 0 <= m <= self.N:
                raise ValueError(f"{name} must lie in [0, N]")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        for name, value in (
            ("batch_size", self.batch_size),
            ("max_steps", self.max_steps),
            ("checkpoint_every", self.checkpoint_every),
            ("validation_size", self.validation_size),
            ("consecutive_successes", self.consecutive_successes),
        ):
            if not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(self.branch_T) or self.branch_T <= 0.0:
            raise ValueError("branch_T must be positive")
        if self.kl_impl not in {"stable", "branched"}:
            raise ValueError("kl_impl must be 'stable' or 'branched'")

    @property
    def a_star(self) -> float:
        return _logit(self.w)

    @property
    def lr(self) -> float:
        """Short, read-only alias used in plotting/table code."""
        return self.learning_rate

    @property
    def law_spec(self) -> LawSpec:
        return _law_p_spec(self.theta)


@dataclass(frozen=True)
class SampleBatch:
    """A batch of exact sufficient statistics and mode labels.

    ``base_logit[i]`` is the exact ``p`` posterior logit for ``(m[i],r[i])``.
    ``w`` is retained so that loss calls cannot silently tilt around the wrong
    truth.  It may be omitted only when an explicit ``w`` is supplied to the
    loss function (useful for small hand-built validation batches).
    """

    m: np.ndarray
    r: np.ndarray
    z: np.ndarray
    base_logit: np.ndarray
    w: float | None = None

    def __post_init__(self) -> None:
        m = np.asarray(self.m, dtype=np.int64)
        r = np.asarray(self.r, dtype=np.int64)
        z = np.asarray(self.z, dtype=np.float64)
        base = np.asarray(self.base_logit, dtype=np.float64)
        if not (m.ndim == r.ndim == z.ndim == base.ndim == 1):
            raise ValueError("batch arrays must be one-dimensional")
        if not (m.size == r.size == z.size == base.size):
            raise ValueError("batch arrays must have equal lengths")
        if m.size == 0:
            raise ValueError("a batch must contain at least one sample")
        if not np.all((z == 0.0) | (z == 1.0)):
            raise ValueError("z must contain only 0 and 1")
        if self.w is not None and not 0.0 < float(self.w) < 1.0:
            raise ValueError("w must lie in (0, 1)")
        object.__setattr__(self, "m", m)
        object.__setattr__(self, "r", r)
        object.__setattr__(self, "z", z)
        object.__setattr__(self, "base_logit", base)

    def __len__(self) -> int:
        return int(self.z.size)


@dataclass(frozen=True)
class ExactVisibilityMetrics:
    """Exact quantities for a single deterministic visible size."""

    N: int
    theta: float
    w: float
    m: int
    lambda_value: float
    a: float
    risk: float
    direct_bce_risk: float
    bayes_risk: float
    excess_risk: float
    log_excess_risk: float
    gradient: float
    log_abs_gradient: float
    hessian: float
    log_hessian: float
    curvature: float
    log_curvature: float

    @property
    def U(self) -> float:
        """Residual mode MMSE; also the truth Hessian in the ``a`` coordinate."""
        return self.curvature

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class ExactScheduleMetrics:
    """Exact population quantities for a two-point schedule.

    ``hessian`` is evaluated at the requested parameter.  ``curvature`` is
    always the Hessian at the truth and hence equals the schedule-averaged
    residual MMSE.  Keeping both names prevents the common coordinate error
    of substituting ``d²D/dlambda²`` for ``d²R/da²``.
    """

    schedule: str
    N: int
    theta: float
    w: float
    m_base: int
    s: int
    pi: float
    lambda_value: float
    a: float
    risk: float
    direct_bce_risk: float
    bayes_risk: float
    excess_risk: float
    log_excess_risk: float
    risk_identity_error: float
    gradient: float
    log_abs_gradient: float
    hessian: float
    log_hessian: float
    curvature: float
    log_curvature: float
    inverse_curvature: float
    predicted_half_time: float

    @property
    def H(self) -> float:
        return self.curvature

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class TrainingCheckpoint:
    """One deterministic or stochastic optimisation checkpoint."""

    step: int
    processed_examples: int
    a: float
    lambda_value: float
    error: float
    exact_excess_risk: float
    log_exact_excess_risk: float
    exact_gradient: float
    exact_hessian: float
    update_gradient: float
    gradient_std: float
    gradient_se: float
    batch_loss: float
    paired_excess_risk: float
    paired_excess_se: float
    threshold_met: bool

    @property
    def lambda_hat(self) -> float:
        """The sole model estimate; necessarily ``sigmoid(a)``."""
        return self.lambda_value

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass(frozen=True)
class TrainingResult:
    """Complete result of one population-GD or minibatch-SGD run."""

    method: Literal["population_gd", "sgd"]
    config: QLambdaConfig
    lambda_init: float
    a_init: float
    seed: int | None
    checkpoints: tuple[TrainingCheckpoint, ...]
    t_half: int | None
    censored: bool
    steps_completed: int
    final_a: float
    final_lambda: float

    @property
    def trajectory(self) -> tuple[TrainingCheckpoint, ...]:
        return self.checkpoints

    def checkpoint_records(self) -> list[dict[str, Any]]:
        common = {
            "method": self.method,
            "schedule": self.config.schedule.name,
            "N": self.config.N,
            "theta": self.config.theta,
            "w": self.config.w,
            "m_base": self.config.schedule.m_base,
            "s": self.config.schedule.s,
            "pi": self.config.schedule.pi,
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "lambda_init": self.lambda_init,
            "seed": self.seed,
            "t_half": self.t_half,
            "censored": self.censored,
        }
        return [{**common, **checkpoint.as_dict()} for checkpoint in self.checkpoints]

    def summary_record(self) -> dict[str, Any]:
        final = self.checkpoints[-1]
        return {
            "method": self.method,
            "schedule": self.config.schedule.name,
            "N": self.config.N,
            "theta": self.config.theta,
            "w": self.config.w,
            "m_base": self.config.schedule.m_base,
            "s": self.config.schedule.s,
            "pi": self.config.schedule.pi,
            "learning_rate": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "max_steps": self.config.max_steps,
            "steps_completed": self.steps_completed,
            "lambda_init": self.lambda_init,
            "seed": self.seed,
            "final_a": self.final_a,
            "final_lambda": self.final_lambda,
            "final_exact_excess_risk": final.exact_excess_risk,
            "final_paired_excess_risk": final.paired_excess_risk,
            "t_half": self.t_half,
            "censored": self.censored,
        }


@dataclass(frozen=True)
class CalibrationTables:
    """Tidy Phase-A records, ready to pass to ``pandas.DataFrame``."""

    visibility: tuple[dict[str, Any], ...]
    schedules: tuple[dict[str, Any], ...]
    risk_profiles: tuple[dict[str, Any], ...]

    def as_dataframes(self) -> dict[str, Any]:
        """Return dataframes lazily, keeping pandas out of the core hot path."""
        import pandas as pd

        return {
            "visibility": pd.DataFrame(self.visibility),
            "schedules": pd.DataFrame(self.schedules),
            "risk_profiles": pd.DataFrame(self.risk_profiles),
        }


# ---------------------------------------------------------------------------
# Stable scalar and loss helpers
# ---------------------------------------------------------------------------


def _law_p_spec(theta: float) -> LawSpec:
    return LawSpec("product", float(theta), True, rf"Law P, $\theta={theta:g}$")


def _logit(probability: float) -> float:
    if not math.isfinite(probability) or not 0.0 < probability < 1.0:
        raise ValueError("probability must lie in (0, 1)")
    return math.log(probability) - math.log1p(-probability)


def lambda_from_a(a: float | np.ndarray) -> float | np.ndarray:
    """Return the model's mode-weight estimate ``sigmoid(a)`` stably."""
    out = expit(np.asarray(a, dtype=np.float64))
    return float(out) if out.ndim == 0 else out


def base_posterior_logit(
    N: int,
    theta: float,
    w: float,
    m: int | np.ndarray,
    r: int | np.ndarray,
) -> float | np.ndarray:
    """Exact ``logit p(Z=1 | X_V)`` for Law P sufficient statistics.

    Arrays of mixed visible sizes are supported.  Each ``r`` must be in
    ``{-m,-m+2,...,m}``; no interpolation or clipping is performed.
    """
    # Reuse QLambdaConfig's mathematical validation without imposing an
    # optimisation setup whose default schedule may not fit a small N.
    if not isinstance(N, (int, np.integer)) or N <= 0 or N % 2 == 0:
        raise ValueError("N must be a positive odd integer")
    if not math.isfinite(theta) or not 0.0 < theta < 1.0:
        raise ValueError("theta must lie in (0, 1)")
    if not math.isfinite(w) or not 0.0 < w < 1.0:
        raise ValueError("w must lie in (0, 1)")

    m_arr, r_arr = np.broadcast_arrays(np.asarray(m), np.asarray(r))
    if not np.issubdtype(m_arr.dtype, np.integer) or not np.issubdtype(r_arr.dtype, np.integer):
        # Accept integer-valued floats, but reject silent truncation.
        if not np.all(np.equal(m_arr, np.floor(m_arr))) or not np.all(np.equal(r_arr, np.floor(r_arr))):
            raise ValueError("m and r must be integer-valued")
    m_int = m_arr.astype(np.int64)
    r_int = r_arr.astype(np.int64)
    if np.any((m_int < 0) | (m_int > N)):
        raise ValueError("m must lie in [0, N]")
    if np.any(np.abs(r_int) > m_int) or np.any((r_int + m_int) % 2 != 0):
        raise ValueError("r must lie on the magnetisation grid {-m,-m+2,...,m}")

    out = np.empty(m_int.shape, dtype=np.float64)
    spec = _law_p_spec(theta)
    for m_value in np.unique(m_int):
        mask = m_int == m_value
        vl = visible_law(spec, int(N), int(m_value), float(w))
        index = ((r_int[mask] + int(m_value)) // 2).astype(np.int64)
        out[mask] = vl.logit_b[index]
    return float(out) if out.ndim == 0 else out


def tilted_posterior_logit(
    base_logit: float | np.ndarray,
    a: float,
    w: float,
) -> float | np.ndarray:
    """Apply the exact common-logit tilt induced by ``q_lambda``."""
    if not math.isfinite(float(a)):
        raise ValueError("a must be finite")
    shift = float(a) - _logit(float(w))
    out = np.asarray(base_logit, dtype=np.float64) + shift
    return float(out) if out.ndim == 0 else out


def _batch_w(batch: SampleBatch, w: float | None) -> float:
    value = batch.w if w is None else w
    if value is None:
        raise ValueError("w must be supplied either in SampleBatch or explicitly")
    if batch.w is not None and w is not None and float(batch.w) != float(w):
        raise ValueError("explicit w disagrees with SampleBatch.w")
    if not 0.0 < float(value) < 1.0:
        raise ValueError("w must lie in (0, 1)")
    return float(value)


def mode_weight_loss(
    a: float,
    batch: SampleBatch,
    w: float | None = None,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> float | np.ndarray:
    """Stable BCE-with-logits for the exact tilted posterior.

    Branching on the binary label avoids the indeterminate ``inf - inf`` in
    ``softplus(y) - z*y`` on deterministic posterior fibres.
    """
    truth_w = _batch_w(batch, w)
    y = np.asarray(tilted_posterior_logit(batch.base_logit, a, truth_w))
    losses = np.where(batch.z == 1.0, softplus(-y), softplus(y))
    if reduction == "none":
        return losses
    if reduction == "sum":
        return float(np.sum(losses, dtype=np.float64))
    if reduction == "mean":
        return float(np.mean(losses, dtype=np.float64))
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def mode_weight_gradients(
    a: float,
    batch: SampleBatch,
    w: float | None = None,
) -> np.ndarray:
    """Per-example analytic gradients ``beta_lambda - Z``."""
    truth_w = _batch_w(batch, w)
    y = np.asarray(tilted_posterior_logit(batch.base_logit, a, truth_w))
    return np.asarray(expit(y) - batch.z, dtype=np.float64)


def paired_excess_losses(
    a: float,
    batch: SampleBatch,
    w: float | None = None,
) -> np.ndarray:
    """Paired loss differences on common examples, never raw-loss contrasts."""
    truth_w = _batch_w(batch, w)
    a_star = _logit(truth_w)
    current = np.asarray(mode_weight_loss(a, batch, truth_w, reduction="none"))
    optimum = np.asarray(mode_weight_loss(a_star, batch, truth_w, reduction="none"))
    # Deterministic fibres produce 0-0, not a clipped approximation.
    return np.asarray(current - optimum, dtype=np.float64)


def paired_excess_risk(
    a: float,
    batch: SampleBatch,
    w: float | None = None,
) -> tuple[float, float]:
    """Return paired excess-risk mean and its usual iid standard error."""
    values = paired_excess_losses(a, batch, w)
    mean = float(np.mean(values, dtype=np.float64))
    if len(values) <= 1:
        return mean, 0.0
    return mean, float(np.std(values, ddof=1) / math.sqrt(len(values)))


# ---------------------------------------------------------------------------
# Exact conditioned sampler
# ---------------------------------------------------------------------------


class ExactLawPScheduleSampler:
    """Cached exact sampler for ``(m,r,Z)`` under a QLambda schedule."""

    def __init__(self, config: QLambdaConfig, rng: np.random.Generator | None = None):
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()
        if not isinstance(self.rng, np.random.Generator):
            raise TypeError("rng must be numpy.random.Generator")
        self._visible: dict[int, VisibleLaw] = {
            m: visible_law(config.law_spec, config.N, m, config.w)
            for m, _ in config.schedule.support()
        }
        self._probabilities: dict[tuple[int, int], np.ndarray] = {}
        for m, vl in self._visible.items():
            for z, log_probs in ((1, vl.log_h_plus), (0, vl.log_h_minus)):
                probabilities = np.exp(log_probs - logsumexp(log_probs))
                probabilities /= probabilities.sum(dtype=np.float64)
                self._probabilities[(m, z)] = probabilities

    def sample(self, batch_size: int | None = None) -> SampleBatch:
        size = self.config.batch_size if batch_size is None else batch_size
        if not isinstance(size, (int, np.integer)) or size <= 0:
            raise ValueError("batch_size must be a positive integer")
        size = int(size)

        z = self.rng.binomial(1, self.config.w, size=size).astype(np.float64)
        if self.config.schedule.pi <= 0.0:
            m = np.full(size, self.config.schedule.m_base, dtype=np.int64)
        elif self.config.schedule.pi >= 1.0:
            m = np.full(size, self.config.schedule.s, dtype=np.int64)
        else:
            boosted = self.rng.random(size) < self.config.schedule.pi
            m = np.where(
                boosted, self.config.schedule.s, self.config.schedule.m_base
            ).astype(np.int64)

        r = np.empty(size, dtype=np.int64)
        base = np.empty(size, dtype=np.float64)
        for m_value in np.unique(m):
            vl = self._visible[int(m_value)]
            for z_value in (0, 1):
                rows = np.flatnonzero((m == m_value) & (z == z_value))
                if rows.size == 0:
                    continue
                count_index = self.rng.choice(
                    int(m_value) + 1,
                    size=rows.size,
                    p=self._probabilities[(int(m_value), z_value)],
                )
                r[rows] = 2 * count_index - int(m_value)
                base[rows] = vl.logit_b[count_index]
        return SampleBatch(m=m, r=r, z=z, base_logit=base, w=self.config.w)


def sample_schedule_batch(
    config: QLambdaConfig,
    rng: np.random.Generator,
    batch_size: int | None = None,
) -> SampleBatch:
    """Sample one batch from the exact conditioned Law P visible marginals."""
    return ExactLawPScheduleSampler(config, rng).sample(batch_size)


# ---------------------------------------------------------------------------
# Exact population evaluator
# ---------------------------------------------------------------------------


def _log_abs_expm1(d: float) -> float:
    if d == 0.0:
        return -math.inf
    if d > 0.0:
        # log(exp(d)-1) = d + log(1-exp(-d)); expm1 is accurate near zero.
        return d + math.log(-math.expm1(-d))
    return math.log(-math.expm1(d))


def _component_log_abs_gradient(vl: VisibleLaw, d: float) -> float:
    """Log absolute ``E[sigmoid(x+d)-sigmoid(x)]`` without cancellation."""
    if d == 0.0:
        return -math.inf
    x = vl.logit_b
    finite = np.isfinite(x)
    if not np.any(finite):
        return -math.inf
    xf = x[finite]
    terms = (
        vl.log_h[finite]
        + xf
        + _log_abs_expm1(d)
        - softplus(xf)
        - softplus(xf + d)
    )
    return float(logsumexp(terms))


def _component_log_hessian(vl: VisibleLaw, d: float) -> float:
    y = vl.logit_b + d
    return float(logsumexp(vl.log_h - softplus(y) - softplus(-y)))


def _component_log_excess(
    vl: VisibleLaw,
    d: float,
    branch_T: float,
    kl_impl: str,
) -> float:
    if d == 0.0:
        return -math.inf
    log_kl = log_kl_from_logit(vl.logit_b, d, branch_T, kl_impl)
    return float(logsumexp(vl.log_h + log_kl))


def _weighted_loss(log_prob: np.ndarray, loss: np.ndarray, prior: float) -> float:
    supported = np.isfinite(log_prob)
    if not np.any(supported):
        return 0.0
    values = np.asarray(loss[supported], dtype=np.float64)
    if np.any(~np.isfinite(values)):
        return math.inf
    return float(prior * np.dot(np.exp(log_prob[supported]), values))


def _component_bce_risk(vl: VisibleLaw, d: float) -> float:
    y = vl.logit_b + d
    return _weighted_loss(vl.log_h_plus, softplus(-y), vl.w) + _weighted_loss(
        vl.log_h_minus, softplus(y), 1.0 - vl.w
    )


def _weighted_logsum(log_values: Sequence[float], weights: Sequence[float]) -> float:
    terms = [math.log(weight) + value for value, weight in zip(log_values, weights) if weight > 0]
    return float(logsumexp(terms)) if terms else -math.inf


class _ExactScheduleEvaluator:
    """Precomputed evaluator shared by calibration and optimisation loops."""

    def __init__(self, config: QLambdaConfig):
        self.config = config
        self.components = config.schedule.support()
        self.visible = {
            m: visible_law(config.law_spec, config.N, m, config.w)
            for m, _ in self.components
        }
        self.weights = tuple(weight for _, weight in self.components)
        self.log_u = tuple(log_U(self.visible[m]) for m, _ in self.components)
        self.log_curvature = _weighted_logsum(self.log_u, self.weights)
        self.curvature = _exp_or_zero(self.log_curvature)
        self.bayes_risk = sum(
            weight * _component_bce_risk(self.visible[m], 0.0)
            for m, weight in self.components
        )

    def gradient(self, a: float) -> float:
        """Exact gradient only, for the population-GD hot loop."""
        if not math.isfinite(a):
            raise ValueError("a must be finite")
        d = a - self.config.a_star
        if d == 0.0:
            return 0.0
        component_logs = tuple(
            _component_log_abs_gradient(self.visible[m], d) for m, _ in self.components
        )
        log_abs = _weighted_logsum(component_logs, self.weights)
        return math.copysign(_exp_or_zero(log_abs), d)

    def metrics_from_a(self, a: float) -> ExactScheduleMetrics:
        if not math.isfinite(a):
            raise ValueError("a must be finite")
        cfg = self.config
        d = a - cfg.a_star
        lambda_value = cfg.w if d == 0.0 else float(lambda_from_a(a))

        component_log_excess = tuple(
            _component_log_excess(self.visible[m], d, cfg.branch_T, cfg.kl_impl)
            for m, _ in self.components
        )
        log_excess = _weighted_logsum(component_log_excess, self.weights)
        excess = _exp_or_zero(log_excess)

        component_log_gradient = tuple(
            _component_log_abs_gradient(self.visible[m], d) for m, _ in self.components
        )
        log_abs_gradient = _weighted_logsum(component_log_gradient, self.weights)
        gradient = 0.0 if d == 0.0 else math.copysign(
            _exp_or_zero(log_abs_gradient), d
        )

        component_log_hessian = tuple(
            _component_log_hessian(self.visible[m], d) for m, _ in self.components
        )
        log_hessian = _weighted_logsum(component_log_hessian, self.weights)
        hessian = _exp_or_zero(log_hessian)

        direct_risk = sum(
            weight * _component_bce_risk(self.visible[m], d)
            for m, weight in self.components
        )
        # This expression is both the exact KL identity and more accurate than
        # subtracting two nearly equal cross-entropies on very flat cells.
        risk = self.bayes_risk + excess
        identity_error = direct_risk - risk
        inverse = math.inf if self.curvature == 0.0 else 1.0 / self.curvature
        predicted = math.inf if self.curvature == 0.0 else (
            math.log(2.0) / (cfg.learning_rate * self.curvature)
        )
        return ExactScheduleMetrics(
            schedule=cfg.schedule.name,
            N=cfg.N,
            theta=cfg.theta,
            w=cfg.w,
            m_base=cfg.schedule.m_base,
            s=cfg.schedule.s,
            pi=cfg.schedule.pi,
            lambda_value=lambda_value,
            a=float(a),
            risk=risk,
            direct_bce_risk=direct_risk,
            bayes_risk=self.bayes_risk,
            excess_risk=excess,
            log_excess_risk=log_excess,
            risk_identity_error=identity_error,
            gradient=gradient,
            log_abs_gradient=log_abs_gradient,
            hessian=hessian,
            log_hessian=log_hessian,
            curvature=self.curvature,
            log_curvature=self.log_curvature,
            inverse_curvature=inverse,
            predicted_half_time=predicted,
        )


def _exp_or_zero(log_value: float) -> float:
    if log_value == -math.inf:
        return 0.0
    return float(math.exp(log_value))


def exact_visibility_metrics(
    N: int,
    theta: float,
    w: float,
    m: int,
    lambda_value: float,
    *,
    branch_T: float = BRANCH_T_DEFAULT,
    kl_impl: str = KL_IMPL_DEFAULT,
) -> ExactVisibilityMetrics:
    """Evaluate one visible-size cell in the trained ``a`` coordinate."""
    schedule = QLambdaSchedule(name=f"m={m}", m_base=m, s=m, pi=0.0)
    config = QLambdaConfig(
        N=N,
        theta=theta,
        w=w,
        schedule=schedule,
        branch_T=branch_T,
        kl_impl=kl_impl,
    )
    metrics = exact_schedule_metrics(config, lambda_value)
    return ExactVisibilityMetrics(
        N=N,
        theta=theta,
        w=w,
        m=m,
        lambda_value=metrics.lambda_value,
        a=metrics.a,
        risk=metrics.risk,
        direct_bce_risk=metrics.direct_bce_risk,
        bayes_risk=metrics.bayes_risk,
        excess_risk=metrics.excess_risk,
        log_excess_risk=metrics.log_excess_risk,
        gradient=metrics.gradient,
        log_abs_gradient=metrics.log_abs_gradient,
        hessian=metrics.hessian,
        log_hessian=metrics.log_hessian,
        curvature=metrics.curvature,
        log_curvature=metrics.log_curvature,
    )


def exact_schedule_metrics(
    config: QLambdaConfig,
    lambda_value: float,
) -> ExactScheduleMetrics:
    """Exact schedule risk, gradient, current Hessian, and truth curvature."""
    if not math.isfinite(lambda_value) or not 0.0 < lambda_value < 1.0:
        raise ValueError("lambda_value must lie in (0, 1)")
    # Preserve exact zero at the truth instead of round-tripping through
    # sigmoid(logit(w)), which need not reproduce asymmetric w bit-for-bit.
    a = config.a_star if lambda_value == config.w else _logit(lambda_value)
    return _ExactScheduleEvaluator(config).metrics_from_a(a)


def exact_schedule_metrics_from_a(
    config: QLambdaConfig,
    a: float,
) -> ExactScheduleMetrics:
    """As :func:`exact_schedule_metrics`, with the optimised coordinate input."""
    return _ExactScheduleEvaluator(config).metrics_from_a(float(a))


def exact_population_risk(config: QLambdaConfig, a: float) -> float:
    """Convenience scalar for finite-difference gradient/Hessian checks."""
    return _ExactScheduleEvaluator(config).metrics_from_a(float(a)).risk


# ---------------------------------------------------------------------------
# Phase-A exact calibration
# ---------------------------------------------------------------------------


def calibrate_schedule_grid(
    *,
    theta: float = 0.8,
    w: float = 0.5,
    N_grid: Sequence[int] = DEFAULT_N_GRID,
    pi_grid: Sequence[float] = DEFAULT_PI_GRID,
    s_grid: Sequence[int] = DEFAULT_S_GRID,
    lambda_probes: Sequence[float] = (0.4, 0.6),
    lambda_grid: Sequence[float] | None = None,
    visibility_grids: Mapping[int, Iterable[int]] | None = None,
    learning_rate: float = 0.1,
    branch_T: float = BRANCH_T_DEFAULT,
    kl_impl: str = KL_IMPL_DEFAULT,
) -> CalibrationTables:
    """Compute all exact Phase-A quantities before any stochastic selection.

    The returned records contain the full visibility curves, every candidate
    schedule/probe gradient and time scale, and dense exact excess-risk
    profiles.  Cell selection is intentionally left to the runner so its
    outcome-independent rule and any compute-budget pruning can be recorded.
    """
    if lambda_grid is None:
        lambda_grid = np.linspace(0.01, 0.99, 197)
    lambda_grid = tuple(float(value) for value in lambda_grid)
    probes = tuple(float(value) for value in lambda_probes)
    s_values = tuple(int(value) for value in s_grid)
    canonical_blind_s = 1 if 1 in s_values else (s_values[0] if s_values else None)

    visibility_records: list[dict[str, Any]] = []
    schedule_records: list[dict[str, Any]] = []
    risk_records: list[dict[str, Any]] = []

    for N_raw in N_grid:
        N = int(N_raw)
        m_base = N // 2
        if visibility_grids is None:
            m_values = range(0, N + 1)
        else:
            if N not in visibility_grids:
                raise ValueError(f"visibility_grids has no entry for N={N}")
            m_values = tuple(int(m) for m in visibility_grids[N])

        # Visibility records use one fixed nearby probe for the optional
        # gradient overlay; U itself is independent of that choice.
        overlay_probe = probes[0]
        for m in m_values:
            vm = exact_visibility_metrics(
                N, theta, w, int(m), overlay_probe, branch_T=branch_T, kl_impl=kl_impl
            )
            visibility_records.append(
                {
                    "N": N,
                    "theta": theta,
                    "w": w,
                    "m": int(m),
                    "U": vm.curvature,
                    "log_U": vm.log_curvature,
                    "lambda_probe": overlay_probe,
                    "gradient": vm.gradient,
                    "abs_gradient": abs(vm.gradient),
                    "log_abs_gradient": vm.log_abs_gradient,
                }
            )

        for s in s_values:
            for pi_raw in pi_grid:
                pi = float(pi_raw)
                # All pi=0 schedules are the same delta_{m_base}; retain one
                # canonical blind row/profile instead of silently triplicating
                # it once for every otherwise-unused boost size.
                if pi == 0.0 and s != canonical_blind_s:
                    continue
                schedule = QLambdaSchedule(
                    name=("blind" if pi == 0.0 else f"s={s},pi={pi:g}"),
                    m_base=m_base,
                    s=s,
                    pi=pi,
                )
                cfg = QLambdaConfig(
                    N=N,
                    theta=theta,
                    w=w,
                    schedule=schedule,
                    learning_rate=learning_rate,
                    branch_T=branch_T,
                    kl_impl=kl_impl,
                )
                evaluator = _ExactScheduleEvaluator(cfg)
                truth = evaluator.metrics_from_a(cfg.a_star)
                for probe in probes:
                    metrics = evaluator.metrics_from_a(_logit(probe))
                    schedule_records.append(
                        {
                            **metrics.as_dict(),
                            "lambda_probe": probe,
                            "H": truth.curvature,
                            "log_H": truth.log_curvature,
                            "inverse_H": truth.inverse_curvature,
                            "predicted_half_time": truth.predicted_half_time,
                        }
                    )
                for lambda_value in lambda_grid:
                    metrics = evaluator.metrics_from_a(
                        cfg.a_star if lambda_value == w else _logit(lambda_value)
                    )
                    risk_records.append(
                        {
                            "schedule": schedule.name,
                            "N": N,
                            "theta": theta,
                            "w": w,
                            "m_base": m_base,
                            "s": s,
                            "pi": pi,
                            "lambda_value": lambda_value,
                            "exact_excess_risk": metrics.excess_risk,
                            "log_exact_excess_risk": metrics.log_excess_risk,
                        }
                    )

    return CalibrationTables(
        visibility=tuple(visibility_records),
        schedules=tuple(schedule_records),
        risk_profiles=tuple(risk_records),
    )


# ---------------------------------------------------------------------------
# Population gradient descent and minibatch SGD
# ---------------------------------------------------------------------------


def _validate_lambda_init(lambda_init: float) -> float:
    value = float(lambda_init)
    if not math.isfinite(value) or not 0.0 < value < 1.0:
        raise ValueError("lambda_init must lie in (0, 1)")
    return value


def _checkpoint(
    evaluator: _ExactScheduleEvaluator,
    *,
    step: int,
    processed_examples: int,
    a: float,
    initial_error: float,
    update_gradient: float = math.nan,
    gradient_std: float = math.nan,
    gradient_se: float = math.nan,
    batch_loss: float = math.nan,
    validation_batch: SampleBatch | None = None,
) -> TrainingCheckpoint:
    metrics = evaluator.metrics_from_a(a)
    if validation_batch is None:
        paired_mean, paired_se = math.nan, math.nan
    else:
        paired_mean, paired_se = paired_excess_risk(a, validation_batch)
    error = abs(metrics.lambda_value - evaluator.config.w)
    return TrainingCheckpoint(
        step=int(step),
        processed_examples=int(processed_examples),
        a=float(a),
        lambda_value=metrics.lambda_value,
        error=error,
        exact_excess_risk=metrics.excess_risk,
        log_exact_excess_risk=metrics.log_excess_risk,
        exact_gradient=metrics.gradient,
        exact_hessian=metrics.hessian,
        update_gradient=float(update_gradient),
        gradient_std=float(gradient_std),
        gradient_se=float(gradient_se),
        batch_loss=float(batch_loss),
        paired_excess_risk=paired_mean,
        paired_excess_se=paired_se,
        threshold_met=error <= 0.5 * initial_error,
    )


def population_gradient_descent(
    config: QLambdaConfig,
    lambda_init: float,
    *,
    max_steps: int | None = None,
    learning_rate: float | None = None,
) -> TrainingResult:
    """Fixed-step deterministic GD on the exactly enumerated population risk."""
    lambda_init = _validate_lambda_init(lambda_init)
    steps_budget = config.max_steps if max_steps is None else int(max_steps)
    eta = config.learning_rate if learning_rate is None else float(learning_rate)
    if steps_budget <= 0:
        raise ValueError("max_steps must be positive")
    if not math.isfinite(eta) or eta <= 0.0:
        raise ValueError("learning_rate must be positive")

    evaluator = _ExactScheduleEvaluator(config)
    a = _logit(lambda_init)
    initial_error = abs(lambda_init - config.w)
    checkpoints = [
        _checkpoint(
            evaluator,
            step=0,
            processed_examples=0,
            a=a,
            initial_error=initial_error,
        )
    ]
    t_half: int | None = 0 if initial_error == 0.0 else None
    steps_completed = 0

    for step in range(1, steps_budget + 1):
        gradient = evaluator.gradient(a)
        a -= eta * gradient
        steps_completed = step
        crossed_now = False
        if t_half is None and abs(float(lambda_from_a(a)) - config.w) <= 0.5 * initial_error:
            t_half = step
            crossed_now = True
        should_log = (
            step == 1
            or step % config.checkpoint_every == 0
            or step == steps_budget
            or crossed_now
        )
        if should_log:
            checkpoints.append(
                _checkpoint(
                    evaluator,
                    step=step,
                    processed_examples=step,
                    a=a,
                    initial_error=initial_error,
                    update_gradient=gradient,
                    gradient_std=0.0,
                    gradient_se=0.0,
                )
            )
        if crossed_now and config.early_stop:
            break

    return TrainingResult(
        method="population_gd",
        config=config,
        lambda_init=lambda_init,
        a_init=_logit(lambda_init),
        seed=None,
        checkpoints=tuple(checkpoints),
        t_half=t_half,
        censored=t_half is None,
        steps_completed=steps_completed,
        final_a=float(a),
        final_lambda=float(lambda_from_a(a)),
    )


def train_mode_weight(
    config: QLambdaConfig,
    lambda_init: float,
    seed: int,
    *,
    max_steps: int | None = None,
    learning_rate: float | None = None,
) -> TrainingResult:
    """Train the single scalar ``a`` with fixed-rate minibatch SGD.

    A fixed, independently sampled validation batch supplies paired excess
    losses.  A stochastic half-time is accepted only after the configured
    number of consecutive *evaluation checkpoints* satisfy the error
    criterion.  A missing crossing remains ``None`` and is marked censored.
    """
    lambda_init = _validate_lambda_init(lambda_init)
    if not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    steps_budget = config.max_steps if max_steps is None else int(max_steps)
    eta = config.learning_rate if learning_rate is None else float(learning_rate)
    if steps_budget <= 0:
        raise ValueError("max_steps must be positive")
    if not math.isfinite(eta) or eta <= 0.0:
        raise ValueError("learning_rate must be positive")

    # Separate deterministic streams keep the trajectory invariant to changes
    # in validation size/checkpoint frequency.
    seed_sequence = np.random.SeedSequence(int(seed))
    train_seed, validation_seed = seed_sequence.spawn(2)
    sampler = ExactLawPScheduleSampler(config, np.random.default_rng(train_seed))
    validation_sampler = ExactLawPScheduleSampler(
        config, np.random.default_rng(validation_seed)
    )
    validation_batch = validation_sampler.sample(config.validation_size)
    evaluator = _ExactScheduleEvaluator(config)

    a = _logit(lambda_init)
    initial_error = abs(lambda_init - config.w)
    checkpoints = [
        _checkpoint(
            evaluator,
            step=0,
            processed_examples=0,
            a=a,
            initial_error=initial_error,
            validation_batch=validation_batch,
        )
    ]
    t_half: int | None = 0 if initial_error == 0.0 else None
    success_count = 0
    success_start: int | None = None
    steps_completed = 0

    for step in range(1, steps_budget + 1):
        batch = sampler.sample(config.batch_size)
        gradients = mode_weight_gradients(a, batch)
        gradient = float(np.mean(gradients, dtype=np.float64))
        gradient_std = (
            float(np.std(gradients, ddof=1)) if len(gradients) > 1 else 0.0
        )
        gradient_se = gradient_std / math.sqrt(len(gradients))
        batch_loss = float(mode_weight_loss(a, batch))
        a -= eta * gradient
        steps_completed = step

        should_log = (
            step == 1
            or step % config.checkpoint_every == 0
            or step == steps_budget
        )
        if not should_log:
            continue
        point = _checkpoint(
            evaluator,
            step=step,
            processed_examples=step * config.batch_size,
            a=a,
            initial_error=initial_error,
            update_gradient=gradient,
            gradient_std=gradient_std,
            gradient_se=gradient_se,
            batch_loss=batch_loss,
            validation_batch=validation_batch,
        )
        checkpoints.append(point)

        if t_half is None:
            if point.threshold_met:
                if success_count == 0:
                    success_start = step
                success_count += 1
                if success_count >= config.consecutive_successes:
                    # Report the first checkpoint of the sustained crossing,
                    # not the later checkpoint at which it was confirmed.
                    t_half = success_start
            else:
                success_count = 0
                success_start = None
        if t_half is not None and config.early_stop:
            break

    return TrainingResult(
        method="sgd",
        config=config,
        lambda_init=lambda_init,
        a_init=_logit(lambda_init),
        seed=int(seed),
        checkpoints=tuple(checkpoints),
        t_half=t_half,
        censored=t_half is None,
        steps_completed=steps_completed,
        final_a=float(a),
        final_lambda=float(lambda_from_a(a)),
    )


# ---------------------------------------------------------------------------
# End-to-end experiment driver and artifact assembly
# ---------------------------------------------------------------------------


def _float_key(value: float) -> str:
    """Filesystem/column-safe, deterministic rendering of a floating value."""
    return f"{value:g}".replace("-", "m").replace(".", "p").replace("+", "")


def _run_id(
    method: str,
    N: int,
    s: int,
    pi: float,
    lambda_init: float,
    seed: int | None,
) -> str:
    stem = (
        f"{method}__N{N}__s{s}__pi{_float_key(pi)}"
        f"__init{_float_key(lambda_init)}"
    )
    return stem if seed is None else f"{stem}__seed{seed}"


def _schedule_table(calibration: CalibrationTables) -> pd.DataFrame:
    """Collapse the probe-tidy calibration records to one row per schedule."""
    raw = pd.DataFrame(calibration.schedules)
    if raw.empty:
        return raw
    keys = ["schedule", "N", "theta", "w", "m_base", "s", "pi"]
    fixed = [
        "H",
        "log_H",
        "inverse_H",
        "predicted_half_time",
        "bayes_risk",
    ]
    out = raw.groupby(keys, as_index=False, sort=False)[fixed].first()
    for probe in sorted(raw.lambda_probe.unique()):
        part = raw[raw.lambda_probe == probe][
            keys
            + [
                "gradient",
                "log_abs_gradient",
                "excess_risk",
                "log_excess_risk",
            ]
        ].copy()
        suffix = _float_key(float(probe))
        part = part.rename(
            columns={
                "gradient": f"gradient_lambda_{suffix}",
                "log_abs_gradient": f"log_abs_gradient_lambda_{suffix}",
                "excess_risk": f"excess_risk_lambda_{suffix}",
                "log_excess_risk": f"log_excess_risk_lambda_{suffix}",
            }
        )
        out = out.merge(part, on=keys, how="left", validate="one_to_one")
    return out


def _evenly_spaced_rows(frame: pd.DataFrame, maximum: int) -> pd.DataFrame:
    """Keep endpoints and log-grid coverage without looking at run outcomes."""
    frame = frame.sort_values("pi")
    if len(frame) <= maximum:
        return frame
    raw_indices = np.linspace(0, len(frame) - 1, maximum)
    indices = sorted({int(round(value)) for value in raw_indices})
    # Rounding can merge middle indices; fill deterministically from the left.
    for index in range(len(frame)):
        if len(indices) >= maximum:
            break
        if index not in indices:
            indices.append(index)
    return frame.iloc[sorted(indices[:maximum])]


def _select_cells(
    schedules: pd.DataFrame,
    *,
    N_grid: Sequence[int],
    s_grid: Sequence[int],
    max_steps: int,
    checkpoint_every: int,
    blind_slow_factor: float = 10.0,
    observable_budget_fraction: float = 0.8,
    minimum_observable_checkpoints: int = 5,
    minimum_boost_scales: int = 3,
    maximum_primary_schedules: int = 4,
    maximum_stochastic_primary: int = 4,
    minimum_curvature_span: float = 10.0,
    minimum_update_ulps: float = 16.0,
    learning_rate: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the main cell using exact time scales, before any trajectories.

    The rule is intentionally simple and fully recorded:

    * use ``s=1`` as the primary non-full-mask intervention when available;
    * call the blind direction slow when its predicted half-time is at least
      ``blind_slow_factor * max_steps``;
    * call a boost observable when its predicted half-time lies between five
      checkpoint intervals and 80% of the common budget;
    * require those boosts to span a factor of ten in curvature and the blind
      update at both local probes to exceed 16 float64 ulps; and
    * select the largest eligible N (the highest-dimensional cell whose blind
      dynamics remain numerically resolvable).

    If no N passes, the deterministic fallback maximises the number of
    observable boosts, then the blind time-scale ratio, and finally prefers
    the smaller N.  No optimiser endpoint enters this decision.
    """
    out = schedules.copy()
    out["selected_N"] = False
    out["retained"] = False
    out["stochastic_retained"] = False
    out["role"] = "not_retained"
    out["reason"] = "candidate not selected by the predeclared calibration rule"

    s_values = tuple(int(s) for s in s_grid)
    non_full = [s for s in s_values if s > 0]
    primary_s = 1 if 1 in non_full else (non_full[0] if non_full else 0)
    min_time = float(minimum_observable_checkpoints * checkpoint_every)
    max_time = float(observable_budget_fraction * max_steps)

    diagnostics: list[dict[str, Any]] = []
    for N_raw in sorted(set(int(n) for n in N_grid)):
        N = int(N_raw)
        cell = out[out.N == N]
        blind = cell[cell.pi == 0.0]
        if blind.empty:
            raise ValueError(f"calibration contains no blind schedule for N={N}")
        blind_time = float(blind.predicted_half_time.iloc[0])
        primary = cell[(cell.s == primary_s) & (cell.pi > 0.0)]
        feasible = primary[
            (primary.predicted_half_time >= min_time)
            & (primary.predicted_half_time <= max_time)
        ]
        curvature_span = (
            float(feasible.H.max() / feasible.H.min()) if len(feasible) >= 2 else 1.0
        )
        blind_row = blind.iloc[0]
        update_ulps: list[float] = []
        for probe in (0.4, 0.6):
            column = f"gradient_lambda_{_float_key(probe)}"
            if column not in blind_row.index:
                continue
            ulp = abs(float(np.spacing(abs(_logit(probe)))))
            update_ulps.append(
                math.inf
                if ulp == 0.0
                else learning_rate * abs(float(blind_row[column])) / ulp
            )
        minimum_observed_ulps = min(update_ulps) if update_ulps else 0.0
        numerically_resolvable = minimum_observed_ulps >= minimum_update_ulps
        diagnostics.append(
            {
                "N": N,
                "primary_s": primary_s,
                "blind_predicted_half_time": blind_time,
                "blind_slow_threshold": blind_slow_factor * max_steps,
                "blind_is_slow": blind_time >= blind_slow_factor * max_steps,
                "observable_time_min": min_time,
                "observable_time_max": max_time,
                "n_observable_primary": int(len(feasible)),
                "observable_curvature_span": curvature_span,
                "minimum_blind_update_ulps": minimum_observed_ulps,
                "minimum_required_update_ulps": minimum_update_ulps,
                "numerically_resolvable": numerically_resolvable,
                "strictly_eligible": bool(
                    blind_time >= blind_slow_factor * max_steps
                    and len(feasible) >= minimum_boost_scales
                    and curvature_span >= minimum_curvature_span
                    and numerically_resolvable
                ),
            }
        )

    diagnostic_frame = pd.DataFrame(diagnostics)
    eligible = diagnostic_frame[diagnostic_frame.strictly_eligible]
    fallback_used = eligible.empty
    if not fallback_used:
        selected_N = int(eligible.sort_values("N").N.iloc[-1])
        selection_reason = (
            "largest numerically resolvable N with blind predicted half-time >= "
            f"{blind_slow_factor:g}x budget and >= {minimum_boost_scales} "
            f"observable s=1 boosts spanning >= {minimum_curvature_span:g}x curvature"
        )
    else:
        ranked = diagnostic_frame.sort_values(
            ["n_observable_primary", "blind_predicted_half_time", "N"],
            ascending=[False, False, True],
        )
        selected_N = int(ranked.N.iloc[0])
        selection_reason = (
            "deterministic fallback: maximise observable primary scales, then "
            "blind predicted time scale, then prefer smaller N"
        )

    out.loc[out.N == selected_N, "selected_N"] = True
    selected = out[out.N == selected_N]
    blind_index = selected[selected.pi == 0.0].index[0]
    out.loc[blind_index, ["retained", "stochastic_retained", "role", "reason"]] = [
        True,
        True,
        "blind",
        "selected blind baseline",
    ]

    primary = selected[(selected.s == primary_s) & (selected.pi > 0.0)]
    feasible = primary[
        (primary.predicted_half_time >= min_time)
        & (primary.predicted_half_time <= max_time)
    ]
    if feasible.empty:
        # Still produce a runnable diagnostic cell under a deliberately tiny
        # quick/test budget: choose the predicted time nearest half the budget.
        target = max(1.0, 0.5 * max_steps)
        distance = np.abs(np.log(primary.predicted_half_time / target))
        feasible = primary.loc[[distance.idxmin()]] if not primary.empty else primary
    retained_primary = _evenly_spaced_rows(feasible, maximum_primary_schedules)
    out.loc[retained_primary.index, "retained"] = True
    out.loc[retained_primary.index, "role"] = "primary_low_visibility"
    out.loc[retained_primary.index, "reason"] = (
        "primary non-full-mask boost retained from exact observable-time window"
    )

    stochastic_primary = _evenly_spaced_rows(
        retained_primary, maximum_stochastic_primary
    )
    out.loc[stochastic_primary.index, "stochastic_retained"] = True

    # Pick one common schedule mass for the boundary/robustness conditions.
    # It is the retained primary pi nearest the geometric centre of that grid,
    # with ties resolved upward.  This uses calibration, never run endpoints.
    if not retained_primary.empty:
        log_pis = np.log(retained_primary.pi.to_numpy(dtype=float))
        centre = float(np.mean(log_pis))
        choices = retained_primary.assign(
            _distance=np.abs(log_pis - centre)
        ).sort_values(
            ["_distance", "pi"], ascending=[True, False]
        )
        control_pi = float(choices.pi.iloc[0])
    else:
        control_pi = math.nan
    for control_s, role in ((0, "full_mask_control"), (4, "s4_robustness")):
        candidates = selected[(selected.s == control_s) & (selected.pi > 0.0)]
        if candidates.empty or control_s == primary_s:
            continue
        if math.isfinite(control_pi) and np.any(candidates.pi == control_pi):
            index = candidates[candidates.pi == control_pi].index[0]
        elif math.isfinite(control_pi):
            distance = np.abs(np.log(candidates.pi / control_pi))
            index = distance.idxmin()
        else:
            index = candidates.sort_values(["H", "pi"], ascending=False).index[0]
        out.loc[index, ["retained", "stochastic_retained", "role", "reason"]] = [
            True,
            True,
            role,
            "control matched to primary mass nearest its log-grid centre",
        ]

    diagnostic_frame["selected_N"] = diagnostic_frame.N == selected_N
    diagnostic_frame["fallback_used"] = fallback_used
    diagnostic_frame["selection_reason"] = selection_reason
    diagnostic_frame["max_steps"] = max_steps
    diagnostic_frame["checkpoint_every"] = checkpoint_every
    diagnostic_frame["minimum_boost_scales"] = minimum_boost_scales
    diagnostic_frame["minimum_curvature_span"] = minimum_curvature_span
    diagnostic_frame["primary_s"] = primary_s
    return out, diagnostic_frame


def _learning_rate_sweep(
    schedules: pd.DataFrame,
    *,
    theta: float,
    w: float,
    requested_learning_rate: float,
    batch_size: int,
    max_steps: int,
    checkpoint_every: int,
    validation_size: int,
    branch_T: float,
    kl_impl: str,
    sweep_inits: Sequence[float] = (0.1, 0.9),
    sweep_seeds: Sequence[int] = (91_731, 91_732),
) -> tuple[float, pd.DataFrame, str]:
    """Choose one common SGD rate by a predeclared stability-only rule.

    The sweep is run once on the globally highest-curvature candidate.  It
    never ranks candidates by recovery speed or final accuracy: a candidate
    either keeps all scalar states finite/bounded and avoids a material exact
    risk explosion, or it does not.  The largest stable candidate is chosen.
    """
    if not math.isfinite(requested_learning_rate) or requested_learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    eta_candidates = tuple(
        sorted(
            {
                requested_learning_rate / 4.0,
                requested_learning_rate / 2.0,
                requested_learning_rate,
            }
        )
    )
    highest = schedules.sort_values(
        ["H", "N", "s", "pi"], ascending=[False, True, True, False]
    ).iloc[0]
    sweep_steps = max(1, min(int(max_steps), 1_000))
    sweep_checkpoint = max(1, min(int(checkpoint_every), max(1, sweep_steps // 10)))
    rows: list[dict[str, Any]] = []

    for eta in eta_candidates:
        schedule = QLambdaSchedule(
            name=str(highest.schedule),
            m_base=int(highest.m_base),
            s=int(highest.s),
            pi=float(highest.pi),
        )
        cell_config = QLambdaConfig(
            N=int(highest.N),
            theta=theta,
            w=w,
            schedule=schedule,
            learning_rate=float(eta),
            batch_size=batch_size,
            max_steps=sweep_steps,
            checkpoint_every=sweep_checkpoint,
            validation_size=validation_size,
            consecutive_successes=1,
            early_stop=False,
            branch_T=branch_T,
            kl_impl=kl_impl,
        )
        for lambda_init in sweep_inits:
            for seed in sweep_seeds:
                result = train_mode_weight(
                    cell_config, float(lambda_init), int(seed)
                )
                risks = np.asarray(
                    [point.exact_excess_risk for point in result.checkpoints],
                    dtype=float,
                )
                parameters = np.asarray(
                    [point.a for point in result.checkpoints], dtype=float
                )
                initial_risk = float(risks[0])
                finite = bool(np.all(np.isfinite(risks)) and np.all(np.isfinite(parameters)))
                bounded = bool(np.all(np.abs(parameters) <= 20.0))
                risk_limit = max(initial_risk * 1.25, initial_risk + 1e-12)
                no_risk_explosion = bool(float(np.max(risks)) <= risk_limit)
                stable = finite and bounded and no_risk_explosion
                rows.append(
                    {
                        "requested_learning_rate": requested_learning_rate,
                        "candidate_learning_rate": eta,
                        "N": int(highest.N),
                        "m_base": int(highest.m_base),
                        "s": int(highest.s),
                        "pi": float(highest.pi),
                        "schedule": str(highest.schedule),
                        "H": float(highest.H),
                        "lambda_init": float(lambda_init),
                        "seed": int(seed),
                        "sweep_steps": sweep_steps,
                        "initial_exact_excess_risk": initial_risk,
                        "maximum_exact_excess_risk": float(np.max(risks)),
                        "final_exact_excess_risk": float(risks[-1]),
                        "maximum_abs_a": float(np.max(np.abs(parameters))),
                        "all_finite": finite,
                        "bounded_abs_a_le_20": bounded,
                        "no_risk_explosion_25pct": no_risk_explosion,
                        "stable": stable,
                    }
                )

    frame = pd.DataFrame(rows)
    by_eta = frame.groupby("candidate_learning_rate", sort=True).stable.all()
    stable_rates = by_eta[by_eta].index.to_numpy(dtype=float)
    if stable_rates.size == 0:
        raise RuntimeError(
            "no candidate learning rate passed the predeclared highest-curvature stability sweep"
        )
    chosen = float(np.max(stable_rates))
    frame["chosen"] = frame.candidate_learning_rate == chosen
    reason = (
        "largest candidate for which every fixed-seed/highest-curvature run "
        "remained finite, |a|<=20, and below the 25% exact-risk explosion bound"
    )
    frame["selection_rule"] = reason
    return chosen, frame, reason


def _km_median(durations: np.ndarray, events: np.ndarray) -> float:
    """Kaplan-Meier median; infinity means it is not identified by follow-up."""
    if durations.size == 0:
        return math.inf
    if np.all(events):
        return float(np.median(durations))
    survival = 1.0
    for time_value in np.unique(durations[events]):
        at_risk = int(np.sum(durations >= time_value))
        event_count = int(np.sum((durations == time_value) & events))
        if at_risk:
            survival *= 1.0 - event_count / at_risk
        if survival <= 0.5:
            return float(time_value)
    return math.inf


def _half_time_summary(
    group: pd.DataFrame,
    *,
    bootstrap_reps: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    events = ~group.censored.to_numpy(dtype=bool)
    event_times = group.t_half.to_numpy(dtype=float)
    censor_times = group.steps_completed.to_numpy(dtype=float)
    durations = np.where(events, event_times, censor_times)
    median = _km_median(durations, events)

    boot = np.empty(bootstrap_reps, dtype=float)
    for index in range(bootstrap_reps):
        draw = rng.integers(0, len(group), size=len(group))
        boot[index] = _km_median(durations[draw], events[draw])
    finite = boot[np.isfinite(boot)]
    undefined_fraction = float(np.mean(~np.isfinite(boot)))
    if finite.size:
        ci_low = float(np.quantile(finite, 0.025))
        # If more than the upper-tail alpha is unidentifiable, reporting a
        # finite upper limit would discard censoring information.
        ci_high = (
            math.inf
            if undefined_fraction > 0.025
            else float(np.quantile(finite, min(1.0, 0.975 / (1.0 - undefined_fraction))))
        )
    else:
        ci_low = math.nan
        ci_high = math.inf
    return {
        "n_runs": int(len(group)),
        "n_events": int(np.sum(events)),
        "n_censored": int(np.sum(~events)),
        "event_fraction": float(np.mean(events)),
        "median_t_half": median,
        "median_censored": not math.isfinite(median),
        "t_half_ci_low": ci_low,
        "t_half_ci_high": ci_high,
        "bootstrap_undefined_fraction": undefined_fraction,
        "half_time_estimator": "median" if np.all(events) else "Kaplan-Meier median",
    }


def _aggregate_runs(
    runs: pd.DataFrame,
    *,
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    if runs.empty:
        return pd.DataFrame()
    keys = [
        "method",
        "schedule",
        "role",
        "N",
        "theta",
        "w",
        "m_base",
        "s",
        "pi",
        "lambda_init",
        "H",
        "inverse_H",
        "predicted_half_time",
    ]
    rng = np.random.default_rng(bootstrap_seed)
    rows: list[dict[str, Any]] = []
    for key, group in runs.groupby(keys, dropna=False, sort=False):
        row = dict(zip(keys, key))
        row.update(_half_time_summary(group, bootstrap_reps=bootstrap_reps, rng=rng))
        row["median_over_predicted"] = (
            row["median_t_half"] / row["predicted_half_time"]
            if math.isfinite(row["median_t_half"])
            else math.inf
        )
        row["median_final_lambda"] = float(np.median(group.final_lambda))
        row["median_final_exact_excess_risk"] = float(
            np.median(group.final_exact_excess_risk)
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _results_markdown(
    schedules: pd.DataFrame,
    runs: pd.DataFrame,
    table: pd.DataFrame,
    selection: pd.DataFrame,
    *,
    run_sgd: bool,
    quick: bool,
    chosen_learning_rate: float,
    learning_rate_sweep: pd.DataFrame,
) -> str:
    selected = schedules[schedules.retained]
    blind = selected[selected.role == "blind"]
    nonblind = selected[selected.role != "blind"]
    boost_increases = bool(
        not blind.empty
        and not nonblind.empty
        and np.all(nonblind.H.to_numpy() > float(blind.H.iloc[0]))
    )
    identity_error = (
        float(np.nanmax(np.abs(runs.risk_identity_error)))
        if "risk_identity_error" in runs and not runs.empty
        else 0.0
    )
    pop = table[table.method == "population_gd"] if not table.empty else table
    pop_events = int(pop.n_events.sum()) if not pop.empty else 0
    pop_total = int(pop.n_runs.sum()) if not pop.empty else 0
    if run_sgd and not table.empty:
        stochastic = table[table.method == "sgd"]
        stochastic_events = int(stochastic.n_events.sum())
        stochastic_total = int(stochastic.n_runs.sum())
        stochastic_line = (
            f"- SGD: {stochastic_events}/{stochastic_total} runs had sustained "
            "half-error crossings; non-crossings remain right-censored."
        )
    else:
        stochastic_line = "- SGD was disabled for this run; no stochastic claim is made."
    selected_N = int(selection.loc[selection.selected_N, "N"].iloc[0])
    fallback = bool(selection.loc[selection.selected_N, "fallback_used"].iloc[0])
    selected_diagnostic = selection[selection.selected_N].iloc[0]
    chosen_sweep = learning_rate_sweep[learning_rate_sweep.chosen.astype(bool)]
    sweep_passed = bool(len(chosen_sweep) and chosen_sweep.stable.astype(bool).all())
    checks = {
        "exact-only cell selection passed its strict rule": bool(
            selected_diagnostic.strictly_eligible
        ),
        "a-coordinate risk identity agreed within 1e-10": identity_error <= 1e-10,
        "every retained low-visibility cell increased curvature": boost_increases,
        "a non-full-mask primary intervention was retained": bool(
            np.any((selected.role == "primary_low_visibility") & (selected.s > 0))
        ),
        "the common learning rate passed the highest-curvature stability sweep": (
            sweep_passed
        ),
    }
    failures = [label for label, passed in checks.items() if not passed]
    status = "SMOKE TEST — NOT REPORTABLE" if quick else (
        "PASS" if not failures else "REVIEW REQUIRED"
    )
    return "\n".join(
        [
            "# Direct mode-weight optimization results",
            "",
            "## Protocol status",
            "",
            f"- Overall status: **{status}**.",
            f"- Selected N={selected_N} from exact calibration only; fallback used: {fallback}.",
            f"- One common learning rate eta={chosen_learning_rate:g} was selected once and held fixed.",
            f"- Low-visibility schedules increased the exact a-coordinate curvature: {boost_increases}.",
            f"- Population GD: {pop_events}/{pop_total} runs reached half error within budget.",
            stochastic_line,
            f"- Maximum recorded population risk-identity discrepancy: {identity_error:.3e}.",
            "",
            "## Acceptance checks",
            "",
            *[
                f"- {'PASS' if passed else 'FAIL'} — {label}."
                for label, passed in checks.items()
            ],
            "",
            "## Failures and limitations",
            "",
            *(
                [f"- {failure}." for failure in failures]
                if failures
                else ["- No protocol check failed in this run."]
            ),
            *(
                ["- Quick mode thins calibration grids and uses reduced budgets; do not cite its numerical values."]
                if quick
                else []
            ),
            "",
            "## Supported interpretation",
            "",
            "The run isolates optimisation along the coherent mode-reweighting family "
            "q_lambda. High visibility supplies weak curvature, while retained "
            "low-visibility schedules supply more curvature and are compared through "
            "half-times versus inverse curvature. Censored cells are not replaced by "
            "their final step.",
            "",
            "This study does not claim that arbitrary masked neural networks learn "
            "wrong global frequencies, that non-full masks universally identify a joint "
            "law, or that fixed-time endpoint slopes test the recovery-radius theorem.",
            "",
        ]
    )


def run_mode_weight_optimization(
    cfg: RunConfig,
    *,
    results_dir: Path | None = None,
    theta: float = 0.8,
    w: float = 0.5,
    N_grid: Sequence[int] = DEFAULT_N_GRID,
    pi_grid: Sequence[float] = DEFAULT_PI_GRID,
    s_grid: Sequence[int] = DEFAULT_S_GRID,
    learning_rate: float = 0.1,
    batch_size: int = 512,
    max_steps: int = 100_000,
    checkpoint_every: int = 100,
    validation_size: int = 10_000,
    n_seeds: int = 5,
    run_sgd: bool = True,
    quick: bool = False,
    population_inits: Sequence[float] = (0.1, 0.3, 0.4, 0.6, 0.7, 0.9),
    stochastic_inits: Sequence[float] = (0.4, 0.6),
    consecutive_successes: int = 3,
    bootstrap_reps: int = 1_000,
    bootstrap_seed: int = 2027,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the complete direct mode-weight study and write semantic outputs.

    ``cfg`` supplies the repository-wide provenance stamp expected by
    :func:`mpi.tableio.write_table`; every study setting is additionally
    stored in ``mode_weight_config`` and an independent
    ``mode_weight_config_hash`` column.

    ``quick=True`` thins only the exact plotting grids. The caller supplies
    the smoke-test budgets explicitly, so a command-line value is never
    silently capped. Quick outputs are marked non-reportable.
    """
    if not isinstance(cfg, RunConfig):
        raise TypeError("cfg must be a RunConfig")
    output_dir = Path(results_dir or RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    N_values = tuple(int(value) for value in N_grid)
    pi_values = tuple(float(value) for value in pi_grid)
    s_values = tuple(int(value) for value in s_grid)
    pop_inits = tuple(float(value) for value in population_inits)
    sgd_inits = tuple(float(value) for value in stochastic_inits)
    if not N_values or not pi_values or not s_values:
        raise ValueError("N_grid, pi_grid, and s_grid must be non-empty")
    if 0.0 not in pi_values:
        raise ValueError("pi_grid must contain 0 for the blind baseline")
    if not isinstance(n_seeds, (int, np.integer)) or n_seeds <= 0:
        raise ValueError("n_seeds must be a positive integer")
    if not isinstance(bootstrap_reps, (int, np.integer)) or bootstrap_reps <= 0:
        raise ValueError("bootstrap_reps must be a positive integer")
    for name, value in (
        ("batch_size", batch_size),
        ("max_steps", max_steps),
        ("checkpoint_every", checkpoint_every),
        ("validation_size", validation_size),
        ("consecutive_successes", consecutive_successes),
    ):
        if not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(learning_rate) or not 0.0 < learning_rate < 8.0:
        raise ValueError(
            "learning_rate must lie in (0, 8), the global fixed-step stability range"
        )

    if not pop_inits:
        raise ValueError("population_inits must be non-empty")
    if run_sgd and not sgd_inits:
        raise ValueError("stochastic_inits must be non-empty when SGD is enabled")
    for value in pop_inits + sgd_inits:
        _validate_lambda_init(value)

    effective_steps = int(max_steps)
    effective_batch = int(batch_size)
    effective_validation = int(validation_size)
    effective_seeds = int(n_seeds)
    effective_checkpoint = int(checkpoint_every)
    effective_pop_inits = pop_inits
    risk_grid = np.linspace(0.05, 0.95, 41) if quick else np.linspace(0.01, 0.99, 197)
    visibility_grids = (
        {
            N: tuple(sorted({0, 1, min(4, N), N // 2, N}))
            for N in N_values
        }
        if quick
        else None
    )

    started = time.time()
    if verbose:
        mode = "quick smoke test" if quick else "full protocol"
        print(f"  direct mode-weight optimization: exact calibration ({mode})")
    calibration = calibrate_schedule_grid(
        theta=theta,
        w=w,
        N_grid=N_values,
        pi_grid=pi_values,
        s_grid=s_values,
        lambda_probes=(0.4, 0.6),
        lambda_grid=risk_grid,
        visibility_grids=visibility_grids,
        learning_rate=learning_rate,
        branch_T=cfg.branch_T,
        kl_impl=cfg.kl_impl,
    )
    calibration_df = pd.DataFrame(calibration.visibility)
    calibration_df.insert(0, "kind", "visibility")
    profiles_df = pd.DataFrame(calibration.risk_profiles).rename(
        columns={"lambda_value": "lambda"}
    )
    schedules_df = _schedule_table(calibration)
    if run_sgd:
        chosen_learning_rate, rate_sweep_df, rate_sweep_reason = _learning_rate_sweep(
            schedules_df,
            theta=theta,
            w=w,
            requested_learning_rate=learning_rate,
            batch_size=effective_batch,
            max_steps=effective_steps,
            checkpoint_every=effective_checkpoint,
            validation_size=effective_validation,
            branch_T=cfg.branch_T,
            kl_impl=cfg.kl_impl,
        )
    else:
        chosen_learning_rate = float(learning_rate)
        rate_sweep_reason = "not run because stochastic optimization was disabled"
        rate_sweep_df = pd.DataFrame(
            [
                {
                    "requested_learning_rate": learning_rate,
                    "candidate_learning_rate": learning_rate,
                    "chosen": True,
                    "stable": True,
                    "status": "not_run_population_only",
                    "selection_rule": rate_sweep_reason,
                }
            ]
        )

    # Curvature H is learning-rate independent, but the predicted discrete-GD
    # half-time is not. Recompute it with the one rate chosen before any main
    # trajectory is inspected, then hold that rate fixed in every cell.
    schedules_df["requested_learning_rate"] = float(learning_rate)
    schedules_df["learning_rate"] = chosen_learning_rate
    schedules_df["predicted_half_time"] = (
        math.log(2.0) / (chosen_learning_rate * schedules_df["H"])
    )
    schedules_df, selection_df = _select_cells(
        schedules_df,
        N_grid=N_values,
        s_grid=s_values,
        max_steps=effective_steps,
        checkpoint_every=effective_checkpoint,
        learning_rate=chosen_learning_rate,
    )
    selected_N = int(selection_df.loc[selection_df.selected_N, "N"].iloc[0])

    settings: dict[str, Any] = {
        "study": "direct_mode_weight_optimization",
        "reportable": not quick,
        "quick": quick,
        "theta": float(theta),
        "w": float(w),
        "N_grid": list(N_values),
        "pi_grid": list(pi_values),
        "s_grid": list(s_values),
        "m_base_rule": "floor(N/2)",
        "requested_learning_rate": float(learning_rate),
        "learning_rate_candidates": sorted(
            set(rate_sweep_df["candidate_learning_rate"].astype(float))
        ),
        "chosen_learning_rate": chosen_learning_rate,
        "learning_rate_selection_rule": rate_sweep_reason,
        "batch_size": effective_batch,
        "max_steps": effective_steps,
        "checkpoint_every": effective_checkpoint,
        "validation_size": effective_validation,
        "n_seeds": effective_seeds,
        "run_sgd": run_sgd,
        "population_inits": list(effective_pop_inits),
        "stochastic_inits": list(sgd_inits),
        "consecutive_successes": int(consecutive_successes),
        "bootstrap_reps": int(bootstrap_reps),
        "bootstrap_seed": int(bootstrap_seed),
        "selection_rule": {
            "primary_s": 1 if 1 in s_values else "first positive s",
            "blind_slow_factor": 10.0,
            "observable_budget_fraction": 0.8,
            "minimum_observable_checkpoints": 5,
            "minimum_boost_scales": 3,
            "minimum_curvature_span": 10.0,
            "minimum_blind_update_ulps": 16.0,
            "N_tiebreak": "largest numerically resolvable eligible N",
        },
        "lambda_grid": list(map(float, risk_grid)),
        "branch_T": cfg.branch_T,
        "kl_impl": cfg.kl_impl,
    }
    settings_blob = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    study_hash = hashlib.sha256(settings_blob.encode("utf-8")).hexdigest()[:16]
    settings["mode_weight_config_hash"] = study_hash

    trajectory_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []

    def execute(
        schedule_row: Any,
        method: Literal["population_gd", "sgd"],
        lambda_init: float,
        seed: int | None,
    ) -> None:
        schedule = QLambdaSchedule(
            name=str(schedule_row.schedule),
            m_base=int(schedule_row.m_base),
            s=int(schedule_row.s),
            pi=float(schedule_row.pi),
        )
        cell_config = QLambdaConfig(
            N=int(schedule_row.N),
            theta=theta,
            w=w,
            schedule=schedule,
            learning_rate=chosen_learning_rate,
            batch_size=effective_batch,
            max_steps=effective_steps,
            checkpoint_every=effective_checkpoint,
            validation_size=effective_validation,
            consecutive_successes=consecutive_successes,
            # Stochastic successes may stop only after their sustained crossing;
            # population trajectories retain the common full budget.
            early_stop=(method == "sgd"),
            branch_T=cfg.branch_T,
            kl_impl=cfg.kl_impl,
        )
        if method == "population_gd":
            result = population_gradient_descent(cell_config, lambda_init)
        else:
            if seed is None:
                raise ValueError("SGD run requires a seed")
            result = train_mode_weight(cell_config, lambda_init, seed)
        identifier = _run_id(
            method,
            cell_config.N,
            schedule.s,
            schedule.pi,
            lambda_init,
            seed,
        )
        for record in result.checkpoint_records():
            record["run_id"] = identifier
            record["role"] = str(schedule_row.role)
            record["lambda_hat"] = record["lambda_value"]
            record["H"] = float(schedule_row.H)
            record["inverse_H"] = float(schedule_row.inverse_H)
            record["predicted_half_time"] = float(schedule_row.predicted_half_time)
            record["selected_N"] = True
            trajectory_rows.append(record)
        summary = result.summary_record()
        final_metrics = exact_schedule_metrics_from_a(cell_config, result.final_a)
        summary.update(
            {
                "run_id": identifier,
                "role": str(schedule_row.role),
                "H": float(schedule_row.H),
                "log_H": float(schedule_row.log_H),
                "inverse_H": float(schedule_row.inverse_H),
                "predicted_half_time": float(schedule_row.predicted_half_time),
                "risk_identity_error": final_metrics.risk_identity_error,
                "selected_N": True,
            }
        )
        run_rows.append(summary)

    retained = schedules_df[schedules_df.retained & (schedules_df.N == selected_N)]
    if verbose:
        print(
            f"  selected N={selected_N}; {len(retained)} population schedules, "
            f"{int(retained.stochastic_retained.sum()) if run_sgd else 0} SGD schedules"
        )
    for schedule_row in retained.itertuples(index=False):
        for lambda_init in effective_pop_inits:
            execute(schedule_row, "population_gd", lambda_init, None)

    if run_sgd:
        stochastic = retained[retained.stochastic_retained]
        for schedule_row in stochastic.itertuples(index=False):
            for lambda_init in sgd_inits:
                for seed in range(effective_seeds):
                    execute(schedule_row, "sgd", lambda_init, seed)

    trajectories_df = pd.DataFrame(trajectory_rows)
    runs_df = pd.DataFrame(run_rows)
    table_df = _aggregate_runs(
        runs_df,
        bootstrap_reps=int(bootstrap_reps),
        bootstrap_seed=int(bootstrap_seed),
    )

    for frame in (
        calibration_df,
        schedules_df,
        profiles_df,
        selection_df,
        trajectories_df,
        runs_df,
        table_df,
        rate_sweep_df,
    ):
        frame["mode_weight_config_hash"] = study_hash
        frame["reportable"] = not quick
        frame["chosen_learning_rate"] = chosen_learning_rate

    config_row = {
        **{
            key: json.dumps(value, sort_keys=True)
            if isinstance(value, (list, dict))
            else value
            for key, value in settings.items()
        },
        **provenance(cfg),
        "elapsed_seconds": time.time() - started,
        "platform_full": platform.platform(),
        "python_full": sys.version,
    }
    config_df = pd.DataFrame([config_row])

    artifacts = {
        "visibility_calibration": calibration_df,
        "schedule_calibration": schedules_df,
        "risk_profiles": profiles_df,
        "cell_selection": selection_df,
        "learning_rate_sweep": rate_sweep_df,
        "trajectories": trajectories_df,
        "run_summaries": runs_df,
        "half_times": table_df,
        "config": config_df,
    }
    for key, frame in artifacts.items():
        write_table(frame, f"mode_weight_{key}", cfg, output_dir)

    config_payload = {**settings, "provenance": provenance(cfg)}
    (output_dir / "mode_weight_config.json").write_text(
        json.dumps(config_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    results_text = _results_markdown(
        schedules_df,
        runs_df,
        table_df,
        selection_df,
        run_sgd=run_sgd,
        quick=quick,
        chosen_learning_rate=chosen_learning_rate,
        learning_rate_sweep=rate_sweep_df,
    )
    (output_dir / "mode_weight_RESULTS.md").write_text(results_text, encoding="utf-8")

    if verbose:
        print(
            f"  direct mode-weight optimization done in {time.time()-started:.1f}s "
            f"({len(runs_df)} runs, {len(trajectories_df)} checkpoints)"
        )
    return artifacts
