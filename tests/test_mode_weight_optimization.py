"""Validation tests for direct optimisation in the q_lambda family.

These tests deliberately use two independent references:

* :mod:`mpi.brute` enumerates every state for the posterior and masked-block
  likelihood checks; and
* the established log-domain estimands supply the population D and U values.

The production study may use sufficient statistics, but none of the
small-N brute-force checks below relies on that reduction.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mpi.brute import build_brute_law
from mpi.config import LawSpec, RunConfig
from mpi.core import binary_kl, visible_law
from mpi.mode_weight_optimization import (
    QLambdaConfig,
    QLambdaSchedule,
    SampleBatch,
    base_posterior_logit,
    exact_schedule_metrics,
    mode_weight_loss,
    paired_excess_risk,
    population_gradient_descent,
    run_mode_weight_optimization,
    sample_schedule_batch,
    tilted_posterior_logit,
    train_mode_weight,
)
from mpi.estimands import log_D, log_U
from mpi.tableio import read_table


LAW_P = LawSpec("product", 0.8, True, "")


def _logit(p: float) -> float:
    return math.log(p / (1.0 - p))


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    x = np.asarray(x, dtype=float)
    ans = np.exp(-np.logaddexp(0.0, -x))
    return float(ans) if ans.ndim == 0 else ans


def _config(
    *,
    N: int = 9,
    theta: float = LAW_P.param,
    w: float = 0.5,
    m_base: int = 4,
    s: int = 1,
    pi: float = 0.25,
    batch_size: int = 128,
    max_steps: int = 20,
) -> QLambdaConfig:
    schedule = QLambdaSchedule("test", m_base=m_base, s=s, pi=pi)
    return QLambdaConfig(
        N=N,
        theta=theta,
        w=w,
        schedule=schedule,
        learning_rate=0.2,
        batch_size=batch_size,
        max_steps=max_steps,
        checkpoint_every=1,
        validation_size=256,
        consecutive_successes=1,
    )


def _batch(m: np.ndarray, r: np.ndarray, z: np.ndarray, base: np.ndarray) -> SampleBatch:
    """Construct a batch while keeping all validation arithmetic in float64."""
    return SampleBatch(
        m=np.asarray(m, dtype=np.int64),
        r=np.asarray(r, dtype=np.int64),
        z=np.asarray(z, dtype=np.float64),
        base_logit=np.asarray(base, dtype=np.float64),
    )


def _loss_array(a: float, batch: SampleBatch, w: float) -> np.ndarray:
    return np.asarray(mode_weight_loss(a, batch, w=w, reduction="none"), dtype=float)


def _direct_schedule_D(cfg: QLambdaConfig, lam: float) -> float:
    spec = LawSpec("product", cfg.theta, True, "")
    base = visible_law(spec, cfg.N, cfg.schedule.m_base, cfg.w)
    boost = visible_law(spec, cfg.N, cfg.schedule.s, cfg.w)
    return (
        (1.0 - cfg.schedule.pi) * math.exp(log_D(base, lam))
        + cfg.schedule.pi * math.exp(log_D(boost, lam))
    )


def _direct_schedule_U(cfg: QLambdaConfig) -> float:
    spec = LawSpec("product", cfg.theta, True, "")
    base = visible_law(spec, cfg.N, cfg.schedule.m_base, cfg.w)
    boost = visible_law(spec, cfg.N, cfg.schedule.s, cfg.w)
    return (
        (1.0 - cfg.schedule.pi) * math.exp(log_U(base))
        + cfg.schedule.pi * math.exp(log_U(boost))
    )


@pytest.mark.parametrize("w,lam", [(0.5, 0.31), (0.67, 0.42)])
def test_posterior_tilt_matches_direct_enumeration(w: float, lam: float) -> None:
    """The shared logit shift agrees with Bayes under explicit q_lambda."""
    N = 7
    brute = build_brute_law(LAW_P, N, w)

    for m in (0, 1, 3):
        V = tuple(range(m))
        plus_v, _ = brute.marginal(brute.p_plus, V)
        minus_v, _ = brute.marginal(brute.p_minus, V)
        beta_p = w * plus_v / (w * plus_v + (1.0 - w) * minus_v)
        beta_q = lam * plus_v / (lam * plus_v + (1.0 - lam) * minus_v)

        keys = brute._visible_keys(V)
        uniq = np.unique(keys)
        if m:
            bits = (uniq[:, None] >> np.arange(m)) & 1
            r = 2 * bits.sum(axis=1) - m
        else:
            r = np.zeros(1, dtype=int)
        m_vec = np.full(r.shape, m, dtype=int)

        base = np.asarray(base_posterior_logit(N, LAW_P.param, w, m_vec, r))
        tilted = np.asarray(tilted_posterior_logit(base, _logit(lam), w))
        np.testing.assert_allclose(_sigmoid(base), beta_p, rtol=0.0, atol=2e-13)
        np.testing.assert_allclose(_sigmoid(tilted), beta_q, rtol=0.0, atol=2e-13)


def test_binary_loss_difference_is_full_conditional_likelihood_difference() -> None:
    """Mode BCE is exactly the lambda-dependent masked-block NLL term."""
    N, m, w, lam = 7, 3, 0.57, 0.29
    brute = build_brute_law(LAW_P, N, w)
    V = tuple(range(m))
    keys = brute._visible_keys(V)
    p_v, inv = brute.marginal(brute.p, V)
    q = brute.q(lam)
    q_v, _ = brute.marginal(q, V)

    r = brute.states[:, :m].sum(axis=1)
    z = (brute.states.sum(axis=1) > 0).astype(float)
    m_vec = np.full(brute.states.shape[0], m, dtype=int)
    base = np.asarray(base_posterior_logit(N, LAW_P.param, w, m_vec, r))
    batch = _batch(m_vec, r, z, base)

    bce_difference = (
        _loss_array(_logit(lam), batch, w)
        - _loss_array(_logit(w), batch, w)
    )
    full_nll_difference = (
        np.log(brute.p) - np.log(p_v[inv])
        - np.log(q) + np.log(q_v[inv])
    )
    np.testing.assert_allclose(
        bce_difference, full_nll_difference, rtol=0.0, atol=3e-13
    )


def test_exact_bce_excess_risk_equals_masked_discrepancy() -> None:
    cfg = _config(w=0.61, m_base=5, s=2, pi=0.3)
    lam = 0.36
    metrics = exact_schedule_metrics(cfg, lam)
    direct = _direct_schedule_D(cfg, lam)

    assert metrics.risk - metrics.bayes_risk == pytest.approx(direct, abs=2e-13)
    assert metrics.excess_risk == pytest.approx(direct, abs=2e-13)
    # The estimate is the scalar parameter itself, not a posterior queried at
    # m=0 (or any other visibility).
    assert metrics.lambda_value == pytest.approx(lam, rel=0.0, abs=2e-16)
    assert float(_sigmoid(metrics.a)) == pytest.approx(lam, rel=0.0, abs=2e-16)


def test_loss_and_population_gradients_match_centered_finite_differences() -> None:
    cfg = _config(w=0.58, m_base=4, s=1, pi=0.2)
    lam = 0.37
    a = _logit(lam)
    h = 2e-5

    # Per-example analytic identity d ell / da = beta_lambda - z.
    base = np.array([-3.0, -0.2, 1.1, 4.0])
    z = np.array([0.0, 1.0, 0.0, 1.0])
    batch = _batch(np.ones(4), np.array([-1, 1, -1, 1]), z, base)
    fd_each = (
        _loss_array(a + h, batch, cfg.w) - _loss_array(a - h, batch, cfg.w)
    ) / (2.0 * h)
    analytic_each = _sigmoid(tilted_posterior_logit(base, a, cfg.w)) - z
    np.testing.assert_allclose(fd_each, analytic_each, rtol=2e-10, atol=2e-10)

    # Population gradient is checked against a centered difference of R(a).
    def risk_at(a_value: float) -> float:
        return exact_schedule_metrics(cfg, _sigmoid(a_value)).risk

    finite_difference = (risk_at(a + h) - risk_at(a - h)) / (2.0 * h)
    assert exact_schedule_metrics(cfg, lam).gradient == pytest.approx(
        finite_difference, rel=2e-8, abs=2e-11
    )


def test_a_curvature_at_truth_is_schedule_residual_mmse() -> None:
    """This specifically guards against reusing d²D/dlambda² as d²R/da²."""
    cfg = _config(w=0.63, m_base=5, s=1, pi=0.17)
    expected_U = _direct_schedule_U(cfg)
    metrics = exact_schedule_metrics(cfg, cfg.w)
    assert metrics.curvature == pytest.approx(expected_U, rel=2e-13, abs=2e-15)

    # Independent numerical Hessian in the trained coordinate a.
    a_star, h = _logit(cfg.w), 2e-4
    g_hi = exact_schedule_metrics(cfg, _sigmoid(a_star + h)).gradient
    g_lo = exact_schedule_metrics(cfg, _sigmoid(a_star - h)).gradient
    assert (g_hi - g_lo) / (2.0 * h) == pytest.approx(expected_U, rel=2e-8)


@pytest.mark.parametrize("pi", [0.0, 1e-3, 0.2, 1.0])
def test_schedule_curvature_is_linear_in_schedule_mass(pi: float) -> None:
    cfg = _config(w=0.5, m_base=5, s=1, pi=pi)
    got = exact_schedule_metrics(cfg, cfg.w).curvature
    u_base = math.exp(log_U(visible_law(LAW_P, cfg.N, cfg.schedule.m_base, cfg.w)))
    u_low = math.exp(log_U(visible_law(LAW_P, cfg.N, cfg.schedule.s, cfg.w)))
    assert got == pytest.approx((1.0 - pi) * u_base + pi * u_low, abs=2e-15)


@pytest.mark.parametrize("w,lam", [(0.5, 0.24), (0.71, 0.43)])
def test_full_mask_endpoint(w: float, lam: float) -> None:
    cfg = _config(w=w, m_base=0, s=0, pi=0.0)
    base = base_posterior_logit(cfg.N, cfg.theta, w, 0, 0)
    tilted = tilted_posterior_logit(base, _logit(lam), w)
    assert float(_sigmoid(base)) == pytest.approx(w, abs=2e-15)
    assert float(_sigmoid(tilted)) == pytest.approx(lam, abs=2e-15)
    assert exact_schedule_metrics(cfg, lam).excess_risk == pytest.approx(
        binary_kl(w, lam), abs=2e-14
    )


def test_exact_sampler_matches_mode_conditioned_law_and_gradient() -> None:
    """Audit z, schedule, visible-statistic frequencies, and minibatch gradient."""
    # Moderate theta makes conditioning on the global sign material, so this
    # rejects the legacy unconditioned product-binomial shortcut.
    cfg = _config(
        N=7, theta=0.35, w=0.59, m_base=3, s=1, pi=0.35, batch_size=60_000
    )
    batch = sample_schedule_batch(cfg, np.random.default_rng(8128))
    m = np.asarray(batch.m)
    r = np.asarray(batch.r)
    z = np.asarray(batch.z)

    assert z.mean() == pytest.approx(cfg.w, abs=0.008)
    assert np.mean(m == cfg.schedule.s) == pytest.approx(cfg.schedule.pi, abs=0.008)
    recomputed = base_posterior_logit(cfg.N, cfg.theta, cfg.w, m, r)
    np.testing.assert_allclose(batch.base_logit, recomputed, rtol=0.0, atol=0.0)

    # Total variation between the sampled and exact joint laws of (m,z,r).
    observed: dict[tuple[int, int, int], float] = {}
    for mi, zi, ri in zip(m, z.astype(int), r, strict=True):
        key = (int(mi), int(zi), int(ri))
        observed[key] = observed.get(key, 0.0) + 1.0 / len(m)

    expected: dict[tuple[int, int, int], float] = {}
    for mi, p_m in (
        (cfg.schedule.m_base, 1.0 - cfg.schedule.pi),
        (cfg.schedule.s, cfg.schedule.pi),
    ):
        vl = visible_law(LawSpec("product", cfg.theta, True, ""), cfg.N, mi, cfg.w)
        for zi, p_z, log_h in (
            (1, cfg.w, vl.log_h_plus),
            (0, 1.0 - cfg.w, vl.log_h_minus),
        ):
            for ri, p_r in zip(vl.r.astype(int), np.exp(log_h), strict=True):
                expected[(mi, zi, int(ri))] = p_m * p_z * float(p_r)

    support = set(expected) | set(observed)
    total_variation = 0.5 * sum(
        abs(observed.get(k, 0.0) - expected.get(k, 0.0)) for k in support
    )
    assert total_variation < 0.015

    lam = 0.38
    a = _logit(lam)
    sample_gradient = np.mean(
        _sigmoid(tilted_posterior_logit(batch.base_logit, a, cfg.w)) - z
    )
    exact_gradient = exact_schedule_metrics(cfg, lam).gradient
    assert sample_gradient == pytest.approx(exact_gradient, abs=0.008)

    # The paired audit must be the common-example loss difference.
    direct_paired = np.mean(
        _loss_array(a, batch, cfg.w) - _loss_array(_logit(cfg.w), batch, cfg.w)
    )
    paired_mean, paired_se = paired_excess_risk(a, batch, cfg.w)
    assert paired_mean == pytest.approx(direct_paired, abs=1e-15)
    assert paired_se >= 0.0


def test_population_quantities_are_reflection_symmetric_at_half_weight() -> None:
    cfg = _config(w=0.5, m_base=5, s=1, pi=0.13)
    left = exact_schedule_metrics(cfg, 0.27)
    right = exact_schedule_metrics(cfg, 0.73)
    assert left.risk == pytest.approx(right.risk, rel=2e-14, abs=2e-15)
    assert left.excess_risk == pytest.approx(right.excess_risk, rel=2e-14, abs=2e-15)
    assert left.gradient == pytest.approx(-right.gradient, rel=2e-13, abs=2e-15)
    assert left.curvature == pytest.approx(right.curvature, rel=2e-13, abs=2e-15)


def test_population_runs_reflect_across_half_weight() -> None:
    """Deterministic trajectories from lambda_0 and 1-lambda_0 reflect exactly."""
    cfg = _config(w=0.5, m_base=4, s=1, pi=0.2, max_steps=24)
    left = population_gradient_descent(cfg, 0.3)
    right = population_gradient_descent(cfg, 0.7)

    assert [point.step for point in left.trajectory] == [
        point.step for point in right.trajectory
    ]
    assert left.t_half == right.t_half
    for point_left, point_right in zip(left.trajectory, right.trajectory, strict=True):
        assert point_left.lambda_value == pytest.approx(
            1.0 - point_right.lambda_value, abs=3e-15
        )
        assert point_left.a == pytest.approx(-point_right.a, abs=3e-14)
        assert point_left.exact_gradient == pytest.approx(
            -point_right.exact_gradient, abs=3e-15
        )


def test_reported_estimate_is_always_sigmoid_of_the_only_parameter() -> None:
    """Guard specifically against restoring an empty-context model readout."""
    cfg = _config(w=0.57, m_base=5, s=2, pi=0.0, max_steps=7)
    result = population_gradient_descent(cfg, 0.31)

    for point in result.trajectory:
        expected = float(_sigmoid(point.a))
        assert point.lambda_value == pytest.approx(expected, abs=2e-16)
        assert point.lambda_hat == point.lambda_value
    assert result.final_lambda == pytest.approx(float(_sigmoid(result.final_a)), abs=2e-16)
    assert result.summary_record()["final_lambda"] == result.final_lambda


def test_fixed_seed_reproduces_stochastic_trajectory_and_summary() -> None:
    cfg = _config(
        N=7,
        theta=0.35,
        w=0.5,
        m_base=3,
        s=1,
        pi=0.25,
        batch_size=64,
        max_steps=12,
    )
    first = train_mode_weight(cfg, lambda_init=0.34, seed=2027)
    second = train_mode_weight(cfg, lambda_init=0.34, seed=2027)

    assert first.summary_record() == second.summary_record()
    assert len(first.trajectory) == len(second.trajectory)
    fields = tuple(first.trajectory[0].__dataclass_fields__)
    for name in fields:
        values_first = np.asarray([getattr(point, name) for point in first.trajectory])
        values_second = np.asarray([getattr(point, name) for point in second.trajectory])
        np.testing.assert_equal(values_first, values_second)


def test_named_runner_writes_only_semantic_artifacts(tmp_path) -> None:
    artifacts = run_mode_weight_optimization(
        RunConfig(laws=("product-0.8",), N_grid=(7,), w_grid=(0.5,)),
        results_dir=tmp_path,
        N_grid=(7,),
        pi_grid=(0.0, 0.1),
        s_grid=(0, 1, 4),
        batch_size=32,
        max_steps=8,
        checkpoint_every=2,
        validation_size=64,
        n_seeds=1,
        quick=True,
        population_inits=(0.4, 0.6),
        stochastic_inits=(0.4,),
        consecutive_successes=1,
        bootstrap_reps=10,
        verbose=False,
    )

    expected = {
        "visibility_calibration",
        "schedule_calibration",
        "risk_profiles",
        "cell_selection",
        "learning_rate_sweep",
        "trajectories",
        "run_summaries",
        "half_times",
        "config",
    }
    assert set(artifacts) == expected
    for suffix in expected:
        assert len(read_table(f"mode_weight_{suffix}", tmp_path)) > 0
    assert (tmp_path / "mode_weight_config.json").exists()
    assert (tmp_path / "mode_weight_RESULTS.md").exists()
    assert not list(tmp_path.glob("e3q_*"))

    sweep = artifacts["learning_rate_sweep"]
    chosen = sweep[sweep.chosen]
    assert len(chosen) > 0
    assert chosen.stable.all()
    assert chosen.candidate_learning_rate.nunique() == 1

    trajectory = artifacts["trajectories"]
    np.testing.assert_allclose(
        trajectory.lambda_hat,
        _sigmoid(trajectory.a.to_numpy()),
        rtol=0.0,
        atol=2e-16,
    )
