"""Unit tests T1-T12 of the implementation specification (section 5).

Run these *before any figure*::

    pytest tests -q

T11 and T12 are the tests that make Law P worth using: they convert the
closed-form proposition of App. `both-assumptions` into executable assertions.
A T7 or T12 failure is not a tolerance issue -- it is a bug or a false theorem.

Slow legs (mpmath at 50 digits, the full run matrix) are marked ``slow`` and
skipped by default; enable them with ``pytest -m slow``.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from mpi.brute import build_brute_law
from mpi.config import (
    LawSpec,
    ProductConstants,
    RunConfig,
    band_constants,
    lambda_grid_I,
    mode_weight_interval,
    visible_sizes,
)
from mpi.core import (
    binary_kl,
    log_h_mode,
    log_kl_branched,
    log_kl_stable,
    visible_law,
)
from mpi.e1 import run_e1
from mpi.e2 import run_e2
from mpi.e4 import log_second_moment, raw_loss_moments, sampling_cost
from mpi.estimands import (
    curvature_exact,
    log_D,
    log_D_grid,
    log_Delta,
    log_flows,
    log_mix,
    log_U,
    rho_grid,
)
from mpi.fits import curvature_richardson, quadratic_curvature_fit
from mpi.highprec import log_D_mp, log_U_mp

# --------------------------------------------------------------------------
# Test grids -- small enough to keep the default suite under a minute
# --------------------------------------------------------------------------

LAWS = [
    LawSpec("product", 0.6, True, ""),
    LawSpec("product", 0.8, True, ""),
    LawSpec("cw", 1.5, True, ""),
    LawSpec("cw", 0.5, False, ""),
]
PRODUCT_LAWS = [s for s in LAWS if s.law == "product"]
N_TEST = [127, 255]
W_TEST = [0.5, 0.9]


def _cells(N_list=N_TEST, w_list=W_TEST, laws=LAWS):
    for spec, N, w in itertools.product(laws, N_list, w_list):
        for m in visible_sizes(N):
            yield spec, N, int(m), w


# ==========================================================================
# T1 -- spin-flip symmetry
# ==========================================================================


@pytest.mark.parametrize("spec", LAWS)
@pytest.mark.parametrize("N", N_TEST)
def test_T1_spin_flip_symmetry(spec, N):
    """``max_r |log h^-(r) - log h^+(-r)| < 1e-10``.

    ``h^-`` is enumerated from scratch (own weight function, own restriction to
    ``X_-``), not obtained by reflecting ``h^+``, so this is a real check.
    """
    for m in visible_sizes(N):
        hp = log_h_mode(spec, N, int(m), +1)
        hm = log_h_mode(spec, N, int(m), -1)
        ref = hp[::-1]
        both_inf = np.isneginf(ref) & np.isneginf(hm)
        assert np.array_equal(np.isneginf(ref), np.isneginf(hm))
        finite = ~both_inf
        err = np.abs(hm[finite] - ref[finite]) if finite.any() else np.zeros(1)
        assert err.max() < 1e-10, (spec.key, N, m, err.max())


# ==========================================================================
# T2 -- posterior-shift lemma
# ==========================================================================


@pytest.mark.parametrize("spec", LAWS)
@pytest.mark.parametrize("w", W_TEST)
@pytest.mark.parametrize("lam", [0.25, 0.4, 0.62, 0.75])
def test_T2_posterior_shift(spec, w, lam):
    """``logit b_{V,lambda} = logit(lambda) + ell(r)`` -- Lem. `logit-shift`(ii).

    The reference side is a *direct Bayes* computation on the explicit state
    space, so this validates the identity rather than restating it.
    """
    N = 9
    bl = build_brute_law(spec, N, w)
    for m in range(0, N):
        V = tuple(range(m))
        vl = visible_law(spec, N, m, w)

        pv_plus, _ = bl.marginal(bl.p_plus, V)
        pv_minus, _ = bl.marginal(bl.p_minus, V)
        num = lam * pv_plus
        den = num + (1.0 - lam) * pv_minus
        ok = (pv_plus > 0) & (pv_minus > 0)
        logit_direct = np.log(num[ok] / (den[ok] - num[ok]))

        # brute-force visible patterns -> visible magnetisation r -> index a
        keys = bl._visible_keys(V)
        uniq = np.unique(keys)
        bits = ((uniq[:, None] >> np.arange(max(m, 1))) & 1)[:, :m] if m else np.zeros((1, 0), int)
        a_idx = bits.sum(axis=1)

        pred = math.log(lam / (1.0 - lam)) + vl.ell[a_idx][ok]
        assert np.max(np.abs(pred - logit_direct)) < 1e-12, (spec.key, w, lam, m)


# ==========================================================================
# T3 -- mpmath cross-check
# ==========================================================================


@pytest.mark.parametrize("spec", LAWS)
def test_T3_mpmath_cross_check(spec):
    """Random cells recomputed at 50 digits with no float64 intermediates.

    An absolute error in ``log D`` *is* a relative error in ``D``, so the
    spec's "rel. < 1e-10" is asserted as ``|log D - log D_mp| < 1e-10``.
    """
    rng = np.random.default_rng(20260725)
    for _ in range(5):
        N = int(rng.choice([127, 255]))
        w = float(rng.choice([0.5, 0.9]))
        m = int(rng.choice(visible_sizes(N)))
        a, b = mode_weight_interval(w)
        lam = float(rng.uniform(a, b))
        if abs(lam - w) < 1e-3:
            lam = b

        got = log_D(visible_law(spec, N, m, w), lam)
        ref = log_D_mp(spec, N, m, w, lam, dps=50)
        assert abs(got - ref) < 1e-10, (spec.key, N, m, w, lam, got, ref)

        got_u = log_U(visible_law(spec, N, m, w))
        ref_u = log_U_mp(spec, N, m, w, dps=50)
        assert abs(got_u - ref_u) < 1e-10, (spec.key, N, m, w, got_u, ref_u)


@pytest.mark.parametrize("spec", LAWS)
@pytest.mark.parametrize("w", W_TEST)
def test_T3b_kl_implementations_agree(spec, w):
    """Branched (spec section 2.3) vs threshold-free ``expm1``/``log1p`` form.

    Two algebraically identical evaluations of the same quantity.  They are
    compared on the *aggregate* ``log D`` rather than fibre by fibre: the
    branched middle branch cancels catastrophically just inside ``|x| = T``
    and carries percent-level error on those individual fibres, but they
    contribute negligibly to the sum.  T3 pins ``stable`` to mpmath; this test
    pins ``branched`` to ``stable`` at the accuracy the branch scheme can
    actually deliver.
    """
    lam_grid = lambda_grid_I(w, 51)
    worst = 0.0
    for spec_, N, m, w_ in _cells([127], [w], [spec]):
        vl = visible_law(spec_, N, m, w_)
        for lam in lam_grid:
            if lam == w_:
                continue
            a = log_D(vl, float(lam), 30.0, "branched")
            b = log_D(vl, float(lam), 30.0, "stable")
            worst = max(worst, abs(a - b))
            assert abs(a - b) < 1e-5, (spec_.key, N, m, lam, a, b)

    # Fibre by fibre the two agree to machine precision only where the middle
    # branch is well conditioned.  Its relative error grows like eps * e^|x|
    # (the identity cancels to O(e^-|x|)), so |x| <~ 12 is its usable range --
    # well short of the prescribed T = 30.  This is the concrete reason the
    # threshold-free form is the default.
    vl = visible_law(spec, 127, 16, w)
    d = 0.5
    a = log_kl_branched(vl.logit_b, d, 30.0)
    b = log_kl_stable(vl.logit_b, d)
    well_conditioned = np.abs(vl.logit_b) <= 12.0
    fin = np.isfinite(a) & np.isfinite(b) & well_conditioned
    assert np.max(np.abs(a[fin] - b[fin])) < 1e-9, worst


# ==========================================================================
# T4 -- curvature identity
# ==========================================================================


@pytest.mark.parametrize("spec", LAWS)
@pytest.mark.parametrize("w", W_TEST)
def test_T4_curvature_identity(spec, w):
    """``kappa_exact = U_eff / (w(1-w))^2`` vs a local numerical second derivative.

    The exact identity is
    ``d^2 D/d lambda^2 |_w = mmse_p(Z|X_V) / (w(1-w))^2`` -- parameter-free.
    """
    lam = lambda_grid_I(w, 201)
    near = np.abs(lam - w) <= 0.02
    checked = 0
    for N in N_TEST:
        for m in visible_sizes(N):
            vl = visible_law(spec, N, int(m), w)
            kappa_ex = curvature_exact(log_U(vl), w)
            if kappa_ex <= 1e-200:
                continue
            kappa_fit = curvature_richardson(lambda l: log_D(vl, l), w)
            rel = abs(kappa_ex - kappa_fit) / kappa_ex
            assert rel < 1e-3, (spec.key, N, m, w, kappa_ex, kappa_fit, rel)

            # the spec-literal grid fit is also checked, but only at w = 1/2
            # where its odd Taylor terms cancel by symmetry
            if w == 0.5:
                kg = quadratic_curvature_fit(lam[near], np.exp(log_D_grid(vl, lam[near])), w)
                assert abs(kappa_ex - kg) / kappa_ex < 1e-3, (spec.key, N, m, kg)
            checked += 1
    assert checked > 0


# ==========================================================================
# T5 -- full mask
# ==========================================================================


@pytest.mark.parametrize("spec", LAWS)
@pytest.mark.parametrize("N", N_TEST)
@pytest.mark.parametrize("w", W_TEST)
def test_T5_full_mask(spec, N, w):
    """``D_{N,0}(lambda) = kl(w||lambda)`` exactly and ``U_{N,0} = w(1-w)``."""
    vl = visible_law(spec, N, 0, w)
    assert abs(math.exp(log_U(vl)) - w * (1.0 - w)) < 1e-15

    err = 0.0
    for lam in lambda_grid_I(w, 201):
        if lam == w:
            continue
        err = max(err, abs(math.exp(log_D(vl, float(lam))) - binary_kl(w, float(lam))))
    assert err < 1e-12, (spec.key, N, w, err)

    # endpoint-flow identity: F^tau_{N,0} = p(X_{-tau})
    lfp, lfm = log_flows(vl)
    assert abs(math.exp(lfp) - (1.0 - w)) < 1e-14
    assert abs(math.exp(lfm) - w) < 1e-14


# ==========================================================================
# T6 -- boost linearity
# ==========================================================================


@pytest.mark.parametrize("spec", LAWS)
@pytest.mark.parametrize("w", W_TEST)
@pytest.mark.parametrize("pi", [1e-3, 1e-2, 1e-1, 0.5])
def test_T6_boost_linearity(spec, w, pi):
    """Linearity vs a directly constructed two-component schedule.

    The reference builds ``mu = (1-pi) delta_{V_m} + pi delta_{V_s}`` over
    *actual mask sets* and evaluates ``D_mu`` from the definition on the
    explicit state space.
    """
    N, m_base, s = 9, 6, 2
    bl = build_brute_law(spec, N, w)
    V_m, V_s = tuple(range(m_base)), tuple(range(s))
    vl_m = visible_law(spec, N, m_base, w)
    vl_s = visible_law(spec, N, s, w)

    for lam in (0.3, 0.45, 0.7):
        direct = bl.D_schedule({V_m: 1.0 - pi, V_s: pi}, lam)
        via_lin = math.exp(float(log_mix(log_D(vl_m, lam), log_D(vl_s, lam), pi)))
        assert abs(direct - via_lin) <= 1e-12 * max(1.0, abs(direct)), (
            spec.key, w, pi, lam, direct, via_lin
        )

    # the same for the MMSE coefficient and the cross-mode flow
    u_direct = (1 - pi) * bl.mmse(V_m) + pi * bl.mmse(V_s)
    u_lin = math.exp(float(log_mix(log_U(vl_m), log_U(vl_s), pi)))
    assert abs(u_direct - u_lin) <= 1e-12 * max(1.0, u_direct)


# ==========================================================================
# T7 -- sensitivity band (this is a theorem: a violation is a bug)
# ==========================================================================


@pytest.mark.parametrize("spec", LAWS)
@pytest.mark.parametrize("w", W_TEST)
def test_T7_sensitivity_band(spec, w):
    """``c_I <= rho_{N,m}(lambda) <= C_I`` over *all* cells -- Lem. `sensitivity`."""
    bands = band_constants(w)
    lam = lambda_grid_I(w, 201)
    lo_seen, hi_seen = math.inf, -math.inf
    for N in N_TEST:
        for m in visible_sizes(N):
            vl = visible_law(spec, N, int(m), w)
            _, rho = rho_grid(log_D_grid(vl, lam), log_U(vl), lam, w)
            rho = rho[np.isfinite(rho)]
            if rho.size == 0:
                continue
            lo_seen, hi_seen = min(lo_seen, rho.min()), max(hi_seen, rho.max())
            assert rho.min() >= bands["c_I"] * (1 - 1e-12), (spec.key, N, m, rho.min())
            assert rho.max() <= bands["C_I"] * (1 + 1e-12), (spec.key, N, m, rho.max())
    # the exact limit rho(w) = 1/(w(1-w)) must sit strictly inside the band
    assert bands["c_I"] < bands["rho_limit"] < bands["C_I"]
    assert lo_seen <= bands["rho_limit"] * (1 + 1e-6)


# ==========================================================================
# T8 -- lambda = w
# ==========================================================================


@pytest.mark.parametrize("spec", LAWS)
@pytest.mark.parametrize("w", W_TEST)
def test_T8_zero_at_w(spec, w):
    """``log D_{N,m}(w) = -inf``."""
    for N in N_TEST:
        for m in visible_sizes(N):
            val = log_D(visible_law(spec, N, int(m), w), w)
            assert val < -1e6, (spec.key, N, m, val)


# ==========================================================================
# T9 -- w = 1/2 symmetry
# ==========================================================================


@pytest.mark.parametrize("spec", LAWS)
def test_T9_symmetry_at_half(spec):
    """``D(lambda) = D(1-lambda)`` when ``w = 1/2``."""
    w = 0.5
    lam = lambda_grid_I(w, 201)
    for N in N_TEST:
        for m in visible_sizes(N):
            vl = visible_law(spec, N, int(m), w)
            v = log_D_grid(vl, lam)
            fin = np.isfinite(v) & np.isfinite(v[::-1])
            assert np.max(np.abs(v[fin] - v[::-1][fin])) < 1e-12, (spec.key, N, m)


# ==========================================================================
# T10 -- monotonicity
# ==========================================================================


@pytest.mark.parametrize("spec", LAWS)
@pytest.mark.parametrize("w", W_TEST)
def test_T10_monotonicity(spec, w):
    """``D`` decreases on ``[a, w]`` and increases on ``[w, b]``: no interior max."""
    lam = lambda_grid_I(w, 201)
    i_w = int(np.argmin(np.abs(lam - w)))
    for N in N_TEST:
        for m in visible_sizes(N):
            v = log_D_grid(visible_law(spec, N, int(m), w), lam)
            left, right = v[: i_w + 1], v[i_w:]
            assert np.all(np.diff(left) <= 1e-12), (spec.key, N, m, "left")
            assert np.all(np.diff(right) >= -1e-12), (spec.key, N, m, "right")
            # the endpoint shortcut must reproduce the grid supremum
            ld, _ = log_Delta(visible_law(spec, N, int(m), w))
            assert ld >= np.nanmax(v[np.isfinite(v)]) - 1e-12


# ==========================================================================
# T11 -- Law P: exact-LLR lemma and the conditioning-term bound
# ==========================================================================


@pytest.mark.parametrize("spec", PRODUCT_LAWS)
@pytest.mark.parametrize("N", N_TEST)
def test_T11_llr_linearity(spec, N):
    r"""Lem. `exact-llr` + Lem. `T-control`.

    (ii) ``|ell(r) - L0 r| <= eta_N`` for ``|V| <= s_ov = floor(alpha N)`` --
    the two-sided bound, which is exactly where Ass. `low-vis-uncertainty` uses it.

    (i) ``ell(r) >= L0 r`` whenever ``r >= 0``, at *every* ``|V|`` -- the
    one-sided bound, which is what mode pinning uses.
    """
    pc = ProductConstants(spec.param, N)
    for m in visible_sizes(N):
        vl = visible_law(spec, N, int(m), 0.5)
        finite = np.isfinite(vl.ell)
        resid = vl.ell[finite] - pc.L0 * vl.r[finite]

        if m <= pc.s_ov:
            assert np.max(np.abs(resid)) <= pc.eta_N + 1e-12, (
                spec.key, N, m, np.max(np.abs(resid)), pc.eta_N
            )
        nonneg = vl.r[finite] >= 0
        if nonneg.any():
            assert np.min(resid[nonneg]) >= -1e-12, (spec.key, N, m, np.min(resid[nonneg]))


# ==========================================================================
# T12 -- Law P: the closed-form proposition, as assertions
# ==========================================================================


@pytest.mark.parametrize("spec", PRODUCT_LAWS)
@pytest.mark.parametrize("N", N_TEST)
def test_T12a_mmse_upper_pointwise(spec, N):
    """Cor. `mmse-upper`: ``mmse_p(Z|X_V) <= (5/4) e^{-c|V|}`` for ``|V| >= s_pin``."""
    pc = ProductConstants(spec.param, N)
    for m in visible_sizes(N):
        if m < pc.s_pin:
            continue
        lu = log_U(visible_law(spec, N, int(m), 0.5))
        assert lu <= math.log(1.25) - pc.c * m + 1e-12, (spec.key, N, m, lu)


@pytest.mark.parametrize("spec", PRODUCT_LAWS)
@pytest.mark.parametrize("N", N_TEST)
def test_T12b_low_vis_pointwise(spec, N):
    """Ass. `low-vis-uncertainty`: ``mmse_p(Z|X_V) >= u0 e^{-L|V|}`` for ``|V| <= s_ov``."""
    pc = ProductConstants(spec.param, N)
    for m in range(0, pc.s_ov + 1):
        lu = log_U(visible_law(spec, N, int(m), 0.5))
        assert lu >= math.log(pc.u0) - pc.L * m - 1e-12, (spec.key, N, m, lu)


def test_T12c_fitted_constants_respect_the_proposition(tmp_path):
    """``c <= c_hat_N <= L``, ``u0_hat >= 1/4``, ``L_hat <= L0 + eta_N``.

    Also checks the bracketing of Rem. `bracketing`: ``c < rate* < L0 <= L``,
    and that the measured rate lands on ``rate*`` rather than on ``L0 theta``.
    """
    cfg = RunConfig(laws=("product-0.6", "product-0.8"), N_grid=(127, 255), w_grid=(0.5,))
    e1 = run_e1(cfg, results_dir=tmp_path, verbose=False)
    e2 = run_e2(cfg, results_dir=tmp_path, verbose=False)

    fits = e1["fits"]
    fits = fits[(fits.xvar == "m")]
    assert len(fits) > 0
    for _, row in fits.iterrows():
        pc = ProductConstants(row.param, int(row.N))
        assert pc.c <= row.rate_hat <= pc.L, (row.param, row.N, row.target, row.rate_hat)
        # the certified envelope must strictly bracket the true rate
        assert pc.c < pc.rate_star < pc.L0 <= pc.L
        # and the measurement must land on rate*, not on the typical-context slope
        assert abs(row.rate_hat - pc.rate_star) / pc.rate_star < 0.05
        assert abs(row.rate_hat - pc.L0 * row.param) / pc.rate_star > 0.5

    us = e2["uscale"]
    assert us.u0_ok.dropna().all()
    assert us.L_ok.dropna().all()
    assert us.assumption_holds.dropna().all()

    # Delta and U must decay at the same rate (the sensitivity lemma, indirectly)
    for (param, N), grp in fits.groupby(["param", "N"]):
        rates = grp.set_index("target").rate_hat
        if {"Delta", "U"} <= set(rates.index):
            assert abs(rates["Delta"] - rates["U"]) < 5e-3, (param, N)


# ==========================================================================
# End-to-end: the count reduction against explicit enumeration
# ==========================================================================


@pytest.mark.parametrize("spec", LAWS)
@pytest.mark.parametrize("w", W_TEST)
def test_brute_force_agreement(spec, w):
    """Every estimand vs explicit enumeration over ``{-1,+1}^N``, ``N = 9``."""
    N = 9
    bl = build_brute_law(spec, N, w)
    for m in range(0, N):
        V = tuple(range(m))
        vl = visible_law(spec, N, m, w)
        assert abs(bl.mmse(V) - math.exp(log_U(vl))) < 1e-13
        assert abs(bl.flow_plus(V) - math.exp(log_flows(vl)[0])) < 1e-13
        for lam in (0.2, 0.5, 0.83):
            got = math.exp(log_D(vl, lam))
            ref = bl.D_K(V, lam)
            assert abs(got - ref) <= 1e-12 * max(1.0, ref), (spec.key, w, m, lam)


@pytest.mark.slow
def test_full_run_matrix(tmp_path):
    """The whole spec run matrix, end to end (slow: enable with ``-m slow``)."""
    cfg = RunConfig()
    run_e1(cfg, results_dir=tmp_path, verbose=False)
    run_e2(cfg, results_dir=tmp_path, verbose=False)


# ==========================================================================
# T13-T15 -- E4, the sampling cost of low-visibility mass
#
# These are additions to the specification's T1-T12, not part of it: E4 was
# added to give Sec. 5.2 ("The cost of the full mask") something measured to
# stand on.  The pattern is the same -- an exact claim checked against
# explicit enumeration, plus the scaling laws the panels assert.
# ==========================================================================


@pytest.mark.parametrize("spec", LAWS)
@pytest.mark.parametrize("w", W_TEST)
def test_T13_second_moment_vs_brute_force(spec, w):
    """``log E[S^2]`` vs explicit enumeration over ``{-1,+1}^9``.

    The closed form exists only because ``S`` is two-point on each fibre; if
    that reduction were wrong this is the test that would catch it.
    """
    N = 9
    bl = build_brute_law(spec, N, w)
    for m in range(0, N):
        V = tuple(range(m))
        vl = visible_law(spec, N, m, w)
        for lam in (0.2, 0.5, 0.83):
            ref_mean, ref_second = bl.excess_loss_moments(V, lam)
            got_mean = math.exp(log_D(vl, lam))
            got_second = math.exp(log_second_moment(vl, lam))
            assert abs(got_mean - ref_mean) <= 1e-12 * max(1.0, ref_mean)
            assert abs(got_second - ref_second) <= 1e-12 * max(1.0, ref_second), (
                spec.key, w, m, lam
            )


@pytest.mark.parametrize("spec", LAWS)
@pytest.mark.parametrize("w", W_TEST)
def test_T14_raw_loss_vs_brute_force(spec, w):
    """``H(X_K | X_V)`` and its variance vs explicit enumeration, ``N = 9``.

    Also pins the identity that makes panel A9(d) meaningful: at ``m = 0`` the
    raw loss *is* the joint entropy, so the full-mask channel prices in the
    whole law rather than a conditional slice of it.
    """
    N = 9
    bl = build_brute_law(spec, N, w)
    for m in range(0, N):
        V = tuple(range(m))
        ref_mean, ref_var = bl.raw_loss_moments(V)
        got_mean, got_var = raw_loss_moments(spec, N, m, w)
        assert abs(got_mean - ref_mean) <= 1e-11 * max(1.0, ref_mean), (spec.key, w, m)
        assert abs(got_var - ref_var) <= 1e-10 * max(1.0, ref_var), (spec.key, w, m)
        # the conditional loss can only fall as more of the sequence is shown
        assert got_mean <= raw_loss_moments(spec, N, 0, w)[0] + 1e-12


@pytest.mark.parametrize("spec", PRODUCT_LAWS)
@pytest.mark.parametrize("w", W_TEST)
def test_T15_detection_cost_scaling(spec, w):
    """``n_star ~ 1/pi`` and ``n_star ~ (lambda-w)^{-2}``, the two A9 slopes.

    Both are exact consequences of the moment structure -- the signal is linear
    in the schedule mass while the noise is not -- so they are asserted to four
    figures, not eyeballed off the panel.
    """
    N, m_base = 255, 127
    a_end, b_end = mode_weight_interval(w)
    vl_base, vl_boost = visible_law(spec, N, m_base, w), visible_law(spec, N, 0, w)

    def cost(pi, lam):
        return sampling_cost(
            float(log_mix(log_D(vl_base, lam), log_D(vl_boost, lam), pi)),
            float(log_mix(log_second_moment(vl_base, lam),
                          log_second_moment(vl_boost, lam), pi)),
        )

    # variance is non-negative wherever the witness differs from p
    for pi in (0.0, 1e-4, 1e-2):
        for lam in (a_end, b_end):
            assert cost(pi, lam).log_var >= 2.0 * cost(pi, lam).log_mean - 1e-9

    # The -1 is exact only in the limit.  Once the boost dominates the base
    # mask, n_star = M_s/(pi D_s^2) - 1, and it is that trailing -1 -- the
    # E[S]^2 in the variance, not the base mask -- that bends the fit: the
    # residual is O(pi) with coefficient D_s^2/M_s.  So the slope must tighten
    # sharply on the small-pi end and nowhere else.  A wrong reduction would
    # move the exponent instead of merely bending it.
    def slope_over(pis):
        n = np.array([cost(float(pi), b_end).log_n_star for pi in pis])
        return float(np.polyfit(np.log(pis), n, 1)[0])

    full = slope_over(np.array([1e-5, 1e-4, 1e-3, 1e-2, 1e-1]))
    small = slope_over(np.array([1e-5, 1e-4, 1e-3]))
    assert abs(full + 1.0) < 2e-3, full
    assert abs(small + 1.0) < 1e-4, small
    assert abs(small + 1.0) < 0.2 * abs(full + 1.0), (small, full)

    errs = np.array([f * (b_end - w) for f in (0.4, 0.2, 0.1, 0.05, 0.02)])
    n_err = np.array([cost(1e-3, w + e).log_n_star for e in errs])
    slope_err = np.polyfit(np.log(errs), n_err, 1)[0]
    assert abs(slope_err + 2.0) < 2e-2, slope_err
