"""Exact low-visibility mask-schedule intervention.

Population-exact verification of the recovery scaling (Cor. `recovery`) and of
the full-mask boundary (Prop. `full-mask`).  This stage adds *no* enumeration cost: for
a two-point schedule ``mu = (1-pi) delta_m + pi delta_s`` every quantity is an
exact convex combination of the two component quantities already computed by
the mode-blindness stage
(spec section 2.5).

Produces ``boost``, ``boostfits`` and ``uscale``.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    LawSpec,
    ProductConstants,
    RunConfig,
    lambda_grid_I,
    mode_weight_interval,
)
from .core import visible_law
from .estimands import (
    boosted_log_U,
    curvature_exact,
    log_D,
    log_D_grid,
    log_flows,
    log_mix,
    log_U,
)
from .fits import (
    anchored_exponential_fit,
    curvature_richardson,
    ols,
    quadratic_curvature_fit,
    recovery_radius,
)
from .tableio import write_table

__all__ = ["run_low_visibility_intervention"]


def _base_sizes(N: int) -> tuple[int, ...]:
    return (N // 4, N // 2)


# --------------------------------------------------------------------------
# Low-visibility uncertainty scale  (Ass. `low-vis-uncertainty`)
# --------------------------------------------------------------------------


def _uscale_rows(spec: LawSpec, N: int, w: float, cfg: RunConfig) -> list[dict]:
    s_vals = list(cfg.s_boost_grid)
    log_u = [log_U(visible_law(spec, N, s, w)) for s in s_vals]

    anchor = math.log(w * (1.0 - w))              # exact: U_{N,0} = w(1-w)
    u0_anchored, L_anchored = anchored_exponential_fit(s_vals, log_u, anchor)
    free = ols([float(s) for s in s_vals], log_u)
    u0_free, L_free = math.exp(free.intercept), -free.slope

    if spec.law == "product":
        pc = ProductConstants(spec.param, N)
        u0_pred, L_pred, s_ov = pc.u0, pc.L, pc.s_ov
    else:
        u0_pred, L_pred, s_ov = np.nan, np.nan, -1

    rows = []
    for s, lu in zip(s_vals, log_u):
        # The assumption itself, checked pointwise (this is the substantive claim;
        # the fit is only a summary of it).
        holds = (
            bool(lu >= math.log(u0_pred) - L_pred * s - 1e-12)
            if spec.law == "product" and s <= s_ov
            else None
        )
        rows.append(
            {
                "law": spec.law, "param": spec.param, "N": N, "w": w, "s": s,
                "logU": lu, "U": math.exp(lu),
                "u0_hat": u0_anchored, "L_hat": L_anchored,
                "u0_hat_free": u0_free, "L_hat_free": L_free,
                "u0_pred": u0_pred, "L_pred": L_pred,
                "s_ov": s_ov,
                "assumption_holds": holds,
                "u0_ok": bool(u0_anchored >= u0_pred - 1e-12)
                if spec.law == "product" else None,
                "L_ok": bool(L_anchored <= L_pred + 1e-12)
                if spec.law == "product" else None,
            }
        )
    return rows


# --------------------------------------------------------------------------
# Boosted schedules
# --------------------------------------------------------------------------


def _boost_rows(spec: LawSpec, N: int, w: float, cfg: RunConfig) -> list[dict]:
    a, b = mode_weight_interval(w)
    lam_I = lambda_grid_I(w, cfg.n_lambda_I)
    near = np.abs(lam_I - w) <= 0.02

    # Component quantities: one enumeration per visible size, reused everywhere.
    comp: dict[int, dict] = {}
    for m in set(_base_sizes(N)) | set(cfg.s_boost_grid):
        vl = visible_law(spec, N, int(m), w)
        lfp, lfm = log_flows(vl)
        comp[int(m)] = {
            "logU": log_U(vl),
            "logFplus": lfp,
            "logFminus": lfm,
            "logD_near": log_D_grid(vl, lam_I[near], cfg.branch_T, cfg.kl_impl),
            "logD_a": log_D(vl, a, cfg.branch_T, cfg.kl_impl),
            "logD_b": log_D(vl, b, cfg.branch_T, cfg.kl_impl),
            "vl": vl,
        }

    pc = ProductConstants(spec.param, N) if spec.law == "product" else None
    rows: list[dict] = []

    for m_base in _base_sizes(N):
        cb = comp[m_base]
        for s in cfg.s_boost_grid:
            cs = comp[s]
            for pi in cfg.pi_grid:
                logU_boost = boosted_log_U(cb["logU"], cs["logU"], pi)
                kappa_ex = curvature_exact(logU_boost, w)

                # exact linearity of the profile (spec 2.5)
                logD_near = log_mix(cb["logD_near"], cs["logD_near"], pi)
                kappa_fit_grid = quadratic_curvature_fit(
                    lam_I[near], np.exp(np.asarray(logD_near, dtype=float)), w
                )

                def logD_boost(lam: float, _cb=cb, _cs=cs, _pi=pi) -> float:
                    return float(
                        log_mix(
                            log_D(_cb["vl"], lam, cfg.branch_T, cfg.kl_impl),
                            log_D(_cs["vl"], lam, cfg.branch_T, cfg.kl_impl),
                            _pi,
                        )
                    )

                kappa_fit = curvature_richardson(logD_boost, w)

                # boost-dominance indicator: pi * U_s vs U_{m_base}
                ratio = math.exp(math.log(pi) + cs["logU"] - cb["logU"]) if pi > 0 else 0.0

                for eps in cfg.eps_grid:
                    rr = recovery_radius(logD_boost, w, (a, b), eps, cfg.bisect_tol)
                    rows.append(
                        {
                            "law": spec.law, "param": spec.param, "N": N, "w": w,
                            "m_base": int(m_base), "s": int(s), "pi": pi, "eps": eps,
                            "kappa_exact": kappa_ex, "kappa_fit": kappa_fit,
                            "kappa_fit_grid": kappa_fit_grid,
                            "kappa_pred_boost": (
                                math.exp(math.log(pi) + cs["logU"]
                                         - 2.0 * math.log(w * (1 - w)))
                                if pi > 0 else 0.0
                            ),
                            "logU_boost": logU_boost,
                            "boost_dominance": ratio,
                            "boost_dominant": bool(ratio >= 10.0),
                            "r_eps": rr.radius, "censored": rr.censored,
                            "side": rr.side,
                            "logFplus_boost": float(
                                log_mix(cb["logFplus"], cs["logFplus"], pi)
                            ),
                            "logFminus_boost": float(
                                log_mix(cb["logFminus"], cs["logFminus"], pi)
                            ),
                            "s_pin": pc.s_pin if pc else -1,
                            "s_ov": pc.s_ov if pc else -1,
                            "certified": (
                                bool(pc.s_pin <= pc.s_ov) if pc else False
                            ),
                        }
                    )
    return rows


# --------------------------------------------------------------------------
# Predicted-scaling fits
# --------------------------------------------------------------------------


def _boost_fit_rows(boost: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    keys = ["law", "param", "N", "w", "m_base", "s"]

    for key, grp in boost.groupby(keys, sort=False):
        meta = dict(zip(keys, key))

        # log kappa vs log pi -- predicted slope +1 in the boost-dominant regime
        kg = grp[(grp.pi > 0) & (grp.eps == grp.eps.iloc[0])].sort_values("pi")
        kg = kg[kg.kappa_exact > 0]
        if len(kg) >= 3:
            f = ols(np.log(kg.pi.to_numpy()), np.log(kg.kappa_exact.to_numpy()))
            rows.append({**meta, "relation": "kappa_vs_pi", "eps": np.nan,
                         "slope": f.slope, "intercept": f.intercept, "R2": f.r2,
                         "n_points": f.n, "slope_pred": 1.0})

        # log r vs log pi at each eps -- predicted slope -1/2
        for eps, g in grp.groupby("eps"):
            g = g[(g.pi > 0) & (~g.censored) & (g.r_eps > 0)].sort_values("pi")
            if len(g) >= 3:
                f = ols(np.log(g.pi.to_numpy()), np.log(g.r_eps.to_numpy()))
                rows.append({**meta, "relation": "r_vs_pi", "eps": eps,
                             "slope": f.slope, "intercept": f.intercept, "R2": f.r2,
                             "n_points": f.n, "slope_pred": -0.5})

        # log r vs log eps at each pi -- predicted slope +1/2
        for pi, g in grp.groupby("pi"):
            g = g[(~g.censored) & (g.r_eps > 0)].sort_values("eps")
            if len(g) >= 3:
                f = ols(np.log(g.eps.to_numpy()), np.log(g.r_eps.to_numpy()))
                rows.append({**meta, "relation": "r_vs_eps", "eps": np.nan,
                             "pi": pi, "slope": f.slope, "intercept": f.intercept,
                             "R2": f.r2, "n_points": f.n, "slope_pred": 0.5})

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def run_low_visibility_intervention(
    cfg: RunConfig, results_dir: Path | None = None, verbose: bool = True
):
    t0 = time.time()
    boost_rows: list[dict] = []
    uscale_rows: list[dict] = []

    for spec in cfg.law_specs():
        for w in cfg.w_grid:
            for N in cfg.N_grid:
                uscale_rows.extend(_uscale_rows(spec, N, w, cfg))
                if not spec.pinned:
                    continue  # The intervention theorem is stated for pinned laws only.
                tic = time.time()
                boost_rows.extend(_boost_rows(spec, N, w, cfg))
                if verbose:
                    print(f"  low visibility {spec.key:>12s}  N={N:<5d} w={w:<4g}"
                          f"  [{time.time()-tic:5.1f}s]")

    boost = pd.DataFrame(boost_rows)
    uscale = pd.DataFrame(uscale_rows)
    boostfits = _boost_fit_rows(boost) if not boost.empty else pd.DataFrame()

    write_table(boost, "boost", cfg, results_dir)
    write_table(boostfits, "boostfits", cfg, results_dir)
    write_table(uscale, "uscale", cfg, results_dir)

    if verbose:
        print(
            f"  low-visibility intervention done in {time.time()-t0:.1f}s "
            f"({len(boost)} boost cells)"
        )
    return {"boost": boost, "boostfits": boostfits, "uscale": uscale}
