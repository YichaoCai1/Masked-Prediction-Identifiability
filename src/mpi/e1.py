"""E1 -- Mode blindness.

Population-exact verification of the blindness scaling
(Thm. `mode-blind`, Eq. (mode-blindness)) and of the sensitivity identity
(Lem. `sensitivity`).  No sampling, no optimisation, no approximation.

Produces ``constants``, ``profiles``, ``summary``, ``fits`` and ``numerics``.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    KL_IMPL_SENSITIVITY,
    RHO_FRACTIONS,
    LawSpec,
    ProductConstants,
    RunConfig,
    band_constants,
    lambda_grid_I,
    lambda_grid_wide,
    reference_rate,
    visible_sizes,
)
from .core import visible_law
from .estimands import log_D_grid, log_Delta, log_U, log_flows, rho_grid
from .fits import best_window_ols, ols, window_shift_spread
from .tableio import write_table

__all__ = ["run_e1"]


# --------------------------------------------------------------------------
# Predicted constants table
# --------------------------------------------------------------------------


def _constants_table(cfg: RunConfig) -> pd.DataFrame:
    rows = []
    for spec in cfg.law_specs():
        for N in cfg.N_grid:
            row = {
                "law": spec.law,
                "param": spec.param,
                "N": N,
                "pinned": spec.pinned,
                "rate_ref": reference_rate(spec),
            }
            if spec.law == "product":
                pc = ProductConstants(spec.param, N)
                row.update(
                    {
                        "L0": pc.L0, "alpha": pc.alpha, "eta_N": pc.eta_N,
                        "c0": pc.c0, "kappa": pc.kappa, "c1": pc.c1, "c": pc.c,
                        "s_pin": pc.s_pin, "u0": pc.u0, "L": pc.L,
                        "s_ov": pc.s_ov, "rate_star": pc.rate_star,
                        "window_empty": pc.window_empty,
                    }
                )
            else:
                row.update(
                    {
                        "L0": np.nan, "alpha": np.nan, "eta_N": np.nan,
                        "c0": 0.5, "kappa": np.nan, "c1": np.nan, "c": np.nan,
                        "s_pin": -1, "u0": np.nan, "L": np.nan,
                        "s_ov": -1, "rate_star": np.nan, "window_empty": True,
                    }
                )
            rows.append(row)
    df = pd.DataFrame(rows)
    for w in cfg.w_grid:
        bands = band_constants(w)
        for key in ("c_I", "C_I", "rho_limit", "delta_I", "m_I"):
            df[f"{key}_w{w:g}"] = bands[key]
    return df


# --------------------------------------------------------------------------
# Per-(law, N, w) sweep
# --------------------------------------------------------------------------


def _sweep_cell(spec: LawSpec, N: int, w: float, cfg: RunConfig):
    """Return (profile rows, summary rows) for one (law, N, w) combination."""
    lam_I = lambda_grid_I(w, cfg.n_lambda_I)
    lam_wide = lambda_grid_wide(cfg.n_lambda_wide)
    bands = band_constants(w)

    prof_rows: list[dict] = []
    summ_rows: list[dict] = []

    for m in visible_sizes(N):
        vl = visible_law(spec, N, int(m), w)

        logD_I = log_D_grid(vl, lam_I, cfg.branch_T, cfg.kl_impl)
        logD_wide = log_D_grid(vl, lam_wide, cfg.branch_T, cfg.kl_impl)
        lU = log_U(vl)
        lFp, lFm = log_flows(vl)
        lDelta, lam_star = log_Delta(vl, cfg.branch_T, cfg.kl_impl)

        _, rho = rho_grid(logD_I, lU, lam_I, w)
        rho_fin = rho[np.isfinite(rho)]

        for grid_name, lams, vals in (("I", lam_I, logD_I), ("wide", lam_wide, logD_wide)):
            prof_rows.extend(
                {
                    "law": spec.law, "param": spec.param, "N": N, "w": w,
                    "m": int(m), "grid": grid_name,
                    "lambda": float(l), "logD": float(v),
                }
                for l, v in zip(lams, vals)
            )

        summ_rows.append(
            {
                "law": spec.law, "param": spec.param, "N": N, "w": w, "m": int(m),
                "logDelta": lDelta, "lambda_star": lam_star,
                "logU": lU, "logFplus": lFp, "logFminus": lFm,
                "rho_min": float(rho_fin.min()) if rho_fin.size else np.nan,
                "rho_max": float(rho_fin.max()) if rho_fin.size else np.nan,
                "c_I": bands["c_I"], "C_I": bands["C_I"],
                "rho_limit": bands["rho_limit"],
            }
        )
    return prof_rows, summ_rows


# --------------------------------------------------------------------------
# Decay-rate fits
# --------------------------------------------------------------------------


def _numerical_ci(
    spec: LawSpec, N: int, w: float, ms: np.ndarray, base_fit, cfg: RunConfig,
    target: str,
) -> tuple[float, float, list[dict]]:
    """Numerical CI on the fitted slope: window shifts + KL evaluation scheme.

    There is no sampling noise, so this is the only meaningful notion of
    uncertainty for E1/E2 (spec section 3.1).  Leg (c), the float64-vs-mpmath
    comparison, is produced separately by ``scripts/audit_numerics.py``
    because it costs arbitrary-precision work per cell.
    """
    slopes: list[float] = [base_fit.slope]
    detail: list[dict] = []

    # (a) fit-window shifts of +-1 grid point
    y_base = _target_series(spec, N, w, ms, cfg.branch_T, cfg.kl_impl, target)
    lo, hi, all_slopes = window_shift_spread(ms, y_base, base_fit)
    slopes.extend(all_slopes)
    detail.append({"knob": "window", "lo": lo, "hi": hi})

    # (b) binary-KL evaluation scheme and branch threshold
    sel = slice(base_fit.idx_lo, base_fit.idx_hi + 1)
    for impl, T in KL_IMPL_SENSITIVITY:
        if (impl, T) == (cfg.kl_impl, cfg.branch_T):
            continue
        y_T = _target_series(spec, N, w, ms, T, impl, target)
        s_T = ols(np.asarray(ms, dtype=float)[sel], y_T[sel]).slope
        slopes.append(s_T)
        label = impl if impl == "stable" else f"{impl}@T={T:g}"
        detail.append({"knob": label, "lo": s_T, "hi": s_T})

    return min(slopes), max(slopes), detail


def _target_series(spec, N, w, ms, T, impl, target) -> np.ndarray:
    out = []
    for m in ms:
        vl = visible_law(spec, N, int(m), w)
        out.append(log_Delta(vl, T, impl)[0] if target == "Delta" else log_U(vl))
    return np.asarray(out, dtype=float)


def _fit_rows(summary: pd.DataFrame, cfg: RunConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    fit_rows: list[dict] = []
    numeric_rows: list[dict] = []

    for spec in cfg.law_specs():
        rate_star = reference_rate(spec)
        for w in cfg.w_grid:
            for N in cfg.N_grid:
                sub = summary[
                    (summary.law == spec.law)
                    & (summary.param == spec.param)
                    & (summary.N == N)
                    & (summary.w == w)
                ].sort_values("m")
                if sub.empty:
                    continue
                ms = sub.m.to_numpy(dtype=float)
                pc = ProductConstants(spec.param, N) if spec.law == "product" else None

                for target, col in (("Delta", "logDelta"), ("U", "logU")):
                    y = sub[col].to_numpy(dtype=float)
                    fit = best_window_ols(ms, y, cfg.fit_min_window)
                    if fit is None:
                        continue
                    ci_lo, ci_hi, detail = _numerical_ci(
                        spec, N, w, ms, fit, cfg, target
                    )
                    slope_hat = -fit.slope   # report the decay *rate*, a positive number
                    envelope_ok = (
                        bool(pc.c <= slope_hat <= pc.L) if pc is not None else None
                    )
                    fit_rows.append(
                        {
                            "law": spec.law, "param": spec.param, "N": N, "w": w,
                            "target": target, "xvar": "m", "frac": np.nan,
                            "slope": fit.slope, "rate_hat": slope_hat,
                            "intercept": fit.intercept, "R2": fit.r2,
                            "n_points": fit.n, "fit_lo": fit.x_lo, "fit_hi": fit.x_hi,
                            "ci_lo": -ci_hi, "ci_hi": -ci_lo,
                            "rate_ci_lo": -ci_hi, "rate_ci_hi": -ci_lo,
                            "rate_star": rate_star,
                            "c_pred": pc.c if pc else np.nan,
                            "L_pred": pc.L if pc else np.nan,
                            "envelope_ok": envelope_ok,
                            "rel_err_vs_rate_star": (
                                abs(slope_hat - rate_star) / rate_star
                                if rate_star else np.nan
                            ),
                        }
                    )
                    numeric_rows.extend(
                        {
                            "law": spec.law, "param": spec.param, "N": N, "w": w,
                            "target": target, **d,
                        }
                        for d in detail
                    )

            # fixed-fraction decay: log Delta_{N, floor(rho N)} vs N
            for frac in RHO_FRACTIONS:
                xs, ys = [], []
                for N in cfg.N_grid:
                    m_target = int(math.floor(frac * N))
                    row = summary[
                        (summary.law == spec.law)
                        & (summary.param == spec.param)
                        & (summary.N == N)
                        & (summary.w == w)
                        & (summary.m == m_target)
                    ]
                    if not row.empty:
                        xs.append(float(N))
                        ys.append(float(row.logDelta.iloc[0]))
                if len(xs) >= 2:
                    f = ols(xs, ys)
                    fit_rows.append(
                        {
                            "law": spec.law, "param": spec.param, "N": np.nan, "w": w,
                            "target": "Delta_frac", "xvar": "N", "frac": frac,
                            "slope": f.slope, "rate_hat": -f.slope,
                            "intercept": f.intercept, "R2": f.r2, "n_points": f.n,
                            "fit_lo": f.x_lo, "fit_hi": f.x_hi,
                            "ci_lo": np.nan, "ci_hi": np.nan,
                            "rate_ci_lo": np.nan, "rate_ci_hi": np.nan,
                            "rate_star": (rate_star * frac) if rate_star else np.nan,
                            "c_pred": np.nan, "L_pred": np.nan,
                            "envelope_ok": None, "rel_err_vs_rate_star": np.nan,
                        }
                    )

    return pd.DataFrame(fit_rows), pd.DataFrame(numeric_rows)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------


def run_e1(cfg: RunConfig, results_dir: Path | None = None, verbose: bool = True):
    t0 = time.time()
    prof_rows: list[dict] = []
    summ_rows: list[dict] = []

    for spec in cfg.law_specs():
        for w in cfg.w_grid:
            for N in cfg.N_grid:
                tic = time.time()
                p, s = _sweep_cell(spec, N, w, cfg)
                prof_rows.extend(p)
                summ_rows.extend(s)
                if verbose:
                    print(
                        f"  E1 {spec.key:>12s}  N={N:<5d} w={w:<4g}"
                        f"  {len(s):2d} visible sizes  [{time.time()-tic:5.1f}s]"
                    )

    profiles = pd.DataFrame(prof_rows)
    summary = pd.DataFrame(summ_rows)
    fits, numerics = _fit_rows(summary, cfg)

    # mark which cells fell inside the Delta fit window
    summary["in_fit_window"] = False
    for _, f in fits[(fits.target == "Delta") & (fits.xvar == "m")].iterrows():
        sel = (
            (summary.law == f.law) & (summary.param == f.param)
            & (summary.N == f.N) & (summary.w == f.w)
            & (summary.m >= f.fit_lo) & (summary.m <= f.fit_hi)
        )
        summary.loc[sel, "in_fit_window"] = True
        summary.loc[
            (summary.law == f.law) & (summary.param == f.param)
            & (summary.N == f.N) & (summary.w == f.w), ["fit_lo", "fit_hi"]
        ] = [f.fit_lo, f.fit_hi]

    write_table(_constants_table(cfg), "constants", cfg, results_dir)
    write_table(profiles, "profiles", cfg, results_dir)
    write_table(summary, "summary", cfg, results_dir)
    write_table(fits, "fits", cfg, results_dir)
    write_table(numerics, "numerics", cfg, results_dir)

    if verbose:
        print(f"  E1 done in {time.time()-t0:.1f}s "
              f"({len(profiles)} profile rows, {len(summary)} cells)")
    return {"profiles": profiles, "summary": summary, "fits": fits, "numerics": numerics}
