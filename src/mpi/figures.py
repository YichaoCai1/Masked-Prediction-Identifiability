"""Semantically named paper figures regenerated from frozen result tables.

Most figures are emitted twice: as a composed multi-panel PDF
``figures/figure_<name>.pdf``, and panel by panel into
``figures/figure_<name>/panel_<key>.pdf`` so single panels can be dropped into
a paper's subfigure slots.  A panel's ``draw`` callback therefore runs twice and
must touch nothing but its own ``Axes`` -- that is the invariant to preserve
when editing one.

The natural-corpus result facets are the one deliberate exception: they are
written only as untitled standalone panels alongside the corpus diagnostics,
because the paper supplies their corpus names as subcaptions.

**No text that LaTeX should own is drawn into a PDF.**  There are no banner
titles; the ``(a)`` / ``(b)`` panel titles appear only on the composed contact
sheet, never on a standalone panel, whose ``\subcaption`` supplies them; and
the narrative lives in the paper text and in the README, not in the artwork.
What panels *do* keep is run context -- ``N``,
``m_base``, the witness -- as muted in-axes corner notes, since that is data
provenance rather than caption prose.

Explicitly *not* produced (spec section 7.2): ``(m, lambda) -> log D`` heatmaps
and 3-D surfaces. They duplicate the profile and decay-rate figures without
adding information.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from matplotlib import patheffects
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .config import (
    FIGURES_DIR,
    RESULTS_DIR,
    ProductConstants,
    band_constants,
    curie_weiss_magnetisation,
    mode_weight_interval,
)
from .style import (
    CAT,
    CONTINUOUS_CMAP,
    CORPUS_COLORS,
    CORPUS_LABELS,
    CORPUS_MARKERS,
    GRID,
    INK,
    MARKERS,
    MUTED,
    NULL_GREY,
    SECONDARY,
    SURFACE,
    annotate_panel,
    apply_style,
    corner_note,
    law_style,
    m_cmap,
    n_color,
    save,
)
from .tableio import read_table

__all__ = ["FIGURE_NAMES", "make_all_figures"]

FIGURE_NAMES = (
    "mode_blindness_and_recovery",
    "masked_discrepancy_profiles",
    "blindness_decay_rates",
    "low_visibility_intervention",
    "residual_uncertainty_by_visibility",
    "boundary_flow_mechanism",
    "asymmetric_mixture_weights",
    "numerical_stability_audit",
    "corpus_mode_uncertainty_diagnostics",
    "low_visibility_detection_cost",
    "low_visibility_loss_cost",
    "mode_weight_optimization",
)

#: The main-text figure uses Law P at theta = 0.8 with the symmetric prior.
MAIN_LAW = ("product", 0.8)
MAIN_W = 0.5
NULL_LAW = ("cw", 0.5)

#: Main-text panels sit at four to a page, so their titles run smaller.
M_TITLE_SIZE = 8.0

#: The strictly richer classifier feature set; every calibrated estimator is an
#: upper bound, so the richer one gives the best available bound.
BEST_FEATURES = "unigram_bigram"

#: Figures whose panels are meant to be re-laid-out as subfigures emit every
#: panel at one size, so none needs rescaling relative to the others. Profile
#: strips and decay-rate panels are deliberate exceptions.
# Authored close to print size.  These panels go in a single row at roughly
# 0.32\linewidth (1.76in) each, so a 3.8in source would be downscaled to 46%
# and its 8.5pt type would render at 3.9pt.  Hardcoded annotation sizes do not
# follow an rcParams bump, so the fix is to author small, not to enlarge type.
# Authored a little under the old 3.8x3.0 so that everything -- type, marker
# sizes, line widths, all fixed in points -- comes out proportionally larger
# once the panel is scaled into its subfigure slot, without dropping any of the
# panel content.
#: Every panel is authored a little smaller than its natural size, which makes
#: its type render correspondingly larger once placed in the paper.  This is a
#: uniform scale on the sizes the panels were designed at -- it does not touch
#: any panel's internal layout, so legends and annotations keep the positions
#: they were tuned for.  Sizing panels per-slot instead (so every figure hits
#: one rendered type size) forces 3-across panels down to ~2.4in, at which
#: point their legends have to be shrunk and repositioned; that changes how the
#: panels look, which is not worth uniformity.
FONT_BUMP = 1.15


def bumped(w: float, h: float) -> tuple[float, float]:
    """The design size, scaled so type renders ``FONT_BUMP`` times larger."""
    return (round(w / FONT_BUMP, 2), round(h / FONT_BUMP, 2))


M_PANEL = bumped(3.2, 2.55)

#: Summary panels sit in a single row, so each gets ~0.32\linewidth (1.76in).
#: At that width there is no room for an in-panel legend or for the off-axis
#: band annotations, and matplotlib does not reflow them -- it overlaps them.
#: Compact mode drops that furniture and shortens the axis labels; the caption
#: carries what it said.  Set False to restore the wide, self-contained panels.
M_COMPACT = False
A8_PANEL = bumped(4.4, 3.3)
#: Narrow corpus-diagnostic panels sit three-across in the paper, in the same
#: slot as the summary panels, so they render at the same type size.
A8_NARROW = (M_PANEL[0], round(M_PANEL[0] * 3.3 / 4.4, 2))

#: Panel M(a) y-axis.  ``True`` plots ``Delta`` itself on a log axis, so the
#: tick labels are magnitudes (10^0 ... 10^-171) and the reader sees *how
#: small* the discrepancy gets.  ``False`` plots ``log Delta`` in nats on a
#: linear axis, so the tick spacing is nats and the fitted slope can be read
#: off the axis directly as the decay rate.  Both draw the identical curve.
M_A_LOG_Y = True


# --------------------------------------------------------------------------
# Panel plumbing
# --------------------------------------------------------------------------


@dataclass
class Panel:
    """One panel, renderable either into a shared figure or on its own."""

    key: str                                  # semantic standalone filename stem
    title: str                                # the letter is added on drawing
    draw: Callable[[Axes], None]              # must only touch the given Axes
    figsize: tuple[float, float]              # size when rendered standalone
    title_size: float | None = None
    title_weight: str | None = None
    show_letter: bool = True
    tight: bool = True


def _grid(nrows: int, ncols: int, **kw):
    """A ``compose`` that lays panels out on a plain ``nrows x ncols`` grid."""

    def compose(fig):
        axes = fig.subplots(nrows, ncols, **kw)
        return list(np.atleast_1d(axes).ravel())

    return compose


def render(name: str, panels: list[Panel], compose, combined_figsize,
           out_dir: Path, png: bool = False, tight_kw: dict | None = None,
           decorate: Callable | None = None) -> None:
    """Emit the composed figure and one standalone PDF per panel.

    Only the *composed* figure carries the ``(a)`` / ``(b)`` panel titles.  A
    standalone panel is destined for a ``\\subfigure`` slot whose
    ``\\subcaption`` supplies that text, so burning it into the PDF would
    duplicate it -- and at a font size and position LaTeX cannot control.  The
    reclaimed strip goes to the axes, so standalone panels have slightly more
    plot area than their contact-sheet counterparts.
    """
    fig = plt.figure(figsize=combined_figsize)
    axes = compose(fig)
    for index, (panel, ax) in enumerate(zip(panels, axes)):
        ax._mpi_standalone = False
        panel.draw(ax)
        annotate_panel(ax, chr(ord("a") + index), panel.title, panel.title_size,
                       fontweight=panel.title_weight,
                       show_letter=panel.show_letter)
    for spare in axes[len(panels):]:      # ragged grids leave empty slots
        spare.set_axis_off()
    if decorate is not None:
        decorate(fig, axes[:len(panels)])
    if tight_kw is not None:
        fig.tight_layout(**tight_kw)
    save(fig, out_dir / f"figure_{name}", png=png)

    for panel in panels:
        f1, ax1 = plt.subplots(figsize=panel.figsize)
        ax1._mpi_standalone = True
        panel.draw(ax1)
        if panel.tight:
            f1.tight_layout()
        # exact figsize, not a content crop, so every panel of a figure comes
        # out identically sized and can be re-laid-out as subfigures
        save(f1, out_dir / f"figure_{name}" / f"panel_{panel.key}", png=png,
             tight=False)

    print(f"  figure_{name}: composed + {len(panels)} panels (untitled)")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _sel(df: pd.DataFrame, **kw) -> pd.DataFrame:
    mask = np.ones(len(df), dtype=bool)
    for k, v in kw.items():
        mask &= (df[k] == v).to_numpy()
    return df[mask]


def _has(df: pd.DataFrame | None) -> bool:
    return df is not None and len(df) > 0


def _load(results_dir: Path) -> dict[str, pd.DataFrame | None]:
    out: dict[str, pd.DataFrame | None] = {}
    aliases = {
        "constants": ("constants",),
        "profiles": ("profiles",),
        "summary": ("summary",),
        "fits": ("fits",),
        "numerics": ("numerics",),
        "boost": ("boost",),
        "boostfits": ("boostfits",),
        "uscale": ("uscale",),
        "corpus_uncertainty": ("corpus_uncertainty", "realdata"),
        "corpus_uncertainty_fits": ("corpus_uncertainty_fits", "realdata_fits"),
        "corpus_uncertainty_stats": ("corpus_uncertainty_stats", "realdata_stats"),
        "corpus_uncertainty_calibration": (
            "corpus_uncertainty_calibration",
            "realdata_calib",
        ),
        "corpus_uncertainty_form": ("corpus_uncertainty_form", "realdata_form"),
        "sampling": ("sampling",),
        "audit": ("audit",),
        "mode_weight_visibility_calibration": ("mode_weight_visibility_calibration",),
        "mode_weight_schedule_calibration": ("mode_weight_schedule_calibration",),
        "mode_weight_risk_profiles": ("mode_weight_risk_profiles",),
        "mode_weight_trajectories": ("mode_weight_trajectories",),
        "mode_weight_run_summaries": ("mode_weight_run_summaries",),
        "mode_weight_half_times": ("mode_weight_half_times",),
    }
    for key, candidates in aliases.items():
        out[key] = None
        for name in candidates:
            try:
                out[key] = read_table(name, results_dir)
                break
            except FileNotFoundError:
                continue
    return out


def _rho_curves(profiles: pd.DataFrame, summary: pd.DataFrame, w: float,
                law: str, param: float, exclude: float = 0.01):
    """Reconstruct ``rho_{N,m}(lambda)`` from the stored profile and summary."""
    prof = _sel(profiles, law=law, param=param, w=w, grid="I")
    summ = _sel(summary, law=law, param=param, w=w)
    lut = {(int(r.N), int(r.m)): r.logU for r in summ.itertuples()}
    for (N, m), g in prof.groupby(["N", "m"]):
        g = g.sort_values("lambda")
        lam = g["lambda"].to_numpy()
        keep = np.abs(lam - w) >= exclude
        if not keep.any():
            continue
        kl = np.array(
            [
                v * math.log(v / l) + (1 - v) * math.log((1 - v) / (1 - l))
                for v, l in zip(np.full(keep.sum(), w), lam[keep])
            ]
        )
        rho = np.exp(g.logD.to_numpy()[keep] - lut[(int(N), int(m))] - np.log(kl))
        yield int(N), int(m), lam[keep], rho


# ==========================================================================
# Mode-blindness and recovery summary
# ==========================================================================


def figure_mode_blindness_and_recovery(
    art: dict, out_dir: Path, png: bool = False
) -> None:
    summary, fits, profiles = art["summary"], art["fits"], art["profiles"]
    boost = art["boost"]
    law, param = MAIN_LAW
    w = MAIN_W

    sub = _sel(summary, law=law, param=param, w=w).sort_values(["N", "m"])
    all_N = sorted(sub.N.unique())
    bands = band_constants(w)
    # ---------------------------------------------------------------- (a)
    null = _sel(summary, law=NULL_LAW[0], param=NULL_LAW[1], w=w)

    def draw_a(ax: Axes) -> None:
        # Matched visible fractions turn the formerly coincident exponential
        # curves into separated groups.  Each unit of bar height is one order
        # of magnitude of blindness, so zero is a meaningful baseline.
        fractions = (0.25, 0.50, 0.75)
        x = np.arange(len(fractions), dtype=float)
        width = 0.78 / len(all_N)
        maximum = 0.0
        for index, N in enumerate(all_N):
            group = sub[sub.N == N].sort_values("m")
            rows = []
            for fraction in fractions:
                target = int(math.floor(fraction * int(N)))
                rows.append(group.iloc[int(np.argmin(np.abs(group.m.to_numpy() - target)))])
            heights = -np.asarray([row.logDelta for row in rows]) / math.log(10.0)
            positions = x + (index - 0.5 * (len(all_N) - 1)) * width
            color = n_color(int(N), all_N)
            ax.bar(
                positions, heights, width=width * 0.88, color=color,
                alpha=1.0 if int(N) == int(all_N[-1]) else 0.76,
                edgecolor=SURFACE, linewidth=0.35,
                label=rf"$N={int(N)}$", zorder=2,
            )
            pc_N = ProductConstants(param, int(N))
            anchor = float(group[group.m == 0].logDelta.iloc[0])
            predicted = -np.asarray(
                [anchor - pc_N.rate_star * float(row.m) for row in rows]
            ) / math.log(10.0)
            # A white under-stroke keeps the prediction tick legible on every
            # shade of the violet bar ramp; the dark core preserves the
            # achromatic "reference" encoding used throughout the figures.
            ax.scatter(positions, predicted, marker="_", s=58, linewidths=3.0,
                       color=SURFACE, zorder=4)
            ax.scatter(positions, predicted, marker="_", s=58, linewidths=1.25,
                       color=INK, zorder=5)
            maximum = max(maximum, float(heights.max()))

        null_color = NULL_GREY
        null_reference = null[null["N"] == null["N"].max()]
        if len(null_reference):
            null_level = -float(null_reference.logDelta.median()) / math.log(10.0)
            ax.axhline(null_level, color=null_color, lw=1.65,
                       ls=(0, (1.2, 1.25)), zorder=1)

        ax.set_xticks(x)
        ax.set_xticklabels([r"$m/N=1/4$", r"$m/N=1/2$", r"$m/N=3/4$"])
        ax.set_ylabel(r"blindness $-\log_{10}\Delta_{N,m}$")
        ax.set_ylim(0.0, maximum * 1.08)
        ax.grid(axis="x", visible=False)
        handles, labels = ax.get_legend_handles_labels()
        handles += [
            Line2D([], [], color=INK, marker="_", markersize=7, lw=0),
            Line2D([], [], color=null_color, lw=1.65, ls=(0, (1.2, 1.25))),
        ]
        labels += ["predicted rate", "null control"]
        ax.legend(handles, labels, loc="upper left", ncol=2, fontsize=6.0,
                  columnspacing=0.8, handlelength=1.1, labelspacing=0.25)

    # ---------------------------------------------------------------- (b)
    curves = list(_rho_curves(profiles, summary, w, law, param))
    rho_lo = min(float(r.min()) for _, _, _, r in curves)
    rho_hi = max(float(r.max()) for _, _, _, r in curves)
    rho_frame = pd.concat(
        [pd.DataFrame({"lambda": lam, "rho": rho}) for _, _, lam, rho in curves],
        ignore_index=True,
    )
    rho_envelope = (
        rho_frame.groupby("lambda")["rho"]
        .agg(low="min", median="median", high="max").reset_index()
    )

    def draw_b(ax: Axes) -> None:
        ax.axhline(bands["rho_limit"], color=CAT[1], lw=1.0, ls=(0, (4, 2)),
                   alpha=0.85, zorder=1)
        ax.fill_between(
            rho_envelope["lambda"], rho_envelope["low"], rho_envelope["high"],
            color=CAT[0], alpha=0.16, linewidth=0, zorder=2,
        )
        ax.plot(rho_envelope["lambda"], rho_envelope["median"],
                color=CAT[0], lw=1.45, zorder=3)
        ax.plot([w], [bands["rho_limit"]], marker="*", ms=10, color=CAT[1],
                zorder=5)

        # The measured range spans a factor of 1.1, where a log axis only buys
        # unreadable "4.5 x 10^0" tick labels.
        span = rho_hi - rho_lo
        ax.set_ylim(rho_lo - 0.35 * span, rho_hi + 0.35 * span)

        ax.annotate(rf"exact limit $\rho(w)={bands['rho_limit']:.0f}$",
                    xy=(w, bands["rho_limit"]),
                    xytext=(0, 8), textcoords="offset points",
                    ha="center", va="bottom", color=CAT[1], fontsize=6.8)
        corner_note(ax, f"median and range across {len(curves)} cells",
                    xy=(0.97, 0.04), ha="right", va="bottom")

        ax.set_xlabel(r"model mode weight $\lambda$")
        ax.set_ylabel(r"$\rho_{N,m}(\lambda)$" if M_COMPACT
                      else r"$\rho_{N,m}(\lambda)=\mathfrak{D}/(U\cdot\mathrm{kl})$")
        # Lem. 4.1 states c_I * U * kl <= D <= C_I * U * kl.  Its two bounds are
        # not constants -- U*kl moves over 174 decades across these cells -- so
        # the inequality is divided through by U*kl > 0, which turns it into a
        # statement about a dimensionless ratio lying between two fixed levels.
        # That normalisation is what lets all 49 cells be checked in one axes.

    # ---------------------------------------------------------------- (c)
    eps_c = 1e-4
    g0 = None
    if _has(boost):
        N_c = int(max(all_N))
        g0 = _sel(boost, law=law, param=param, w=w, N=N_c, eps=eps_c)
        m_base_c = int(sorted(g0.m_base.unique())[-1])
        g0 = _sel(g0, m_base=m_base_c)

    def draw_c(ax: Axes) -> None:
        if g0 is None:
            ax.text(0.5, 0.5, "intervention artifacts not found", ha="center", va="center",
                    color=MUTED, transform=ax.transAxes)
            ax.set_axis_off()
            return
        schedule_handles = []
        schedule_values = [0, 1, 4]
        for j, s in enumerate(schedule_values):
            gs = g0[(g0.s == s) & (g0.pi > 0)].sort_values("pi")
            if gs.empty:
                continue
            free, cens = gs[~gs.censored], gs[gs.censored]
            lbl = rf"$s={s}$"
            ax.plot(free.pi, free.r_eps, color=CAT[j], marker=MARKERS[j],
                    ms=4.0, lw=1.3, label=lbl, zorder=3)
            schedule_handles.append(
                Line2D([], [], color=CAT[j], marker=MARKERS[j], ms=3.8,
                       lw=1.2, label=lbl)
            )
            if not cens.empty:
                ax.plot(cens.pi, cens.r_eps, color=CAT[j], marker=MARKERS[j],
                        ms=4.6, lw=0, mfc="none", mew=1.0, zorder=3)
            ax.plot(gs.pi, gs.r_eps, color=CAT[j], lw=0.7, alpha=0.35, zorder=2)

        ref = g0[(g0.s == 0) & (~g0.censored) & (g0.pi > 0)].sort_values("pi")
        if not ref.empty:
            p0, r0 = float(ref.pi.iloc[-1]), float(ref.r_eps.iloc[-1])
            pp = np.array([float(ref.pi.min()) * 0.6, p0 * 1.6])
            ax.plot(pp, r0 * (pp / p0) ** -0.5, color=INK, lw=1.0,
                    ls=(0, (4, 2)), alpha=0.65, zorder=1)
            ax.annotate("slope $-1/2$", xy=(p0 * 0.62, r0 * 0.60), color=INK,
                        fontsize=7.0, rotation=-19, ha="center")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"mass $\pi_s(\mu)$" if M_COMPACT
                      else r"low-visibility mass $\pi_s(\mu)$")
        ax.set_ylabel(r"recovery radius $\varepsilon=10^{-4}$")

        # The censored cells all sit at r = c0/2, the half-width of I: there the
        # discrepancy is below eps across the *whole* admissible interval, so no
        # mode weight in I is distinguishable from w.  That is mode blindness in
        # This is the intervention theorem's language, not a failed measurement.
        # is drawn and named rather than left as an unexplained plateau.
        a, b = mode_weight_interval(w)
        limit_color = CAT[len(schedule_values) % len(CAT)]
        ax.axhline(b - w, color=limit_color, lw=0.7,
                   ls=(0, (1, 2)), zorder=0)
        ax.set_ylim(top=(b - w) * 1.75)
        positive_pi = g0.loc[g0.pi > 0, "pi"].astype(float)
        ax.set_xlim(float(positive_pi.min()) / 1.5, float(positive_pi.max()) * 2.0)

        semantic_handles = [
            Line2D([], [], color=MUTED, marker="o", mfc="none", mew=1.0,
                   lw=0, ms=4.2, label="hollow: censored"),
            Line2D([], [], color=limit_color, lw=0.8, ls=(0, (1, 2)),
                   label="dotted: indistinguishability limit"),
        ]
        ax.legend(handles=schedule_handles + semantic_handles, loc="lower left",
                  ncol=1, fontsize=5.8, labelspacing=0.25,
                  columnspacing=0.8, handlelength=1.5, borderpad=0.3)
        # a single decade of major ticks reads as an unlabelled axis
        ax.yaxis.set_minor_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:g}" if v in (0.02, 0.05, 0.2) else "")
        )
        if not M_COMPACT:
            corner_note(ax, rf"$N={N_c}$, $m_{{\rm base}}={m_base_c}$",
                        xy=(0.98, 0.97), va="top")

    # one size for all three, so they can be re-laid-out as subfigures
    panels = [
        Panel("blindness_scaling", r"Blindness scaling ($\theta=0.8$)", draw_a,
              M_PANEL, M_TITLE_SIZE),
        Panel("sensitivity_ratio", f"Sensitivity ratio ({len(curves)} cells)", draw_b,
              M_PANEL, M_TITLE_SIZE),
        Panel("low_visibility_intervention", "Intervention", draw_c, M_PANEL, M_TITLE_SIZE),
    ]
    render("mode_blindness_and_recovery", panels, _grid(1, 3),
           (3 * M_PANEL[0], M_PANEL[1] + 0.2),
           out_dir, png, tight_kw=dict(pad=0.9, w_pad=1.8))


# ==========================================================================
# Mode-blindness and recovery summary table
# ==========================================================================


def table_mode_blindness_and_recovery(art: dict, out_dir: Path) -> pd.DataFrame:
    """Measured against predicted -- the right-hand column earns the space."""
    rows = []
    fits, summary, uscale = art["fits"], art["summary"], art["uscale"]

    for (law, param, w), _ in fits.groupby(["law", "param", "w"]):
        if law != "product":
            continue
        for N in sorted(_sel(fits, law=law, param=param, w=w).N.dropna().unique()):
            pc = ProductConstants(param, int(N))
            f = _sel(fits, law=law, param=param, w=w, N=N, xvar="m")
            fd = f[f.target == "Delta"]
            fu = f[f.target == "U"]
            us = _sel(uscale, law=law, param=param, w=w, N=int(N)) if _has(uscale) else None
            sm = _sel(summary, law=law, param=param, w=w, N=int(N))
            bands = band_constants(w)
            rows.append(
                {
                    "law": law, "theta": param, "N": int(N), "w": w,
                    "rate_hat_Delta": float(fd.rate_hat.iloc[0]) if len(fd) else np.nan,
                    "rate_hat_U": float(fu.rate_hat.iloc[0]) if len(fu) else np.nan,
                    "rate_star": pc.rate_star,
                    "rel_err_pct": (
                        100 * abs(float(fd.rate_hat.iloc[0]) - pc.rate_star) / pc.rate_star
                        if len(fd) else np.nan
                    ),
                    "num_ci_lo": float(fd.rate_ci_lo.iloc[0]) if len(fd) else np.nan,
                    "num_ci_hi": float(fd.rate_ci_hi.iloc[0]) if len(fd) else np.nan,
                    "c_pred": pc.c, "L_pred": pc.L,
                    "envelope_ok": (
                        bool(pc.c <= float(fd.rate_hat.iloc[0]) <= pc.L) if len(fd) else None
                    ),
                    "u0_hat": float(us.u0_hat.iloc[0]) if _has(us) else np.nan,
                    "u0_pred": pc.u0,
                    "L_hat": float(us.L_hat.iloc[0]) if _has(us) else np.nan,
                    "L_hat_pred": pc.L,
                    "rho_min": float(sm.rho_min.min()),
                    "rho_max": float(sm.rho_max.max()),
                    "c_I": bands["c_I"], "C_I": bands["C_I"],
                    "rho_band_ok": bool(
                        sm.rho_min.min() >= bands["c_I"] and sm.rho_max.max() <= bands["C_I"]
                    ),
                    "s_pin": pc.s_pin, "s_ov": pc.s_ov,
                    "window_empty": pc.window_empty,
                }
            )
    tab = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    tab.to_csv(out_dir / "table_mode_blindness_and_recovery.csv", index=False)
    (out_dir / "table_mode_blindness_and_recovery.tex").write_text(
        _mode_blindness_and_recovery_table_tex(tab), encoding="utf-8"
    )
    print(
        "  wrote table_mode_blindness_and_recovery.csv / "
        "table_mode_blindness_and_recovery.tex"
    )
    return tab


def _mode_blindness_and_recovery_table_tex(tab: pd.DataFrame) -> str:
    lines = [
        r"% Mode-blindness summary -- measured vs predicted (Law P, w = 1/2).",
        r"% Uncertainties are NUMERICAL (fit window / KL scheme / precision),",
        r"% not statistical: the exact studies contain no sampling noise.",
        r"\begin{tabular}{llrrrrl}",
        r"\toprule",
        r"$\theta$ & $N$ & $\hat c_N$ & $\mathrm{rate}^\star$ & rel.\ err. & "
        r"$[\hat u_0,\hat L]$ & $c\le\hat c_N\le L$ \\",
        r"\midrule",
    ]
    # The table body is the symmetric case; the rho band depends on w through
    # I = [c0/2, 1-c0/2], so the summary line must use the same subset.
    sym = tab[tab.w == 0.5]
    for r in sym.itertuples():
        tick = r"\checkmark" if r.envelope_ok else r"$\times$"
        lines.append(
            f"{r.theta:g} & {r.N} & {r.rate_hat_Delta:.5f} & {r.rate_star:.5f} & "
            f"{r.rel_err_pct:.2f}\\% & "
            f"[{r.u0_hat:.3f}, {r.L_hat:.3f}] & {tick} \\\\"
        )
    if len(sym):
        r0 = sym.iloc[0]
        lines += [
            r"\midrule",
            rf"\multicolumn{{7}}{{l}}{{$\rho$ over all cells at $w=1/2$: "
            rf"$[{sym.rho_min.min():.3f},\,{sym.rho_max.max():.3f}]"
            rf"\subseteq[c_I,C_I]=[{r0.c_I:.3f},\,{r0.C_I:.0f}]$}} \\",
            rf"\multicolumn{{7}}{{l}}{{predicted $u_0=1/4$, "
            rf"$L=L_0+\eta_N$, $c=\theta^2/16$}} \\",
        ]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# ==========================================================================
# Supporting figures
# ==========================================================================


def figure_masked_discrepancy_profiles(
    art: dict, out_dir: Path, png: bool = False
) -> None:
    """Profiles: rows = N, cols = regime (pinned | null), + a |w - lambda| strip."""
    profiles, summary = art["profiles"], art["summary"]
    w = MAIN_W
    regimes = [("pinned", MAIN_LAW), ("null", NULL_LAW)]
    Ns = [int(n) for n in
          sorted(_sel(summary, law=MAIN_LAW[0], param=MAIN_LAW[1], w=w).N.unique())]
    cmap = m_cmap()

    def make_profile(N: int, law: str, param: float, show_legend: bool):
        def draw(ax: Axes) -> None:
            prof = _sel(profiles, law=law, param=param, w=w, N=N, grid="I")
            if prof.empty:
                ax.set_axis_off()
                return
            ms = sorted({m for m in (4, 16, N // 4, N // 2, (3 * N) // 4)
                         if m in set(prof.m.unique())})
            lo_v, hi_v = np.inf, -np.inf
            for color_index, m in enumerate(ms):
                g = prof[prof.m == m].sort_values("lambda")
                # D(w) = 0 exactly (test T8); keeping that point would stretch
                # the log axis by 300 decades and hide everything else.
                keep = g["lambda"].to_numpy() != w
                vals = np.exp(g.logD.to_numpy()[keep])
                lo_v, hi_v = min(lo_v, vals.min()), max(hi_v, vals.max())
                ax.plot(g["lambda"].to_numpy()[keep], vals,
                        color=cmap.colors[color_index % cmap.N], lw=1.1,
                        label=f"$m={m}$")
            ax.set_ylim(lo_v * 0.25, hi_v * 6.0)
            ax.axvline(w, color=MUTED, lw=0.7, ls=(0, (1, 2)))
            a, b = float(prof["lambda"].min()), float(prof["lambda"].max())
            for edge in (a, b):
                ax.axvspan(edge - 0.012, edge + 0.012, color=GRID, alpha=0.7, lw=0)
            ax.set_yscale("log")
            ax.set_xlabel(r"$\lambda$")
            ax.set_ylabel(r"$\mathfrak{D}_{N,m}(\lambda)$")
            if show_legend:
                ax.legend(fontsize=6.2, ncol=2, loc="lower center",
                          columnspacing=0.8, handlelength=1.2, frameon=True,
                          framealpha=0.88, edgecolor="none", facecolor=SURFACE)

        return draw

    def make_strip(law: str, param: float):
        def draw(ax: Axes) -> None:
            lam = np.linspace(*_sel(profiles, law=law, param=param, w=w, grid="I")
                              ["lambda"].agg(["min", "max"]).values, 201)
            ax.plot(lam, np.abs(w - lam), color=CAT[0], lw=1.4)
            ax.axvline(w, color=MUTED, lw=0.7, ls=(0, (1, 2)))
            ax.set_xlabel(r"$\lambda$")
            ax.set_ylabel(r"$D_{\rm TV}$", fontsize=7.5)

        return draw

    panels: list[Panel] = []
    for N in Ns:
        for regime, (law, param) in regimes:
            panels.append(Panel(f"profile_N{N}_{regime}", f"$N={N}$, {regime}",
                                make_profile(N, law, param, regime == "pinned"),
                                (3.7, 2.6), title_weight="normal",
                                show_letter=False))
    for regime, (law, param) in regimes:
        panels.append(Panel(f"total_variation_{regime}",
                            f"total variation, {regime}",
                            make_strip(law, param), (3.7, 2.6),
                            title_weight="normal", show_letter=False))

    def compose(fig):
        gs = fig.add_gridspec(len(Ns) + 1, 2,
                              height_ratios=[1.0] * len(Ns) + [0.40],
                              hspace=0.72, wspace=0.32)
        axes = [fig.add_subplot(gs[row, col])
                for row in range(len(Ns)) for col in range(2)]
        axes += [fig.add_subplot(gs[len(Ns), col]) for col in range(2)]
        return axes
    render(
        "masked_discrepancy_profiles",
        panels,
        compose,
        (7.0, 1.85 * len(Ns) + 1.5),
        out_dir,
        png,
    )
def figure_blindness_decay_rates(art: dict, out_dir: Path, png: bool = False) -> None:
    """Rates: ``c_hat_N`` vs ``N`` with numerical CIs, and fixed-fraction decay."""
    fits, summary = art["fits"], art["summary"]
    w = MAIN_W
    laws = [("product", 0.6), ("product", 0.8), ("cw", 1.5), ("cw", 0.5)]

    def draw_a(ax: Axes) -> None:
        for j, (law, param) in enumerate(laws):
            g = _sel(fits, law=law, param=param, w=w, target="Delta",
                     xvar="m").sort_values("N")
            if g.empty:
                continue
            pinned = not (law == "cw" and param == 0.5)
            st = law_style(law, param, pinned)
            yerr = np.vstack([
                (g.rate_hat - g.rate_ci_lo).to_numpy(),
                (g.rate_ci_hi - g.rate_hat).to_numpy(),
            ])
            sym = r"$\theta$" if law == "product" else r"$\beta$"
            name = "" if law == "product" else "Curie-Weiss model, "
            ax.errorbar(g.N, g.rate_hat, yerr=yerr, marker=MARKERS[j], ms=4.2,
                        capsize=2.0, elinewidth=0.9, **st)
            label_row = g.iloc[-2] if len(g) > 1 else g.iloc[-1]
            ax.annotate(
                f"{name}{sym}$={param:g}$",
                xy=(float(label_row.N), float(label_row.rate_hat)),
                xytext=(4, 8), textcoords="offset points", ha="left", va="center",
                fontsize=6.5, color=st["color"],
            )
            ref = (-0.5 * math.log(1 - param**2)) if law == "product" else (
                (lambda ms: -0.5 * math.log(1 - ms**2) if ms > 0 else None)(
                    curie_weiss_magnetisation(param))
            )
            if ref is not None:
                ax.axhline(ref, color=st["color"], lw=0.8, ls=(0, (5, 2)), alpha=0.55)
        ax.set_xscale("log")
        ax.set_xticks(sorted(fits.N.dropna().unique()))
        ax.xaxis.set_minor_locator(plt.NullLocator())   # minor log ticks collide
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("$N$")
        ax.set_ylabel(r"fitted decay rate $\hat c_N$")
        ax.margins(x=0.16)
        corner_note(ax, "dashed: reference rate",
                    xy=(0.03, 0.97), ha="left", va="top")

    ff = _sel(fits, law=MAIN_LAW[0], param=MAIN_LAW[1], w=w, target="Delta_frac")
    sub = _sel(summary, law=MAIN_LAW[0], param=MAIN_LAW[1], w=w)

    def draw_b(ax: Axes) -> None:
        for j, (frac, g) in enumerate(ff.groupby("frac")):
            pts = []
            for N in sorted(sub.N.unique()):
                r = sub[(sub.N == N) & (sub.m == int(math.floor(frac * N)))]
                if not r.empty:
                    pts.append((float(N), float(r.logDelta.iloc[0])))
            if pts:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color=CAT[j % len(CAT)], marker=MARKERS[j], ms=3.8,
                        lw=0)
            xs = np.array([g.fit_lo.iloc[0], g.fit_hi.iloc[0]])
            ax.plot(xs, g.intercept.iloc[0] + g.slope.iloc[0] * xs,
                    color=CAT[j % len(CAT)], lw=1.2)
            x_label = float(pts[-2][0]) if len(pts) > 1 else float(np.mean(xs))
            y_label = float(g.intercept.iloc[0] + g.slope.iloc[0] * x_label)
            ax.annotate(rf"$\varrho={frac:g}$", xy=(x_label, y_label),
                        xytext=(8, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=6.6,
                        color=CAT[j % len(CAT)])
        ax.set_xlabel("$N$")
        ax.set_ylabel(r"$\log\Delta_{N,\lfloor\varrho N\rfloor}$")
        ax.margins(x=0.14)
    panels = [
        Panel("measured_vs_predicted_rate", "measured rate vs prediction", draw_a,
              bumped(5.0, 3.3)),
        Panel("fixed_visible_fraction", "decay at fixed visible fraction", draw_b,
              bumped(4.0, 3.3)),
    ]
    render("blindness_decay_rates", panels,
           _grid(1, 2, gridspec_kw={"width_ratios": [1.55, 1.0]}),
           (8.4, 3.3), out_dir, png, tight_kw={})
def figure_low_visibility_intervention(
    art: dict, out_dir: Path, png: bool = False
) -> None:
    """Intervention, full: curvature, radius vs tolerance, boosted profiles."""
    boost, profiles = art["boost"], art["profiles"]
    if not _has(boost):
        return
    law, param, w = MAIN_LAW[0], MAIN_LAW[1], MAIN_W
    N = int(max(boost.N.unique()))
    g_all = _sel(boost, law=law, param=param, w=w, N=N)
    m_base = int(sorted(g_all.m_base.unique())[-1])
    g_all = _sel(g_all, m_base=m_base)
    note = rf"$N={N}$,  $m_{{\rm base}}={m_base}$"

    def draw_a(ax: Axes) -> None:
        for j, s in enumerate(sorted(g_all.s.unique())):
            g = g_all[(g_all.s == s) & (g_all.pi > 0)
                      & (g_all.eps == g_all.eps.min())].sort_values("pi")
            ax.plot(g.pi, g.kappa_exact, color=CAT[j % len(CAT)], lw=1.3,
                    zorder=2)
            ax.plot(g.pi, g.kappa_fit, color=CAT[j % len(CAT)], lw=0,
                    marker=MARKERS[j], ms=4.2, mfc="none", zorder=3)
            ax.annotate(rf"$s={s}$", xy=(float(g.pi.iloc[-1]),
                        float(g.kappa_exact.iloc[-1])), xytext=(4, 0),
                        textcoords="offset points", fontsize=6.5,
                        color=CAT[j % len(CAT)], ha="left", va="center",
                        clip_on=False)
        ref = g_all[(g_all.s == 0) & (g_all.pi > 0)].sort_values("pi")
        if not ref.empty:
            p0, k0 = float(ref.pi.iloc[-1]), float(ref.kappa_exact.iloc[-1])
            pp = np.array([float(ref.pi.min()), p0])
            ax.plot(pp, k0 * (pp / p0), color=INK, lw=0.9, ls=(0, (4, 2)), alpha=0.6)
            ax.annotate("slope 1", xy=(0.08, 0.80), xycoords="axes fraction",
                        fontsize=6.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\pi_s(\mu)$")
        ax.set_ylabel(r"$\varkappa(\pi,s)$")
        ax.margins(x=0.13)
        corner_note(ax, "lines: exact; open markers: numerical fit",
                    xy=(0.03, 0.97), ha="left", va="top")

    def draw_b(ax: Axes) -> None:
        for j, s in enumerate(sorted(g_all.s.unique())):
            g = g_all[(g_all.s == s) & (np.isclose(g_all.pi, 1e-2))
                      & (~g_all.censored)].sort_values("eps")
            if g.empty:
                continue
            ax.plot(g.eps, g.r_eps, color=CAT[j % len(CAT)], marker=MARKERS[j],
                    ms=3.8, markevery=max(1, len(g) // 4), lw=1.3)
            ax.annotate(rf"$s={s}$", xy=(float(g.eps.iloc[-1]),
                        float(g.r_eps.iloc[-1])), xytext=(4, 0),
                        textcoords="offset points", fontsize=6.5,
                        color=CAT[j % len(CAT)], ha="left", va="center",
                        clip_on=False)
        g = g_all[(g_all.s == 0) & (np.isclose(g_all.pi, 1e-2)) & (~g_all.censored)]
        if not g.empty:
            e0 = float(g.eps.max())
            r0 = float(g[g.eps == e0].r_eps.iloc[0])
            ee = np.array([float(g.eps.min()), e0])
            ax.plot(ee, r0 * (ee / e0) ** 0.5, color=INK, lw=0.9, ls=(0, (4, 2)),
                    alpha=0.6)
            ax.annotate("slope $+1/2$", xy=(0.06, 0.82), xycoords="axes fraction",
                        fontsize=6.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_ylabel(r"$r(\varepsilon)$")
        ax.margins(x=0.13)
        corner_note(ax, note, xy=(0.03, 0.97), ha="left", va="top")

    def draw_c(ax: Axes) -> None:
        prof = _sel(profiles, law=law, param=param, w=w, N=N, grid="I")
        base = prof[prof.m == m_base].sort_values("lambda")
        lam = base["lambda"].to_numpy()
        keep = lam != w  # D(w) = 0 exactly; see masked-discrepancy profiles.
        base_log = base.logD.to_numpy()[keep]
        pi = 1e-2
        gain_curves = []
        for j, s in enumerate([0, 1, 4]):
            bs = prof[prof.m == s].sort_values("lambda")
            if bs.empty:
                continue
            mixed = np.logaddexp(math.log1p(-pi) + base.logD.to_numpy(),
                                 math.log(pi) + bs.logD.to_numpy())
            gain_orders = (mixed[keep] - base_log) / math.log(10.0)
            gain_curves.append(gain_orders)
            ax.plot(lam[keep], gain_orders, color=CAT[j], lw=1.25,
                    ls=[(0, ()), (0, (5, 1.5)), (0, (1, 1.4))][j])
            ax.annotate(rf"$s={s}$", xy=(float(lam[keep][-1]),
                        float(gain_orders[-1])), xytext=(4, (j - 1) * 5),
                        textcoords="offset points", fontsize=6.5,
                        color=CAT[j], ha="left", va="center", clip_on=False)
        if gain_curves:
            all_gain = np.concatenate(gain_curves)
            spread = float(np.ptp(all_gain))
            padding = max(0.2, 0.15 * spread)
            ax.set_ylim(float(all_gain.min()) - padding,
                        float(all_gain.max()) + padding)
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"restoration $\log_{10}(\mathfrak{D}_{\rm boost}/\mathfrak{D}_{\rm base})$")
        ax.margins(x=0.10, y=0.12)
        corner_note(ax, note + r", $\pi=10^{-2}$",
                    xy=(0.97, 0.05), ha="right", va="bottom")

    A3_PANEL = bumped(3.8, 3.1)
    panels = [
        Panel("curvature", "curvature", draw_a, A3_PANEL, M_TITLE_SIZE),
        Panel("recovery_radius", r"radius vs tolerance ($\pi=10^{-2}$)",
              draw_b, A3_PANEL, M_TITLE_SIZE),
        Panel("restored_discrepancy", r"restoration over the blind baseline",
              draw_c, A3_PANEL, M_TITLE_SIZE),
    ]
    render("low_visibility_intervention", panels, _grid(1, 3), (9.4, 3.1),
           out_dir, png, tight_kw={})


def figure_residual_uncertainty_by_visibility(
    art: dict, out_dir: Path, png: bool = False
) -> None:
    """Low-visibility uncertainty: ``log U_{N,s}`` vs ``s`` with the exact anchor."""
    uscale = art["uscale"]
    if not _has(uscale):
        return
    law, param, w = MAIN_LAW[0], MAIN_LAW[1], MAIN_W
    g_all = _sel(uscale, law=law, param=param, w=w)
    Ns = sorted(g_all.N.unique())

    def draw_a(ax: Axes) -> None:
        envelope = (
            g_all.groupby("s")["logU"]
            .agg(low="min", median="median", high="max").reset_index()
        )
        ax.fill_between(envelope.s, envelope.low, envelope.high,
                        color=CAT[0], alpha=0.18, linewidth=0)
        ax.plot(envelope.s, envelope["median"], color=CAT[0], marker="o",
                ms=4.0, lw=1.4)

        anchor = math.log(w * (1 - w))
        ax.plot([0], [anchor], marker="*", ms=8, color=CAT[1], zorder=5)
        ax.annotate("exact $s=0$ anchor", xy=(0, anchor), xytext=(7, -4),
                    textcoords="offset points", fontsize=6.6,
                    color=CAT[1], va="top")

        pc = ProductConstants(param, int(Ns[-1]))
        ss = np.linspace(0, float(g_all.s.max()), 50)
        ax.plot(ss, math.log(pc.u0) - pc.L * ss, color=INK, lw=1.1,
                ls=(0, (6, 2, 1, 2)), alpha=0.75)
        ax.annotate("certified envelope", xy=(float(ss[-1]),
                    float(math.log(pc.u0) - pc.L * ss[-1])), xytext=(-4, 4),
                    textcoords="offset points", ha="right", va="bottom",
                    fontsize=6.6, color=INK)
        ax.set_xlabel("$s$ (visible coordinates under the boost)")
        ax.set_ylabel(r"$\log U_{N,s}$")
        corner_note(ax, rf"median and range, $N={int(Ns[0])}$--${int(Ns[-1])}$",
                    xy=(0.97, 0.97), va="top")
    panels = [Panel("residual_mode_uncertainty", "low-visibility residual mode uncertainty", draw_a,
                    bumped(5.2, 3.5), M_TITLE_SIZE)]
    render("residual_uncertainty_by_visibility", panels, _grid(1, 1),
           bumped(5.2, 3.5), out_dir, png, tight_kw={})
def figure_boundary_flow_mechanism(
    art: dict, out_dir: Path, png: bool = False
) -> None:
    """Cross-mode flows alongside the blindness scaling, and under boosting."""
    summary, boost = art["summary"], art["boost"]
    law, param, w = MAIN_LAW[0], MAIN_LAW[1], MAIN_W
    sub = _sel(summary, law=law, param=param, w=w).sort_values(["N", "m"])
    Ns = sorted(sub.N.unique())

    def draw_a(ax: Axes) -> None:
        fractions = (0.25, 0.50, 0.75)
        x = np.arange(len(fractions), dtype=float)
        width = 0.78 / len(Ns)
        minimum = 0.0
        for index, N in enumerate(Ns):
            group = sub[sub.N == N].sort_values("m")
            rows = []
            for fraction in fractions:
                target = int(math.floor(fraction * int(N)))
                rows.append(group.iloc[int(np.argmin(np.abs(group.m.to_numpy() - target)))])
            values = np.asarray([row.logFplus for row in rows], dtype=float)
            positions = x + (index - 0.5 * (len(Ns) - 1)) * width
            ax.bar(positions, values, width=width * 0.88,
                   color=n_color(int(N), Ns), edgecolor=SURFACE, linewidth=0.35,
                   label=rf"$N={int(N)}$", zorder=2)
            minimum = min(minimum, float(values.min()))
        ax.set_xticks(x)
        ax.set_xticklabels([r"$m/N=1/4$", r"$m/N=1/2$", r"$m/N=3/4$"])
        ax.set_ylabel(r"$\log F^+_{N,m}$ (nats)")
        ax.set_ylim(minimum * 1.08, 0.0)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="lower left", ncol=2, fontsize=6.1,
                  columnspacing=0.8, handlelength=1.1)

        gap = _sel(sub, N=int(Ns[-1])).sort_values("m").copy()
        gap["flow_gap"] = gap.logFplus - gap.logDelta
        gap_rows = []
        for fraction in fractions:
            target = int(math.floor(fraction * int(Ns[-1])))
            gap_rows.append(gap.iloc[int(np.argmin(np.abs(gap.m.to_numpy() - target)))])
        inset = ax.inset_axes([0.055, 0.26, 0.42, 0.30])
        inset.bar(np.arange(len(fractions)), [row.flow_gap for row in gap_rows],
                  width=0.62, color=CAT[0], alpha=0.85)
        inset.set_xticks(np.arange(len(fractions)))
        inset.set_xticklabels([r"$1/4$", r"$1/2$", r"$3/4$"])
        inset.tick_params(labelsize=5.8)
        inset.set_title(
            rf"$\log(F^+/\Delta)$ gap at $N={int(Ns[-1])}$",
            fontsize=5.5,
            pad=1,
        )
        inset.grid(axis="x", visible=False)

    have_boost = _has(boost)
    if have_boost:
        N_b = int(max(boost.N.unique()))
        gb = _sel(boost, law=law, param=param, w=w, N=N_b)
        m_base_b = int(sorted(gb.m_base.unique())[-1])
        gb = _sel(gb, m_base=m_base_b, eps=float(gb.eps.min()))

    def draw_b(ax: Axes) -> None:
        if not have_boost:
            ax.text(0.5, 0.5, "intervention artifacts not found", ha="center", va="center",
                    color=MUTED, transform=ax.transAxes)
            ax.set_axis_off()
            return
        pi_values = sorted(float(value) for value in gb.loc[gb.pi > 0, "pi"].unique())
        s_values = sorted(int(value) for value in gb.s.unique())
        x = np.arange(len(pi_values), dtype=float)
        width = 0.78 / len(s_values)
        minimum = 0.0
        for index, s in enumerate(s_values):
            group = gb[(gb.s == s) & (gb.pi > 0)].sort_values("pi")
            values = group.logFplus_boost.to_numpy(dtype=float) / math.log(10.0)
            positions = x + (index - 0.5 * (len(s_values) - 1)) * width
            ax.bar(positions, values, width=width * 0.88,
                   color=CAT[index % len(CAT)], edgecolor=SURFACE, linewidth=0.35,
                   label=rf"$s={s}$", zorder=2)
            minimum = min(minimum, float(values.min()))
        ax.set_xticks(x)
        ax.set_xticklabels([
            rf"$10^{{{int(round(math.log10(value)))}}}$" for value in pi_values
        ])
        ax.set_xlabel(r"$\pi_s(\mu)$")
        ax.set_ylabel(r"$\log_{10}F^+_{\rm boost}$")
        ax.set_ylim(minimum * 1.08, 0.0)
        ax.grid(axis="x", visible=False)
        ax.legend(loc="lower right", bbox_to_anchor=(0.99, 0.115),
                  borderaxespad=0.0, ncol=2, fontsize=6.2,
                  columnspacing=0.8, handlelength=1.1)
        corner_note(ax, rf"$N={N_b}$,  $m_{{\rm base}}={m_base_b}$",
                    xy=(0.97, 0.04), ha="right", va="bottom")
    panels = [
        Panel("conditional_flow", r"cross-mode flow and flow/discrepancy gap",
              draw_a, bumped(4.3, 3.2)),
        Panel("boosted_flow", "boosted flow", draw_b, bumped(4.3, 3.2)),
    ]
    render("boundary_flow_mechanism", panels, _grid(1, 2), (8.2, 3.2),
           out_dir, png, tight_kw={})
def figure_asymmetric_mixture_weights(
    art: dict, out_dir: Path, png: bool = False
) -> None:
    """Asymmetric case ``w = 0.9``: profiles and the (wider) sensitivity band."""
    profiles, summary = art["profiles"], art["summary"]
    w = 0.9
    if not len(_sel(summary, w=w)):
        return
    law, param = MAIN_LAW
    bands = band_constants(w)
    Ns = sorted(_sel(summary, law=law, param=param, w=w).N.unique())
    N = int(Ns[-1])

    def draw_a(ax: Axes) -> None:
        prof = _sel(profiles, law=law, param=param, w=w, N=N, grid="I")
        ms = sorted({4, 16, N // 4, N // 2, (3 * N) // 4} & set(prof.m.unique()))
        cmap = m_cmap()
        lo_v, hi_v = np.inf, -np.inf
        label_offsets = {ms[0]: 5, ms[1]: -5} if len(ms) > 1 else {}
        for color_index, m in enumerate(ms):
            g = prof[prof.m == m].sort_values("lambda")
            keep = g["lambda"].to_numpy() != w
            vals = np.exp(g.logD.to_numpy()[keep])
            lo_v, hi_v = min(lo_v, vals.min()), max(hi_v, vals.max())
            color = cmap.colors[color_index % cmap.N]
            ax.plot(g["lambda"].to_numpy()[keep], vals,
                    color=color, lw=1.2)
            ax.annotate(rf"$m={m}$", xy=(float(g["lambda"].to_numpy()[keep][-1]),
                        float(vals[-1])), xytext=(4, label_offsets.get(m, 0)),
                        textcoords="offset points",
                        fontsize=6.3, color=color, ha="left",
                        va="center", clip_on=False)
        ax.axvline(w, color=MUTED, lw=0.7, ls=(0, (1, 2)))
        ax.set_yscale("log")
        ax.set_ylim(lo_v * 0.25, hi_v * 6.0)
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"$\mathfrak{D}_{N,m}(\lambda)$")
        ax.margins(x=0.12)

    def draw_b(ax: Axes) -> None:
        curves = list(_rho_curves(profiles, summary, w, law, param))
        lo = min(float(r.min()) for _, _, _, r in curves)
        hi = max(float(r.max()) for _, _, _, r in curves)
        frame = pd.concat(
            [pd.DataFrame({"lambda": lam, "rho": rho})
             for _, _, lam, rho in curves], ignore_index=True,
        )
        envelope = (frame.groupby("lambda")["rho"]
                    .agg(low="min", median="median", high="max").reset_index())
        ax.axhline(bands["rho_limit"], color=CAT[1], lw=1.0, ls=(0, (4, 2)),
                   alpha=0.85, zorder=1)
        ax.fill_between(envelope["lambda"], envelope.low, envelope.high,
                        color=CAT[0], alpha=0.16, linewidth=0, zorder=2)
        ax.plot(envelope["lambda"], envelope["median"], color=CAT[0],
                lw=1.45, zorder=3)
        ax.plot([w], [bands["rho_limit"]], marker="*", ms=9, color=CAT[1], zorder=5)
        span = hi - lo
        ax.set_ylim(lo - 0.35 * span, hi + 0.35 * span)
        ax.annotate(rf"exact $\rho(w)={bands['rho_limit']:.1f}$",
                    xy=(w, bands["rho_limit"]), xytext=(-4, 7),
                    textcoords="offset points", fontsize=6.6, color=CAT[1],
                    ha="right", va="bottom")
        corner_note(ax, f"median and range across {len(curves)} cells",
                    xy=(0.04, 0.06), ha="left", va="bottom")
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"$\rho_{N,m}(\lambda)$")

    panels = [
        Panel("discrepancy_profiles", rf"profiles, $w=0.9$, $N={N}$", draw_a,
              bumped(4.3, 3.2)),
        Panel("sensitivity_ratio", rf"$\rho$ band, $\rho(w)={bands['rho_limit']:.1f}$", draw_b,
              bumped(4.3, 3.2)),
    ]
    render("asymmetric_mixture_weights", panels, _grid(1, 2), (8.4, 3.2),
           out_dir, png, tight_kw={})


def figure_numerical_stability_audit(
    art: dict, out_dir: Path, png: bool = False
) -> None:
    """Numerics audit: mpmath-vs-float64 error, and slope sensitivity."""
    audit, numerics, fits = art["audit"], art["numerics"], art["fits"]

    def draw_a(ax: Axes) -> None:
        if not _has(audit):
            ax.text(0.5, 0.5, "run scripts/audit_numerics.py", ha="center",
                    va="center", color=MUTED, transform=ax.transAxes)
            ax.set_axis_off()
            return
        for j, (impl, g) in enumerate(audit.groupby("impl")):
            lbl = ("branched, $T=30$" if impl == "branched" else "stable")
            ax.scatter(g.logD_mp, np.abs(g.abs_err_log).clip(1e-18),
                       s=15, color=CAT[j % len(CAT)], marker=MARKERS[j],
                       facecolors="none" if impl == "branched" else CAT[j % len(CAT)],
                       linewidths=0.8, alpha=0.8, label=lbl)
        ax.set_yscale("log")
        ax.set_xlabel(r"$\log\mathfrak{D}$ (mpmath, 50 digits)")
        ax.set_ylabel(r"$|\log\mathfrak{D}_{\rm float64}-\log\mathfrak{D}_{\rm mp}|$")
        ax.legend(loc="upper left")

    def draw_b(ax: Axes) -> None:
        if not _has(numerics):
            ax.set_axis_off()
            return
        g = numerics[numerics.target == "Delta"]
        knobs = list(dict.fromkeys(g.knob))
        base = fits[(fits.target == "Delta") & (fits.xvar == "m")]
        for i, knob in enumerate(knobs):
            merged = g[g.knob == knob].merge(
                base[["law", "param", "N", "w", "slope"]],
                on=["law", "param", "N", "w"], suffixes=("", "_base"),
            )
            spread = np.maximum(np.abs(merged.lo - merged.slope),
                                np.abs(merged.hi - merged.slope))
            values = np.abs(spread.to_numpy(dtype=float)).clip(1e-18)
            order = np.argsort(values)
            offsets = np.linspace(-0.16, 0.16, len(values)) if len(values) > 1 else np.array([0.0])
            jitter = np.empty(len(values))
            jitter[order] = offsets
            color = CAT[i % len(CAT)]
            ax.scatter(i + jitter, values, s=13, color=color, alpha=0.55,
                       edgecolors="none", zorder=2)
            if len(values):
                low, med, high = np.quantile(values, [0.1, 0.5, 0.9])
                ax.plot([i, i], [low, high], color=INK, lw=1.0, zorder=3)
                ax.plot([i - 0.10, i + 0.10], [med, med], color=INK,
                        lw=1.6, zorder=4)
        ax.set_xticks(range(len(knobs)))
        ax.set_xticklabels(knobs, rotation=20, ha="right", fontsize=6.8)
        ax.set_yscale("log")
        ax.set_ylabel(r"$|\Delta\hat c_N|$")

    panels = [
        Panel("float64_vs_high_precision", "float64 vs mpmath", draw_a,
              bumped(4.3, 3.2)),
        Panel("fit_sensitivity", "slope sensitivity per knob", draw_b,
              bumped(4.3, 3.2)),
    ]
    render("numerical_stability_audit", panels, _grid(1, 2), (8.4, 3.2),
           out_dir, png, tight_kw={})


def _corpus_result_draw(art: dict, corpora, main_text: bool = False,
                        show_legend: bool = True,
                        show_regions: bool = True):
    r"""Corpus result panel shared by the summary and diagnostic figures.

    The shaded band per corpus/arm runs from ``mmse_hat`` up to the Brier score.
    Brier upper-bounds the true MMSE for *any* predictor and equals ``mmse_hat``
    exactly when the predictor is calibrated -- so a thin band means the
    measurement is trustworthy and the truth is pinned between the two, and a
    fat band means the estimator has stopped measuring the data law.  That puts
    the reason a cell is censored right next to the censored cell.

    ``main_text`` drops the fitted constants (they belong in the caption) and
    adds a slope guide showing the decay law that was *measured* rather than the
    exponential Asm. 4.1 assumes.  One code path, so the appendix and main-text
    versions cannot drift apart.
    """
    rd = art["corpus_uncertainty"]
    rdf = art["corpus_uncertainty_fits"]

    def draw(ax: Axes) -> None:
        # every calibrated estimator is an upper bound, so the *lowest* curve is
        # the best available one: use the strictly richer feature set
        best = rd[(rd.arm == "small") | (rd.features == BEST_FEATURES)]
        xmin, xmax = 0.62, float(best.V.max()) * 2.2
        small_hi = float(best[best.arm == "small"].V.max())
        large_lo = float(best[best.arm == "large"].V.min())
        fs = 6.4 if main_text else 7.5

        # Which estimator is valid where; the strip between the arms is the
        # empirical analogue of [s_pin, s_ov].  Only the strip and the
        # classifier side are tinted: the surface is white by default, so
        # shading is spent on the distinction that matters rather than on
        # colouring in the whole panel.  Marker shape already separates the
        # arms, so the tint only has to say "the bound changes direction here".
        ax.axvspan(small_hi * 1.06, large_lo * 0.94, color=CAT[3], alpha=0.18,
                   lw=0, zorder=0)
        ax.axvspan(large_lo * 0.94, xmax, color=MUTED, alpha=0.10, lw=0, zorder=0)
        # At main-text width these three region labels have no room; the
        # caption names the arms instead.
        if show_regions and not (main_text and M_COMPACT):
            # Left-aligned and set in from the spine: centring it in a band
            # this narrow puts the first letter on the axis, and starting at
            # the very left runs it into the curves, which are still near the
            # top of the panel at their first few visible sizes.
            ax.annotate("posterior", xy=(xmin * 2.8, 0.985),
                        xycoords=("data", "axes fraction"), ha="left", va="top",
                        fontsize=fs - 0.4, color=SECONDARY, linespacing=0.95)
            ax.annotate("classifier", xy=(math.sqrt(large_lo * xmax), 0.985),
                        xycoords=("data", "axes fraction"), ha="center", va="top",
                        fontsize=fs - 0.4, color=SECONDARY)
            # the window strip is the one *structural* boundary in the panel --
            # where the estimator, and with it the bound direction, changes --
            # so it is named in the palette red rather than in the muted ink
            ax.annotate("window", xy=(math.sqrt(small_hi * large_lo), 0.72),
                        xycoords=("data", "axes fraction"), ha="center", va="center",
                        fontsize=fs - 0.4, color=CAT[1], rotation=90)

        for corpus in corpora:
            c = CORPUS_COLORS.get(corpus, CAT[0])
            for arm in ("small", "large"):
                g = best[(best.corpus == corpus) & (best.arm == arm)].sort_values("V")
                if g.empty:
                    continue
                ax.fill_between(g.V, g.mmse_hat, g.brier, color=c, alpha=0.16,
                                lw=0, zorder=1)
                ax.plot(g.V, g.brier, color=c, lw=0.8, ls=(0, (2, 1.6)),
                        alpha=0.7, zorder=2)
                free, cens = g[~g.floored], g[g.floored]
                ax.errorbar(free.V, free.mmse_hat,
                            yerr=[free.mmse_hat - free.ci_lo,
                                  free.ci_hi - free.mmse_hat],
                            color=c, marker=CORPUS_MARKERS[arm], ms=3.6, lw=1.2,
                            capsize=1.5, elinewidth=0.7, zorder=3)
                if len(cens):
                    ax.plot(cens.V, cens.mmse_hat, color=c, marker=CORPUS_MARKERS[arm],
                            ms=4.2, lw=0, mfc="none", mew=1.0, zorder=3)

        live = best[(best.arm == "large") & (~best.floored)]
        if main_text:
            ref = live[live.corpus == corpora[-1]].sort_values("V")
            if len(ref):
                V0, y0 = float(ref.V.iloc[0]), float(ref.mmse_hat.iloc[0])
                vv = np.array([V0, float(ref.V.iloc[-1]) * 1.5])
                yy = 1.9 * y0 * (vv / V0) ** -1.0
                ax.plot(vv, yy, color=INK, lw=1.0, ls=(0, (4, 2)), alpha=0.7,
                        zorder=4)
                # Below the near end: above it collides with the "classifier
                # (upper bd.)" region label, and the far end runs into the data.
                if not M_COMPACT:
                    # further right and lower: the guide descends to the right,
                    # so a label just below its start still crosses it
                    ax.annotate(r"$\propto |V|^{-1}$", xy=(vv[0], yy[0]),
                                xytext=(28, -8), textcoords="offset points",
                                ha="left", va="top", fontsize=fs, color=INK)
            drops = []
            for corpus in corpora:
                a = best[(best.corpus == corpus) & (best.V == best.V.min())]
                b = live[live.corpus == corpus].sort_values("V")
                if len(a) and len(b):
                    drops.append(float(a.mmse_hat.iloc[0]) / float(b.mmse_hat.iloc[-1]))
        elif _has(rdf):
            prim = rdf[(rdf.corpus == "code_prose") & (rdf.arm == "small")]
            if len(prim):
                ax.annotate(rf"$\hat u_0={prim.u0_hat.iloc[0]:.3f}$, "
                            rf"$\hat L={prim.L_hat.iloc[0]:.3f}$",
                            xy=(0.03, 0.30), xycoords="axes fraction",
                            fontsize=7.5, color=CORPUS_COLORS["code_prose"])
            sec = rdf[(rdf.corpus == "bilingual") & (rdf.arm == "large")]
            if len(sec):
                ax.annotate(rf"$\hat\kappa={sec.kappa_hat.iloc[0]:.4f}$",
                            xy=(0.98, 0.78), xycoords="axes fraction", ha="right",
                            fontsize=7.5, color=CORPUS_COLORS["bilingual"])

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel(r"visible tokens $|V|$")
        ax.set_ylabel(r"$\widehat{\mathrm{mmse}}(Z\mid X_V)$")

        h = [Line2D([], [], color=CORPUS_COLORS.get(c, CAT[0]), lw=2.0) for c in corpora]
        lab = [CORPUS_LABELS.get(c, c) for c in corpora]
        if not (main_text and M_COMPACT):   # the caption explains the band
            h.append(Patch(facecolor=MUTED, alpha=0.3, lw=0))
            lab.append(r"band: $\widehat{\mathrm{mmse}}$ to Brier")
        # only the classifier arm can be censored, so the proxy carries that
        # arm's marker; an "o" proxy would name a symbol that appears nowhere
        if bool(rd.floored.any()):
            h.append(Line2D([], [], color=MUTED, marker=CORPUS_MARKERS["large"],
                            lw=0, mfc="none", mew=1.0))
            lab.append("censored")
        if show_legend:
            ax.legend(h, lab, fontsize=fs - 0.4, loc="lower left",
                      labelspacing=0.3, handlelength=1.6, frameon=True,
                      framealpha=0.85, edgecolor="none", facecolor=SURFACE)

    return draw


def _write_corpus_mode_uncertainty_panels(
    art: dict, out_dir: Path, png: bool = False
) -> None:
    """Write untitled per-corpus result panels beside the diagnostics.

    The paper identifies each corpus in its subcaption, so the PDFs contain no
    redundant ``code vs prose`` or ``German vs English`` title.  Both panels
    use common axis limits and dimensions so they can be placed side by side
    without further geometric correction.
    """
    rd = art["corpus_uncertainty"]
    if not _has(rd):
        return
    corpora = list(dict.fromkeys(rd.corpus))

    best = rd[(rd.arm == "small") | (rd.features == BEST_FEATURES)]
    lower = best[["mmse_hat", "ci_lo"]].to_numpy(dtype=float).ravel()
    upper = best[["brier", "ci_hi"]].to_numpy(dtype=float).ravel()
    lower = lower[np.isfinite(lower) & (lower > 0.0)]
    upper = upper[np.isfinite(upper) & (upper > 0.0)]
    y_limits = None
    if lower.size and upper.size:
        lo, hi = float(np.log(lower.min())), float(np.log(upper.max()))
        pad = 0.05 * max(hi - lo, 1.0)
        y_limits = (math.exp(lo - pad), math.exp(hi + pad))

    panel_size = bumped(4.8, 2.85)
    panel_dir = out_dir / "figure_corpus_mode_uncertainty_diagnostics"
    file_keys = {
        "code_prose": "code_prose",
        "bilingual": "german_english",
    }
    for corpus in corpora:
        fig, ax = plt.subplots(figsize=panel_size)
        _corpus_result_draw(
            art, [corpus], main_text=True, show_legend=False,
            show_regions=False,
        )(ax)
        if y_limits is not None:
            ax.set_ylim(*y_limits)
        fig.tight_layout()
        key = file_keys.get(str(corpus), str(corpus).replace(" ", "_"))
        save(fig, panel_dir / f"panel_{key}", png=png, tight=False)
    print(
        "  figure_corpus_mode_uncertainty_diagnostics: "
        f"{len(corpora)} untitled corpus result panels"
    )


def figure_corpus_mode_uncertainty_diagnostics(
    art: dict, out_dir: Path, png: bool = False
) -> None:
    """Reliability, Brier check, backoff order, and corpus metadata."""
    rd = art["corpus_uncertainty"]
    calib = art["corpus_uncertainty_calibration"]
    stats = art["corpus_uncertainty_stats"]
    if not _has(rd):
        return
    corpora = list(dict.fromkeys(rd.corpus))

    def draw_a(ax: Axes) -> None:
        r"""How tight is the upper bound, and what makes it loose?

        A calibrated predictor satisfies ``Brier == mmse`` exactly, so
        ``Brier / mmse-hat`` *is* the slack in the bound (a) reports, and is
        the quantity the censoring rule thresholds at 2.  The dashed series
        isolates the part of that slack attributable to calibration -- the
        Murphy reliability term ``sum_i n_i (emp_i - pred_i)^2 / sum_i n_i``,
        aggregated from the stored per-bin curve.  The gap between the two is
        everything else.

        This replaces a reliability diagram, which cannot show any of it.  At
        the cells that actually get censored the isotonic map collapses to two
        bins at the extremes, and there a predicted ``1.7e-5`` against an
        empirical ``1.7e-3`` -- a factor of 98, and the whole reason the cell
        is censored -- plots as two dots sitting on the corners of the
        diagonal.  Probability ratios near 0 and 1 need a log axis; a
        reliability diagram is linear in probability by construction, so it
        showed 18 curves agreeing on the one thing that was never in doubt.
        """
        large = (rd[(rd.arm == "large") & (rd.features == BEST_FEATURES)]
                 if _has(rd) else None)
        if large is None or large.empty or not _has(calib):
            ax.set_axis_off()
            return
        cal = calib[calib.features == BEST_FEATURES]
        if cal.empty:                       # older artifact, single feature set
            cal = calib
        thresh = float(stats.brier_ratio_max.iloc[0]) if (
            _has(stats) and "brier_ratio_max" in stats.columns) else 2.0

        top = float(large.brier_ratio.max()) * 2.4
        ax.set_ylim(0.86, top)
        ax.axhline(1.0, color=INK, lw=0.9, ls=(0, (3, 2)), alpha=0.6, zorder=1)
        ax.axhspan(thresh, top, color=MUTED, alpha=0.13, lw=0, zorder=0)

        for corpus in corpora:
            col = CORPUS_COLORS.get(corpus, CAT[0])
            g = large[large.corpus == corpus].sort_values("V")
            if g.empty:
                continue
            # markers drawn per censoring state rather than overlaid: an open
            # marker painted on top of a filled one still reads as filled, and
            # then the legend promises a mark that never appears
            ax.plot(g.V, g.brier_ratio, color=col, lw=1.4, zorder=3)
            for sub, face in ((g[~g.floored], col), (g[g.floored], SURFACE)):
                if not sub.empty:
                    ax.plot(sub.V, sub.brier_ratio, color=col, lw=0, marker="o",
                            ms=4.2, mfc=face, mew=1.1, zorder=4)
            pts = []
            for V, h in cal[cal.corpus == corpus].groupby("V"):
                row = g[g.V == V]
                if row.empty or float(row.mmse_hat.iloc[0]) <= 0:
                    continue
                rel = np.average((h.emp_freq - h.pred_mean) ** 2, weights=h.n)
                pts.append((float(V), 1.0 + rel / float(row.mmse_hat.iloc[0])))
            if pts:
                pts.sort()
                ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col,
                        lw=1.0, ls=(0, (4, 1.6)), marker="s", ms=3.0,
                        mfc="none", alpha=0.9, zorder=2)
            if corpus == "bilingual":
                endpoint = g.iloc[-2] if len(g) > 1 else g.iloc[-1]
                label_offset = (-5, 0)
                label_ha, label_va = "right", "center"
            elif corpus == "code_prose":
                endpoint = g.iloc[min(5, len(g) - 1)]
                label_offset = (0, 5)
                label_ha, label_va = "center", "bottom"
            else:
                endpoint = g.iloc[-2] if len(g) > 1 else g.iloc[-1]
                label_offset = (4, 0)
                label_ha, label_va = "left", "center"
            ax.annotate(CORPUS_LABELS.get(corpus, corpus),
                        xy=(float(endpoint.V), float(endpoint.brier_ratio)),
                        xytext=label_offset, textcoords="offset points",
                        ha=label_ha, va=label_va, fontsize=6.1, color=col,
                        clip_on=False)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"visible tokens $|V|$")
        ax.set_ylabel(r"Brier $/\ \widehat{\mathrm{mmse}}$")
        corner_note(ax, rf"shaded: censored above ${thresh:g}\times$",
                    xy=(0.97, 0.97), ha="right", va="top")
        h = [Line2D([], [], color=MUTED, lw=1.4, marker="o", ms=4.0),
             Line2D([], [], color=MUTED, lw=1.0, ls=(0, (4, 1.6)), marker="s",
                    ms=3.0, mfc="none"),
             Line2D([], [], color=MUTED, lw=0, marker="o", ms=4.0, mfc="none")]
        ax.legend(h, ["total slack", "calibration only", "censored"],
                  fontsize=6.2, loc="upper left", labelspacing=0.3)
    def draw_c(ax: Axes) -> None:
        """Realised backoff order as a compact corpus-by-context heatmap."""
        g_all = rd[rd.arm == "small"]
        table = (g_all.pivot_table(index="corpus", columns="V", values="eff_order",
                                   aggfunc="first").reindex(corpora))
        values = table.to_numpy(dtype=float)
        finite_orders = sorted({int(value) for value in values[np.isfinite(values)]})
        order_cmap = ListedColormap(CAT[:len(finite_orders)], name="mpi_order")
        order_norm = BoundaryNorm(
            np.arange(min(finite_orders) - 0.5, max(finite_orders) + 1.5),
            order_cmap.N,
        )
        image = ax.imshow(values, aspect="auto", cmap=order_cmap, norm=order_norm)
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                if np.isfinite(values[row, col]):
                    ax.text(col, row, f"{values[row, col]:.0f}", ha="center",
                            va="center", fontsize=7.0, color=CAT[1],
                            path_effects=[
                                patheffects.withStroke(
                                    linewidth=1.2, foreground=SURFACE,
                                )
                            ])
        ax.set_xticks(np.arange(len(table.columns)))
        ax.set_xticklabels([str(int(v)) for v in table.columns])
        ax.set_yticks(np.arange(len(table.index)))
        ax.set_yticklabels([CORPUS_LABELS.get(c, c) for c in table.index])
        ax.set_xlabel(r"$|V|$")
        ax.set_ylabel("")
        ax.grid(False)
        ax.set_xticks(np.arange(-0.5, values.shape[1], 1.0), minor=True)
        ax.set_yticks(np.arange(-0.5, values.shape[0], 1.0), minor=True)
        ax.grid(which="minor", color=NULL_GREY, linestyle="-", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)
        colorbar = ax.figure.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
        colorbar.set_label("order", fontsize=6.5)
        colorbar.set_ticks([1, 2])
        colorbar.ax.tick_params(labelsize=6.0)

    def draw_e(ax: Axes) -> None:
        """Model-class test: does a strictly richer classifier change the law?

        Bag-of-tokens is order-blind, so a flattening curve could equally be
        the corpus decaying slowly or the model class hitting its ceiling.
        Adding a hashed bigram block gives a strictly richer class -- it can
        reproduce the unigram solution exactly -- on identical windows, splits
        and calibration.  If the curve steepens, the earlier one was
        model-limited; if only its level moves, the law belongs to the corpus.
        """
        large = rd[rd.arm == "large"]
        styles = {"unigram": (0, (4, 1.6)), BEST_FEATURES: (0, ())}
        for corpus in corpora:
            c = CORPUS_COLORS.get(corpus, CAT[0])
            for fs, ls in styles.items():
                g = large[(large.corpus == corpus) & (large.features == fs)
                          & (~large.floored)].sort_values("V")
                if g.empty:
                    continue
                ax.plot(g.V, g.mmse_hat, color=c, ls=ls, lw=1.3,
                        marker="o" if fs == "unigram" else "s", ms=3.2,
                        mfc="none" if fs == "unigram" else c)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"visible tokens $|V|$")
        ax.set_ylabel(r"$\widehat{\mathrm{mmse}}(Z\mid X_V)$")
        h = [Line2D([], [], color=MUTED, ls=styles["unigram"], marker="o",
                    ms=3.2, mfc="none"),
             Line2D([], [], color=MUTED, ls=styles[BEST_FEATURES], marker="s",
                    ms=3.2)]
        # Short feature labels, matching tab:exp-corpora-form's "uni" / "uni+bi":
        # the corpus names are already long, and the legend has to fit into the
        # corner the data vacates -- past the last classifier point at |V|=256
        # only the lower curve remains, so the free region is about 30% of the
        # axis width.  Narrowing the labels is what buys the fit; the frame is
        # there so the descending curve cannot cut through a word if a longer
        # corpus name is ever added.
        lab = ["unigram", "+ bigrams"]
        h += [Line2D([], [], color=CORPUS_COLORS.get(c, CAT[0]), lw=2.0) for c in corpora]
        lab += [CORPUS_LABELS.get(c, c) for c in corpora]
        ax.legend(h, lab, fontsize=6.2, loc="upper right", ncol=1,
                  labelspacing=0.24, handlelength=1.2, handletextpad=0.5,
                  borderpad=0.3, frameon=True, framealpha=0.9,
                  edgecolor="none", facecolor=SURFACE)

    def draw_f(ax: Axes) -> None:
        """Which functional form fits: the exponential Asm. 4.1 assumes, or a
        power law?  Fitted on the cells uncensored for *every* feature set, so
        the model classes are compared like with like."""
        form = art["corpus_uncertainty_form"]
        if not _has(form) or "fit_range" not in form.columns:
            ax.text(0.5, 0.5, "re-run the corpus study for this comparison", ha="center",
                    va="center", color=MUTED, transform=ax.transAxes)
            ax.set_axis_off()
            return
        f = form[(form.fit_range == "common") & (form.arm == "large")]
        rows = list(f.groupby(["corpus", "features"], sort=False))
        y = np.arange(len(rows))[::-1]
        for yi, ((corpus, features), group) in zip(y, rows):
            exp_r2 = float(group[group.model == "exponential"].R2.iloc[0])
            power_row = group[group.model == "power_law"].iloc[0]
            power_r2 = float(power_row.R2)
            ax.plot([exp_r2, power_r2], [yi, yi], color=GRID, lw=2.0,
                    solid_capstyle="round", zorder=1)
            ax.plot(exp_r2, yi, marker="s", color=CAT[1], ms=4.2, zorder=3)
            ax.plot(power_r2, yi, marker="o", color=CAT[0], ms=4.2, zorder=3)
            ax.annotate(rf"$V^{{{float(power_row.rate_or_exponent):.2f}}}$",
                        xy=(power_r2, yi), xytext=(4, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=6.2, color=CAT[0])
        ax.set_yticks(y)
        ax.set_yticklabels(
            [("Ge. vs Eng." if c == "bilingual" else CORPUS_LABELS.get(c, c))
             + "\n" + fs.replace("_", "+")
             for (c, fs), _ in rows], fontsize=6.7)
        ax.set_xlim(0.72, 1.03)
        ax.set_xlabel(r"fit $R^2$")
        ax.grid(axis="y", visible=False)
        ax.legend(
            handles=[
                Line2D([], [], color=CAT[1], marker="s", lw=0,
                       label="exponential"),
                Line2D([], [], color=CAT[0], marker="o", lw=0,
                       label="power law"),
            ], loc="lower left", fontsize=6.7,
        )

    panels = [
        # Ordered result -> the two analyses that qualify it -> the two
        # diagnostics that license it (one per estimator arm).  All panels are
        # emitted at one size so they can be re-laid-out freely as subfigures;
        # the composed 2x3 is a contact sheet, not a layout prescription.
        # Corpora / licences / protocol are not a figure -- they live in the
        # README and the paper's appendix.
        Panel("decay_function", "decay-model comparison", draw_f,
              A8_PANEL, M_TITLE_SIZE),
        Panel("classifier_richness", "classifier richness", draw_e,
              A8_PANEL, M_TITLE_SIZE),
        Panel("bound_slack", "upper-bound slack", draw_a,
              A8_PANEL, M_TITLE_SIZE),
        Panel("posterior_backoff_order", "posterior backoff order",
              draw_c, A8_PANEL, M_TITLE_SIZE),
    ]
    render("corpus_mode_uncertainty_diagnostics", panels, _grid(2, 2),
           (2 * A8_PANEL[0], 2 * A8_PANEL[1]),
           out_dir, png, tight_kw={})
    _write_corpus_mode_uncertainty_panels(art, out_dir, png)


# ==========================================================================
# Sampling cost of low-visibility mass
# ==========================================================================


def figure_low_visibility_sampling_cost(
    art: dict, out_dir: Path, png: bool = False, part: str = "detection"
) -> None:
    """Render one focused view of the low-visibility sampling-cost study.

    ``part='detection'`` covers sample complexity; ``part='loss'`` covers the
    per-example loss and its variance decomposition.
    """
    smp = art["sampling"]
    if not _has(smp):
        return
    law, param, w = MAIN_LAW[0], MAIN_LAW[1], MAIN_W
    N = int(max(smp.N.unique()))
    cell = _sel(smp, law=law, param=param, w=w, N=N)
    m_base = int(sorted(cell.m_base.unique())[-1])
    g_all = _sel(cell, m_base=m_base)
    end = g_all[g_all.lambda_frac == 1.0]          # the interval endpoint of I
    lam_end = float(end["lambda"].iloc[0])
    s_vals = [int(s) for s in sorted(g_all.s.unique())]
    note = rf"$N={N}$,  $m_{{\rm base}}={m_base}$"

    def _guide(ax, x, y0, x0, slope, label, offset=0.30, gap=0.55,
               va="top") -> None:
        """A reference line of the predicted slope, drawn *beside* the data.

        Anchoring it on the data would hide it underneath the very curve it is
        meant to characterise -- these exponents are exact, so the guide and
        the series coincide to the pixel.  ``offset`` displaces it clear of
        them and the label rides at its midpoint.
        """
        x = np.sort(np.asarray(x, dtype=float))
        y = y0 * offset * (x / x0) ** slope
        ax.plot(x, y, color=INK, lw=0.9, ls=(0, (4, 2)), alpha=0.6, zorder=1)
        mid = len(x) // 2
        ax.annotate(label, xy=(x[mid], y[mid] * gap), ha="center", va=va,
                    fontsize=8.5, color=INK)

    def draw_a(ax: Axes) -> None:
        """Examples needed to detect the largest error I asks about."""
        for j, s in enumerate(s_vals):
            g = end[(end.s == s) & (end.pi > 0)].sort_values("pi")
            ax.plot(g.pi, g.n_star, color=CAT[j % len(CAT)], lw=1.3,
                    marker=MARKERS[j], ms=3.4)
            ax.annotate(rf"$s={s}$", xy=(float(g.pi.iloc[-1]),
                        float(g.n_star.iloc[-1])), xytext=(4, 0),
                        textcoords="offset points", fontsize=6.5,
                        color=CAT[j % len(CAT)], ha="left", va="center",
                        clip_on=False)
        ref = end[(end.s == 0) & (end.pi > 0)].sort_values("pi")
        pp = ref.pi.to_numpy()
        _guide(ax, pp, float(ref.n_star.iloc[0]), float(pp[0]), -1.0,
               "slope $-1$")
        # An untruncated diffusion schedule puts only Theta(1/N) on the full
        # mask; that is the operating point the paper's Sec. 5.2 is about.
        reference_color = CAT[len(s_vals) % len(CAT)]
        ax.axvline(1.0 / N, color=reference_color, lw=1.0,
                   ls=(0, (1, 1.6)), zorder=1)
        ax.annotate(r"$\pi_0 = 1/N$", xy=(1.0 / N, 0.97),
                    xycoords=ax.get_xaxis_transform(), rotation=90,
                    ha="right", va="top", fontsize=7.5, color=reference_color)
        blind = float(end[(end.s == 0) & (end.pi == 0)].log10_n_star.iloc[0])
        corner_note(ax, "mode-blind ($\\pi_0=0$):\n"
                        rf"$n^\star \approx 10^{{{blind:.0f}}}$",
                    xy=(0.97, 0.97), ha="right", va="top", fontsize=7.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"low-visibility schedule mass $\pi_s(\mu)$")
        ax.set_ylabel(r"$n^\star$ (masked examples)")
        ax.margins(x=0.12)

    def draw_b(ax: Axes) -> None:
        """Why (a) has slope -1: the signal is linear in pi, the noise is not."""
        g = end[(end.s == 0) & (end.pi > 0)].sort_values("pi")
        pp = g.pi.to_numpy()
        sig, noi = np.exp(g.logD.to_numpy()), g.sd.to_numpy()
        ax.plot(pp, sig, color=CAT[0], lw=1.4, marker="o", ms=3.4,
                label=r"signal  $\mathbb{E}[S]=\mathfrak{D}_\mu$")
        ax.plot(pp, noi, color=CAT[1], lw=1.4, marker="s", ms=3.4,
                label=r"noise  $\mathrm{sd}[S]$")
        _guide(ax, pp, sig[0], pp[0], 1.0, r"$\propto \pi_0$")
        _guide(ax, pp, noi[0], pp[0], 0.5, r"$\propto \sqrt{\pi_0}$",
               offset=2.5, gap=1.5, va="bottom")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"full-mask schedule mass $\pi_0$")
        ax.set_ylabel("nats per example")
        ax.legend(loc="upper left", fontsize=8.0)
        corner_note(ax, note + rf",  $s=0$,  $\lambda={lam_end:g}$",
                    xy=(0.97, 0.03), ha="right", va="bottom", fontsize=7.5)

    def draw_c(ax: Axes) -> None:
        """The same cost as a function of how small an error must be resolved."""
        pi_c = 1e-3
        sub = g_all[np.isclose(g_all.pi, pi_c)]
        for j, s in enumerate(s_vals):
            g = sub[sub.s == s].sort_values("lambda_err")
            ax.plot(g.lambda_err, g.n_star, color=CAT[j % len(CAT)], lw=1.3,
                    marker=MARKERS[j], ms=3.4)
            ax.annotate(rf"$s={s}$", xy=(float(g.lambda_err.iloc[-1]),
                        float(g.n_star.iloc[-1])), xytext=(4, 0),
                        textcoords="offset points", fontsize=6.5,
                        color=CAT[j % len(CAT)], ha="left", va="center",
                        clip_on=False)
        ref = sub[sub.s == 0].sort_values("lambda_err")
        xx = ref.lambda_err.to_numpy()
        _guide(ax, xx, float(ref.n_star.iloc[0]), float(xx[0]), -2.0,
               "slope $-2$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"mode-weight error $|\lambda - w|$")
        ax.set_ylabel(r"$n^\star$ (masked examples)")
        ax.margins(x=0.12)
        corner_note(ax, note + rf",  $\pi = 10^{{{math.log10(pi_c):.0f}}}$",
                    xy=(0.97, 0.97), ha="right", va="top", fontsize=7.5)

    def draw_d(ax: Axes) -> None:
        """The raw loss a single example costs, by mask size.

        The full mask sits at the joint entropy by construction; the point of
        the panel is that ``s = 1, 2, 4`` sit there too, so this axis cannot
        tell a near-full mask apart from the boundary itself.
        """
        rows = [(rf"$s={s}$", float(end[end.s == s].H_boost.iloc[0]),
                 CAT[j % len(CAT)]) for j, s in enumerate(s_vals)]
        rows += [(rf"$m={int(mb)}$",
                  float(cell[cell.m_base == mb].H_base.iloc[0]), MUTED)
                 for mb in sorted(cell.m_base.unique())]
        y = np.arange(len(rows))[::-1]
        for yi, (_, h, col) in zip(y, rows):
            ax.plot([0.0, h], [yi, yi], color=col, lw=2.0, alpha=0.45,
                    solid_capstyle="butt", zorder=2)
            ax.plot([h], [yi], color=col, marker="o", ms=5.0, zorder=3)
        h_joint = float(end.H_joint.iloc[0])
        ax.axvline(h_joint, color=INK, lw=0.9, ls=(0, (4, 2)), alpha=0.6, zorder=1)
        ax.annotate(rf"$H(X) = {h_joint:.0f}$ nats",
                    xy=(h_joint * 0.98, len(rows) - 0.45), ha="right",
                    va="bottom", fontsize=6.6, color=INK)
        ax.set_yticks(y)
        ax.set_yticklabels([r[0] for r in rows])
        ax.set_ylim(-0.7, len(rows) + 0.25)
        ax.set_xlim(0.0, h_joint * 1.22)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(r"raw loss per example  $H(X_K \mid X_V)$  (nats)")
        worst = min(r[1] for r in rows[:len(s_vals)])
        corner_note(ax, rf"$N={N}$" "\n"
                        rf"$s=4$: ${100*(h_joint-worst)/h_joint:.2f}\%$ below $H(X)$",
                    xy=(0.97, 0.05), ha="right", va="bottom")

    def draw_e(ax: Axes) -> None:
        """Decompose raw-loss variance into within- and between-mask terms."""
        g = end[(end.s == 0) & (end.pi > 0)].sort_values("pi")
        within = g.raw_var_within.to_numpy(dtype=float)
        between = g.raw_var_between.to_numpy(dtype=float)
        total = within + between
        ax.fill_between(g.pi, 0.0, within, color=MUTED, alpha=0.24,
                        linewidth=0, label="within-mask")
        ax.fill_between(g.pi, within, total, color=CAT[1], alpha=0.28,
                        linewidth=0, label="schedule mixture")
        ax.plot(g.pi, total, color=CAT[0], lw=1.4, marker="o", ms=3.4,
                label="total")
        cross = g[g.raw_var_schedule_frac >= 0.5]
        if not cross.empty:
            x_cross = float(cross.pi.iloc[0])
            ax.axvline(x_cross, color=INK, lw=0.8, ls=(0, (4, 2)), alpha=0.7)
            ax.text(0.97, 0.72, "schedule term dominates",
                    transform=ax.transAxes, fontsize=6.3,
                    color=SECONDARY, ha="right", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE,
                              ec="none", alpha=0.85))
        ax.set_xscale("log")
        ax.set_xlabel(r"full-mask schedule mass $\pi_0$")
        ax.set_ylabel(r"variance of raw loss $L$ (nats$^2$)")
        ax.set_ylim(bottom=0.0)
        ax.legend(loc="upper left", fontsize=6.8, ncol=1)
        corner_note(ax, note + ",  $s=0$", xy=(0.97, 0.03), ha="right", va="bottom")
    if part == "detection":
        panels = [
            Panel("detection_cost", "discrepancy detection cost", draw_a,
                  A8_PANEL, M_TITLE_SIZE),
            Panel("schedule_signal_to_noise", "signal and noise scaling",
                  draw_b, A8_PANEL, M_TITLE_SIZE),
            Panel("mode_weight_resolution", "mode-weight resolution cost",
                  draw_c, A8_PANEL, M_TITLE_SIZE),
        ]
        render("low_visibility_detection_cost", panels, _grid(1, 3),
               (3 * A8_PANEL[0], A8_PANEL[1]),
               out_dir, png, tight_kw={})
    elif part == "loss":
        panels = [
            Panel("raw_loss_by_mask_size", "raw loss by mask size", draw_d,
                  A8_PANEL, M_TITLE_SIZE),
            Panel("raw_loss_variance", "raw-loss variance components", draw_e,
                  A8_PANEL, M_TITLE_SIZE),
        ]
        render("low_visibility_loss_cost", panels, _grid(1, 2),
               (2 * A8_PANEL[0], A8_PANEL[1]),
               out_dir, png, tight_kw={})
    else:
        raise ValueError(f"unknown sampling-cost figure part: {part!r}")


# ==========================================================================
# Direct optimization inside q_lambda
# ==========================================================================


def _mode_weight_selected_N(schedules: pd.DataFrame, runs: pd.DataFrame) -> int:
    """Recover the calibration-selected dimension without redoing selection."""
    for frame in (schedules, runs):
        if "selected_N" in frame and len(frame):
            selected = frame[frame["selected_N"].astype(bool)]
            if len(selected):
                return int(selected["N"].iloc[0])
    if len(runs):
        # The runner trains exactly one selected dimension.  This fallback is
        # for early development artifacts that predate the explicit flag.
        return int(runs["N"].mode().iloc[0])
    return int(schedules["N"].mode().iloc[0])


def _mode_weight_plot_schedule_keys(
    frame: pd.DataFrame,
) -> list[tuple[str, int, float]]:
    """Blind plus at most two representative non-full-mask interventions."""
    keys: list[tuple[str, int, float]] = []
    blind = frame[frame["pi"] == 0.0]
    if len(blind):
        row = blind.iloc[0]
        keys.append((str(row.schedule), int(row.s), float(row.pi)))

    boosted = frame[(frame["pi"] > 0.0) & (frame["s"] == 1)].copy()
    if "plot" in boosted:
        flagged = boosted[boosted["plot"].astype(bool)]
        if len(flagged):
            boosted = flagged
    elif "role" in boosted:
        flagged = boosted[
            boosted["role"].astype(str).str.contains("representative|primary", case=False)
        ]
        if len(flagged):
            boosted = flagged
    boosted = boosted.sort_values("pi").drop_duplicates(["schedule", "s", "pi"])
    if len(boosted) > 2:
        # Keep the weak schedule used in the trajectory demonstration, plus
        # the strongest retained intervention for contrast.
        target = 3e-3
        weak_index = int(np.argmin(np.abs(np.log(boosted.pi.to_numpy() / target))))
        boosted = boosted.iloc[sorted({weak_index, len(boosted) - 1})]
    for row in boosted.itertuples():
        key = (str(row.schedule), int(row.s), float(row.pi))
        if key not in keys:
            keys.append(key)
    return keys


def _mode_weight_key_mask(
    frame: pd.DataFrame, keys: list[tuple[str, int, float]]
) -> np.ndarray:
    mask = np.zeros(len(frame), dtype=bool)
    for schedule, s, pi in keys:
        mask |= (
            (frame["schedule"].astype(str) == schedule)
            & (frame["s"].astype(int) == s)
            & np.isclose(frame["pi"].astype(float), pi)
        ).to_numpy()
    return mask


def _mode_weight_label(schedule: str, s: int, pi: float) -> str:
    return "blind" if pi == 0.0 else rf"$s={s},\ \pi={pi:g}$"


def figure_mode_weight_optimization(
    art: dict, out_dir: Path, png: bool = False
) -> None:
    """Direct-q_lambda curvature, trajectories, and convergence time.

    Visibility curves use distinct sparse marks, trajectories are faceted by
    schedule, and symmetric initializations are summarized before the
    half-time comparison so agreement is visible rather than overprinted.
    """
    calibration = art["mode_weight_visibility_calibration"]
    schedules = art["mode_weight_schedule_calibration"]
    trajectories = art["mode_weight_trajectories"]
    runs = art["mode_weight_run_summaries"]
    if not all(_has(frame) for frame in (calibration, schedules, trajectories, runs)):
        raise ValueError(
            "mode-weight artifacts are incomplete; run the "
            "mode-weight-optimization study"
        )

    assert calibration is not None and schedules is not None
    assert trajectories is not None and runs is not None
    selected_N = _mode_weight_selected_N(schedules, runs)
    selected_runs = runs[runs["N"] == selected_N].copy()
    selected_traj = trajectories[trajectories["N"] == selected_N].copy()
    keys = _mode_weight_plot_schedule_keys(selected_runs)
    if keys:
        selected_traj = selected_traj[_mode_weight_key_mask(selected_traj, keys)]

    schedule_colors = {key: CAT[i % len(CAT)] for i, key in enumerate(keys)}

    # ------------------------------------------------------------------ (a)
    def draw_a(ax: Axes) -> None:
        all_N = sorted(int(value) for value in calibration["N"].unique())
        if not all_N:
            raise ValueError("no visibility-curvature cells are available")

        fractions = (0.25, 0.50, 0.75)
        x = np.arange(len(fractions), dtype=float)
        width = 0.76 / len(all_N)
        maximum = 0.0
        for index, N in enumerate(all_N):
            group = calibration[calibration["N"] == N].sort_values("m")
            group = group[np.isfinite(group["U"]) & (group["U"] > 0.0)]
            if not len(group):
                continue
            rows = []
            for fraction in fractions:
                target = int(math.floor(fraction * N))
                rows.append(group.iloc[int(np.argmin(np.abs(group.m.to_numpy() - target)))])
            heights = -np.log10(np.asarray([row.U for row in rows], dtype=float))
            positions = x + (index - 0.5 * (len(all_N) - 1)) * width
            color = n_color(int(N), all_N)
            ax.bar(
                positions, heights, width=width * 0.88, color=color,
                alpha=1.0 if int(N) == selected_N else 0.68,
                edgecolor=INK if int(N) == selected_N else SURFACE,
                linewidth=0.7 if int(N) == selected_N else 0.35,
                label=(rf"$N={int(N)}$ selected" if int(N) == selected_N
                       else rf"$N={int(N)}$"), zorder=2,
            )

            # Fit log H against visibility using only this optimization-
            # calibration artifact.  The middle-half window matches the three
            # reported fractions and avoids letting either boundary dominate.
            fit_lo = int(math.floor(0.25 * N))
            fit_hi = int(math.floor(0.75 * N))
            fit_group = group[(group["m"] >= fit_lo) & (group["m"] <= fit_hi)]
            slope, intercept = np.polyfit(
                fit_group["m"].to_numpy(dtype=float),
                np.log(fit_group["U"].to_numpy(dtype=float)),
                deg=1,
            )
            fit_heights = -np.asarray(
                [intercept + slope * float(row.m) for row in rows], dtype=float,
            ) / math.log(10.0)
            tick_half_width = width * 0.30
            for position, fit_height in zip(positions, fit_heights):
                ax.plot(
                    [position - tick_half_width, position + tick_half_width],
                    [fit_height, fit_height], color=SURFACE, lw=2.5,
                    solid_capstyle="round", zorder=4,
                )
                ax.plot(
                    [position - tick_half_width, position + tick_half_width],
                    [fit_height, fit_height], color=INK, lw=1.0,
                    solid_capstyle="round", zorder=5,
                )
            maximum = max(maximum, float(fit_heights.max()))
            maximum = max(maximum, float(heights.max()))
        ax.set_xticks(x)
        ax.set_xticklabels([r"$m/N=1/4$", r"$m/N=1/2$", r"$m/N=3/4$"])
        ax.set_ylabel(r"curvature loss $-\log_{10}H_{N,m}$")
        ax.set_ylim(0.0, maximum * 1.08)
        ax.grid(axis="x", visible=False)
        n_legend = ax.legend(
            loc="upper left", ncol=2, fontsize=5.8, handlelength=1.0,
            labelspacing=0.20, columnspacing=0.65, borderpad=0.25,
        )
        ax.add_artist(n_legend)
        convention_handles = [
            Patch(facecolor=MUTED, edgecolor=SURFACE, label="bars: exact"),
            Line2D([], [], color=INK, lw=1.0, label="ticks: fitted"),
        ]
        ax.legend(
            handles=convention_handles, loc="upper left",
            bbox_to_anchor=(0.0, 0.82), fontsize=5.8, handlelength=1.2,
            labelspacing=0.20, borderpad=0.25,
        )
        corner_note(ax, r"$\theta=0.8$, $w=1/2$",
                    xy=(0.97, 0.97), ha="right", va="top")

    # ------------------------------------------------------------------ (b)
    def draw_b(ax: Axes) -> None:
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        if not keys:
            ax.text(0.5, 0.5, "no retained schedules", transform=ax.transAxes,
                    ha="center", va="center", color=MUTED)
            return

        init_values = np.sort(selected_traj["lambda_init"].astype(float).unique())
        init_norm = plt.Normalize(float(init_values.min()), float(init_values.max()))
        init_cmap = CONTINUOUS_CMAP
        # Keep the three trajectory facets at their original aspect ratio.  A
        # wider standalone canvas supplies the right margin for the vertical
        # colorbar rather than compressing the data axes.
        gap = 0.050
        left, right = 0.075, 0.891
        width = (right - left - gap * (len(keys) - 1)) / len(keys)
        w = float(selected_traj["w"].iloc[0])
        max_step = float(selected_traj["step"].max())
        tick_values = [0.0, max_step]
        tick_positions = np.log10(1.0 + np.asarray(tick_values, dtype=float))

        def step_tick_label(value: float) -> str:
            if value == 0.0:
                return "0"
            exponent = int(math.floor(math.log10(value)))
            coefficient = value / (10.0 ** exponent)
            if math.isclose(coefficient, 1.0):
                return rf"$10^{{{exponent}}}$"
            return rf"${coefficient:g}\times10^{{{exponent}}}$"

        style_handles = [
            Line2D([], [], color=SECONDARY, lw=1.15, ls="-",
                   label="population GD"),
            Line2D([], [], color=SECONDARY, lw=0.85, ls=(0, (2, 1.4)),
                   label="SGD"),
            Line2D([], [], color=INK, lw=0.8, ls=(0, (5, 2)),
                   label=rf"truth $w={w:g}$"),
        ]

        for index, key in enumerate(keys):
            subax = ax.inset_axes([left + index * (width + gap), 0.24, width, 0.68])
            local = selected_traj[_mode_weight_key_mask(selected_traj, [key])]
            population = local[local["method"].astype(str) == "population_gd"]
            for init, group in population.groupby("lambda_init"):
                group = group.sort_values("step")
                subax.plot(
                    np.log10(1.0 + group["step"].to_numpy(dtype=float)),
                    group["lambda_value"],
                    color=init_cmap(init_norm(float(init))), lw=1.15,
                    alpha=0.92, zorder=3,
                )

            stochastic = local[local["method"].astype(str) == "sgd"]
            for init, init_group in stochastic.groupby("lambda_init"):
                color = init_cmap(init_norm(float(init)))
                if key[2] == 0.0:
                    for _, run in init_group.groupby("run_id"):
                        run = run.sort_values("step")
                        subax.plot(
                            np.log10(1.0 + run["step"].to_numpy(dtype=float)),
                            run["lambda_value"], color=color,
                            lw=0.7, ls=(0, (2, 1.4)), alpha=0.42, zorder=2,
                        )
                else:
                    quantiles = (
                        init_group.groupby("step")["lambda_value"]
                        .quantile([0.1, 0.5, 0.9]).unstack()
                    )
                    if len(quantiles):
                        x = np.log10(1.0 + quantiles.index.to_numpy(dtype=float))
                        subax.fill_between(
                            x, quantiles[0.1].to_numpy(dtype=float),
                            quantiles[0.9].to_numpy(dtype=float), color=color,
                            alpha=0.14, linewidth=0, zorder=1,
                        )
                        subax.plot(
                            x, quantiles[0.5].to_numpy(dtype=float), color=color,
                            lw=0.8, ls=(0, (2, 1.4)), alpha=0.9, zorder=2,
                        )

            subax.axhline(w, color=INK, lw=0.8, ls=(0, (5, 2)), zorder=0)
            subax.set_xlim(0.0, math.log10(1.0 + max_step))
            subax.set_xticks(tick_positions)
            subax.set_xticklabels([step_tick_label(value) for value in tick_values])
            boundary_labels = subax.get_xticklabels()
            if boundary_labels:
                boundary_labels[0].set_ha("left")
                boundary_labels[-1].set_ha("right")
            subax.set_ylim(-0.02, 1.02)
            subax.set_title(_mode_weight_label(*key), fontsize=7.4, pad=3,
                            color=schedule_colors.get(key, SECONDARY))
            if index == 0:
                subax.set_ylabel(r"estimate $\widehat\lambda_t$")
            else:
                subax.tick_params(labelleft=False)
            if index == len(keys) - 1:
                subax.legend(
                    handles=style_handles, loc="lower right", ncol=1,
                    frameon=False, fontsize=5.8, borderaxespad=0.45,
                    handlelength=1.8, handletextpad=0.5, labelspacing=0.28,
                )
        ax.text(0.5 * (left + right), 0.13, "optimization step (log-compressed)",
                 transform=ax.transAxes, ha="center", va="center", fontsize=7.0)

        cax = ax.inset_axes([0.910, 0.31, 0.017, 0.50])
        scalar = plt.cm.ScalarMappable(norm=init_norm, cmap=init_cmap)
        colorbar = ax.figure.colorbar(scalar, cax=cax, orientation="vertical")
        colorbar.set_ticks(init_values)
        colorbar.ax.tick_params(labelsize=6.0, length=2)
        colorbar.ax.set_title("initial\n" r"$\lambda_0$", fontsize=6.3, pad=4)

    # ------------------------------------------------------------------ (c)
    def draw_c(ax: Axes) -> None:
        frame = selected_runs.copy()
        if "H" not in frame:
            lookup = schedules.drop_duplicates(["N", "s", "pi"])[
                ["N", "s", "pi", "H", "inverse_H", "predicted_half_time"]
            ]
            frame = frame.merge(lookup, on=["N", "s", "pi"], how="left")
        max_steps = float(frame["max_steps"].max()) if "max_steps" in frame else float(
            frame["steps_completed"].max()
        )

        summary = art["mode_weight_half_times"]
        assert summary is not None
        summary = summary[summary["N"] == selected_N].copy()
        if keys:
            summary = summary[_mode_weight_key_mask(summary, keys)]
        summary_x = "inverse_H" if "inverse_H" in summary else "inverse_curvature"
        local_population = summary[
            (summary["method"] == "population_gd")
            & np.isclose(
                np.abs(summary["lambda_init"].astype(float) - summary["w"].astype(float)),
                0.1,
            )
        ]
        stochastic = summary[summary["method"] == "sgd"]
        x = np.arange(len(keys), dtype=float)
        width = 0.32
        methods = ((local_population, "population GD", -0.52 * width, True),
                   (stochastic, "SGD median", 0.52 * width, False))
        inverse_values = []
        for key_index, key in enumerate(keys):
            key_summary = summary[_mode_weight_key_mask(summary, [key])]
            inverse_values.append(float(key_summary[summary_x].median()))
            color = schedule_colors[key]
            for data, _, offset, filled in methods:
                group = data[_mode_weight_key_mask(data, [key])]
                if group.empty:
                    continue
                finite = group[
                    ~group["median_censored"].astype(bool)
                    & np.isfinite(group["median_t_half"].astype(float))
                ]
                censored = finite.empty
                value = max_steps if censored else float(finite["median_t_half"].median())
                height = math.log10(value)
                position = float(key_index) + offset
                ax.bar(
                    [position], [height], width=width, color=(color if filled else SURFACE),
                    edgecolor=color, linewidth=1.0,
                    zorder=3,
                )
                if censored:
                    ax.annotate(
                        "", xy=(position, height + 0.46),
                        xytext=(position, height + 0.12),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9,
                                        shrinkA=0, shrinkB=0, mutation_scale=7),
                    )
                    continue
                lows = finite["t_half_ci_low"].astype(float)
                highs = finite["t_half_ci_high"].astype(float)
                lo = float(lows[np.isfinite(lows)].min()) if np.isfinite(lows).any() else value
                hi = float(highs[np.isfinite(highs)].max()) if np.isfinite(highs).any() else value
                if 0.0 < lo <= value <= hi and (lo < value or hi > value):
                    ax.errorbar(
                        [position], [height],
                        yerr=[[height - math.log10(lo)], [math.log10(hi) - height]],
                        fmt="none", ecolor=color, elinewidth=0.8,
                        capsize=2.0, zorder=5,
                    )

        def inverse_text(value: float) -> str:
            exponent = int(math.floor(math.log10(value)))
            mantissa = value / (10.0 ** exponent)
            return rf"$1/H={mantissa:.1f}\times10^{{{exponent}}}$"

        labels = [
            _mode_weight_label(*key) + "\n" + inverse_text(inverse)
            for key, inverse in zip(keys, inverse_values)
        ]
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.1)
        for label, key in zip(ax.get_xticklabels(), keys):
            label.set_color(schedule_colors[key])
        upper = math.log10(max_steps) + 0.55
        ticks = np.arange(2, int(math.floor(upper)) + 1)
        ax.set_yticks(ticks)
        ax.set_yticklabels([rf"$10^{{{int(tick)}}}$" for tick in ticks])
        ax.set_ylim(0.0, upper)
        ax.set_ylabel(r"half-time $t_{1/2}$ (steps)")
        ax.grid(axis="x", visible=False)
        method_handles = [
            Patch(facecolor=SECONDARY, edgecolor=SECONDARY, label="population GD"),
            Patch(facecolor=SURFACE, edgecolor=SECONDARY, label="SGD median"),
            Line2D([], [], color=SECONDARY, linestyle="none",
                   marker=r"$\uparrow$", markersize=6, label="right-censored"),
        ]
        ax.legend(handles=method_handles, loc="upper right", fontsize=6.2,
                  bbox_to_anchor=(1.015, 1.0),
                  labelspacing=0.55, handlelength=0.8, handletextpad=0.4)

    # The calibration bars benefit from horizontal separation, so this panel
    # is deliberately wider and shorter than its previous near-square canvas.
    calibration_panel = bumped(4.9, 3.2)
    half_time_panel = M_PANEL
    # At 0.96\linewidth this panel is displayed at three times the width of an
    # M_PANEL summary.  A roughly three-times-larger aspect ratio therefore
    # gives it the same printed height as the first-row panels.  The small width
    # increase and substantial height reduction also make each inset trajectory
    # wider and shorter without changing its data or axis conventions.
    trajectory_panel = bumped(10.4, 2.77)
    panels = [
        Panel("curvature_by_visibility", "Curvature versus visibility", draw_a,
              calibration_panel, M_TITLE_SIZE),
        Panel("mode_weight_trajectories", "Mode-weight trajectories", draw_b,
              trajectory_panel, M_TITLE_SIZE, tight=False),
        Panel("half_time_vs_inverse_curvature", "Convergence time versus inverse curvature", draw_c,
              half_time_panel, M_TITLE_SIZE),
    ]

    def compose(fig):
        grid = fig.add_gridspec(
            1, 3,
            width_ratios=(calibration_panel[0], trajectory_panel[0], half_time_panel[0]),
            wspace=0.24,
        )
        return [fig.add_subplot(grid[0, index]) for index in range(3)]

    render("mode_weight_optimization", panels, compose,
           (calibration_panel[0] + trajectory_panel[0] + half_time_panel[0],
            max(calibration_panel[1], trajectory_panel[1], half_time_panel[1])),
           out_dir, png, tight_kw=None)


def table_mode_weight_optimization(art: dict, out_dir: Path) -> pd.DataFrame:
    """Write the runner's pre-aggregated, censoring-aware appendix table."""
    table = art["mode_weight_half_times"]
    if not _has(table):
        raise ValueError("no mode-weight half-time table; run mode-weight-optimization")
    assert table is not None
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_dir / "table_mode_weight_optimization.csv", index=False)
    preferred = [
        "method", "schedule", "role", "s", "pi", "lambda_init", "H",
        "inverse_H", "predicted_half_time", "n_runs", "n_events",
        "n_censored", "median_t_half", "median_censored", "t_half_ci_low",
        "t_half_ci_high", "median_over_predicted",
    ]
    shown = [name for name in preferred if name in table]
    tex_table = table[shown] if shown else table
    latex = tex_table.to_latex(index=False, escape=False, float_format=lambda x: f"{x:.4g}")
    (out_dir / "table_mode_weight_optimization.tex").write_text(
        "% Direct q_lambda mode-weight optimization.\n" + latex,
        encoding="utf-8",
    )
    print(
        "  wrote table_mode_weight_optimization.csv / "
        "table_mode_weight_optimization.tex"
    )
    return table


# ==========================================================================
# Driver
# ==========================================================================


def make_all_figures(results_dir: Path | None = None, out_dir: Path | None = None,
                     only: list[str] | None = None, png: bool = False) -> None:
    apply_style()
    results_dir = Path(results_dir or RESULTS_DIR)
    out_dir = Path(out_dir or FIGURES_DIR)
    requested = set(only or [])
    unknown = requested.difference(FIGURE_NAMES)
    if unknown:
        raise ValueError(f"unknown figure name(s): {', '.join(sorted(unknown))}")
    art = _load(results_dir)

    todo = {
        "mode_blindness_and_recovery": lambda: (
            figure_mode_blindness_and_recovery(art, out_dir, png),
            table_mode_blindness_and_recovery(art, out_dir),
        ),
        "masked_discrepancy_profiles": lambda: figure_masked_discrepancy_profiles(
            art, out_dir, png
        ),
        "blindness_decay_rates": lambda: figure_blindness_decay_rates(
            art, out_dir, png
        ),
        "low_visibility_intervention": lambda: figure_low_visibility_intervention(
            art, out_dir, png
        ),
        "residual_uncertainty_by_visibility": lambda: (
            figure_residual_uncertainty_by_visibility(art, out_dir, png)
        ),
        "boundary_flow_mechanism": lambda: figure_boundary_flow_mechanism(
            art, out_dir, png
        ),
        "asymmetric_mixture_weights": lambda: figure_asymmetric_mixture_weights(
            art, out_dir, png
        ),
        "numerical_stability_audit": lambda: figure_numerical_stability_audit(
            art, out_dir, png
        ),
        "corpus_mode_uncertainty_diagnostics": lambda: (
            figure_corpus_mode_uncertainty_diagnostics(art, out_dir, png)
        ),
        "low_visibility_detection_cost": lambda: figure_low_visibility_sampling_cost(
            art, out_dir, png, part="detection"
        ),
        "low_visibility_loss_cost": lambda: figure_low_visibility_sampling_cost(
            art, out_dir, png, part="loss"
        ),
        "mode_weight_optimization": lambda: (
            figure_mode_weight_optimization(art, out_dir, png),
            table_mode_weight_optimization(art, out_dir),
        ),
    }
    for name, fn in todo.items():
        if only and name not in only:
            continue
        try:
            fn()
        except Exception as exc:
            if only:
                raise RuntimeError(
                    f"could not generate requested figure {name!r}: {exc}"
                ) from exc
            print(f"  [warn] figure {name} skipped: {type(exc).__name__}: {exc}")
