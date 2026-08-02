"""Figures.

One main-text figure (M, 2x2) plus appendix figures A1-A8, all regenerated
from the frozen artifacts in ``results/`` -- no figure recomputes anything.

Every figure is emitted twice: as a composed multi-panel PDF
``figures/figure_<name>.pdf``, and panel by panel into
``figures/figure_<name>/panel_<key>.pdf`` so single panels can be dropped into
a paper's subfigure slots.  A panel's ``draw`` callback therefore runs twice and
must touch nothing but its own ``Axes`` -- that is the invariant to preserve
when editing one.

**No text that LaTeX should own is drawn into a PDF.**  There are no banner
titles; the ``(a)`` / ``(b)`` panel titles appear only on the composed contact
sheet, never on a standalone panel, whose ``\subcaption`` supplies them; and
the narrative lives in the paper text and in the README, not in the artwork.
What panels *do* keep is run context -- ``N``,
``m_base``, the witness -- as muted in-axes corner notes, since that is data
provenance rather than caption prose.

Explicitly *not* produced (spec section 7.2): ``(m, lambda) -> log D`` heatmaps
and 3-D surfaces.  They duplicate A1 and A2 without adding information.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LogNorm
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
    E3_COLORS,
    E3_LABELS,
    E3_MARKERS,
    GRID,
    INK,
    MARKERS,
    MUTED,
    NULL_ACCENT,
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

__all__ = ["make_all_figures"]

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
#: panel at one size, so none needs rescaling relative to the others.  (A1's
#: total-variation strips and A2's two panels are deliberately different
#: shapes; everything else is uniform within its figure.)
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

#: figure_M's panels sit in a single row, so each gets ~0.32\linewidth (1.76in).
#: At that width there is no room for an in-panel legend or for the off-axis
#: band annotations, and matplotlib does not reflow them -- it overlaps them.
#: Compact mode drops that furniture and shortens the axis labels; the caption
#: carries what it said.  Set False to restore the wide, self-contained panels.
M_COMPACT = False
A8_PANEL = bumped(4.4, 3.3)
#: A8(c)-(e) sit three-across in the paper, in the same slot as figure_M's
#: panels, so they are authored at figure_M's width and therefore render at
#: the same type size.  A8(a)-(b) sit two-across and keep the wider size.
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

    key: str                                  # "a", "b", ... -> panel_<key>.pdf
    title: str                                # the letter is added on drawing
    draw: Callable[[Axes], None]              # must only touch the given Axes
    figsize: tuple[float, float]              # size when rendered standalone
    title_size: float | None = None
    title_weight: str | None = None
    show_letter: bool = True


def _grid(nrows: int, ncols: int, **kw):
    """A ``compose`` that lays panels out on a plain ``nrows x ncols`` grid."""

    def compose(fig):
        axes = fig.subplots(nrows, ncols, **kw)
        return list(np.atleast_1d(axes).ravel())

    return compose


def render(name: str, panels: list[Panel], compose, combined_figsize,
           out_dir: Path, png: bool = False, tight_kw: dict | None = None) -> None:
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
    for panel, ax in zip(panels, axes):
        panel.draw(ax)
        annotate_panel(ax, panel.key, panel.title, panel.title_size,
                       fontweight=panel.title_weight,
                       show_letter=panel.show_letter)
    for spare in axes[len(panels):]:      # ragged grids leave empty slots
        spare.set_axis_off()
    if tight_kw is not None:
        fig.tight_layout(**tight_kw)
    save(fig, out_dir / f"figure_{name}", png=png)

    for panel in panels:
        f1, ax1 = plt.subplots(figsize=panel.figsize)
        panel.draw(ax1)
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
    for name in (
        "constants", "profiles", "summary", "fits", "numerics",
        "boost", "boostfits", "uscale",
        "realdata", "realdata_fits", "realdata_stats", "realdata_calib",
        "realdata_form", "sampling", "audit", "training",
    ):
        try:
            out[name] = read_table(name, results_dir)
        except FileNotFoundError:
            out[name] = None
    if out["summary"] is None:
        raise FileNotFoundError(
            f"no E1 artifacts in {results_dir}; run `python scripts/run_all.py` first"
        )
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
# Figure M -- the single main-text figure
# ==========================================================================


def figure_M(art: dict, out_dir: Path, png: bool = False) -> None:
    summary, fits, profiles = art["summary"], art["fits"], art["profiles"]
    boost = art["boost"]
    law, param = MAIN_LAW
    w = MAIN_W

    sub = _sel(summary, law=law, param=param, w=w).sort_values(["N", "m"])
    all_N = sorted(sub.N.unique())
    bands = band_constants(w)
    pc = ProductConstants(param, int(all_N[-1]))

    # ---------------------------------------------------------------- (a)
    null = _sel(summary, law=NULL_LAW[0], param=NULL_LAW[1], w=w)
    fit_rows = _sel(fits, law=law, param=param, w=w, target="Delta", xvar="m")

    def draw_a(ax: Axes) -> None:
        # y() maps a log-domain value onto whichever axis convention is in use,
        # so the two renderings share one code path and cannot drift apart.
        y = np.exp if M_A_LOG_Y else (lambda v: v)

        # Null control first, so it sits behind everything.  It is flat on this
        # scale; that flatness is the mechanism-absent signature, so it gets an
        # accent colour rather than receding into grey.
        for _, g in null.sort_values("m").groupby("N"):
            ax.plot(g.m, y(g.logDelta), color=NULL_ACCENT, lw=1.3,
                    ls=(0, (1, 1.5)), zorder=6)

        # For Law P the curves for different N coincide to within eta_N, so
        # they draw as one trunk, separated only by their endpoint markers.
        for N, g in sub.groupby("N"):
            c = n_color(N, all_N)
            ax.plot(g.m, y(g.logDelta), color=c, lw=1.4, zorder=3)
            ax.plot(g.m, y(g.logU), color=c, lw=0.8, ls=(0, (1, 1.4)),
                    alpha=0.9, zorder=2)
            ax.plot([g.m.iloc[-1]], [y(g.logDelta.iloc[-1])], marker="o", ms=4.2,
                    color=c, zorder=5)
            ax.annotate(f"$N={int(N)}$",
                        xy=(g.m.iloc[-1], y(g.logDelta.iloc[-1])),
                        xytext=(5, 3), textcoords="offset points", fontsize=6.6,
                        color=c, ha="left", va="bottom")

        for row in fit_rows.itertuples():
            xs = np.array([row.fit_lo, row.fit_hi])
            ax.plot(xs, y(row.intercept + row.slope * xs), color=INK, lw=1.0,
                    ls=(0, (4, 2)), alpha=0.55, zorder=4)

        m_ref = np.array([0.0, float(sub.m.max())])
        anchor = float(sub[sub.m == 0].logDelta.iloc[0])
        ax.plot(m_ref, y(anchor - pc.rate_star * m_ref), color=INK, lw=1.0,
                ls=(0, (6, 2, 1, 2)), alpha=0.75, zorder=4)

        ax.set_xlabel(r"visible size $m$" if M_COMPACT
                      else r"visible coordinates $m=|V|$")
        if M_A_LOG_Y:
            ax.set_yscale("log")
            ax.set_ylabel(r"$\Delta_{N,m}$")
            # round decades; matplotlib's default picks 21-decade steps here
            lo = float(np.exp(sub.logDelta.min()))
            ax.set_yticks([10.0 ** e for e in
                           range(0, int(np.floor(np.log10(lo))) - 1, -40)])
            ax.yaxis.set_minor_locator(plt.NullLocator())
        else:
            ax.set_ylabel(r"$\log\,\Delta_{N,m}$   (nats)")
        ax.set_xlim(-30, float(sub.m.max()) * 1.16)
        if M_COMPACT:
            return
        c0 = n_color(all_N[-1], all_N)
        ax.legend(
            [
                Line2D([], [], color=c0, lw=1.4),
                Line2D([], [], color=c0, lw=0.8, ls=(0, (1, 1.4))),
                Line2D([], [], color=INK, lw=1.0, ls=(0, (4, 2)), alpha=0.55),
                Line2D([], [], color=INK, lw=1.0, ls=(0, (6, 2, 1, 2)), alpha=0.75),
                Line2D([], [], color=NULL_ACCENT, lw=1.3, ls=(0, (1, 1.5))),
            ],
            [
                r"$\Delta_{N,m}$" if M_A_LOG_Y else r"$\log\Delta_{N,m}$",
                r"$U_{N,m}$" if M_A_LOG_Y else r"$\log U_{N,m}$",
                "fitted window",
                r"slope $-\mathrm{rate}^\star$", r"null ($\beta=0.5$)",
            ],
            # one column: the empty region is the triangle *below* the trunk,
            # which is tall and narrow, not wide
            loc="lower left", ncol=1,
        )

    # ---------------------------------------------------------------- (b)
    curves = list(_rho_curves(profiles, summary, w, law, param))
    rho_lo = min(float(r.min()) for _, _, _, r in curves)
    rho_hi = max(float(r.max()) for _, _, _, r in curves)

    def draw_b(ax: Axes) -> None:
        ax.axhline(bands["rho_limit"], color=CAT[1], lw=1.0, ls=(0, (4, 2)),
                   alpha=0.85, zorder=1)
        for N, m, lam, rho in curves:
            ax.plot(lam, rho, color=n_color(N, all_N), lw=0.55, alpha=0.75, zorder=2)
        ax.plot([w], [bands["rho_limit"]], marker="*", ms=10, color=CAT[1],
                zorder=5)

        # The measured range spans a factor of 1.1, where a log axis only buys
        # unreadable "4.5 x 10^0" tick labels.
        span = rho_hi - rho_lo
        ax.set_ylim(rho_lo - 0.35 * span, rho_hi + 0.35 * span)

        # The certified band is ~4 decades wide because the proof of Lem. 4.1
        # applies an e^{+-delta_I} variance comparison twice; drawing it to
        # scale would leave the panel empty.  Both edges are therefore shown
        # off-axis with their exact margins.
        if not M_COMPACT:
            ax.annotate(
                rf"$\rho\leq C_I={bands['C_I']:.0f}$  "
                rf"(${bands['C_I']/rho_hi:.0f}\times$ above)",
                xy=(0.5, 0.995), xytext=(0.5, 0.905), xycoords="axes fraction",
                textcoords="axes fraction", ha="center", va="center",
                fontsize=6.8, color=CAT[0],
                arrowprops=dict(arrowstyle="-|>", color=CAT[0], lw=0.9,
                                shrinkA=1.5, shrinkB=0),
            )
            ax.annotate(
                # 4 decimals, not 3: c_I = 0.0493827... and rounding to 0.049
                # loses the digit that makes the 81x margin reproducible
                rf"$\rho\geq c_I={bands['c_I']:.4f}$  "
                rf"(${rho_lo/bands['c_I']:.0f}\times$ below)",
                xy=(0.5, 0.005), xytext=(0.5, 0.095), xycoords="axes fraction",
                textcoords="axes fraction", ha="center", va="center",
                fontsize=6.8, color=CAT[0],
                arrowprops=dict(arrowstyle="-|>", color=CAT[0], lw=0.9,
                                shrinkA=1.5, shrinkB=0),
            )
        # Above the star, not below the line: below it the label runs into the
        # lower band annotation, which is centred and reaches left.  Directly
        # over lambda = w is the one clear spot -- every curve attains its
        # minimum there, so they fan away from it on both sides.
        # mathtext, unlike LaTeX, requires the braced argument: \mathcal I fails
        ax.annotate(rf"$\mathcal{{I}}(w)={bands['rho_limit']:.0f}$" if M_COMPACT
                    else rf"exact limit $\rho(w)={bands['rho_limit']:.0f}$",
                    xy=(w, bands["rho_limit"]),
                    xytext=(0, 9), textcoords="offset points",
                    ha="center", va="bottom", color=CAT[1], fontsize=6.8)

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
            ax.text(0.5, 0.5, "E2 not run", ha="center", va="center",
                    color=MUTED, transform=ax.transAxes)
            ax.set_axis_off()
            return
        for j, s in enumerate([0, 1, 4]):
            gs = g0[(g0.s == s) & (g0.pi > 0)].sort_values("pi")
            if gs.empty:
                continue
            free, cens = gs[~gs.censored], gs[gs.censored]
            lbl = r"$s=0$ (full mask)" if s == 0 else rf"$s={s}$"
            ax.plot(free.pi, free.r_eps, color=CAT[j], marker=MARKERS[j],
                    ms=4.0, lw=1.3, label=lbl, zorder=3)
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
        ax.set_ylabel(r"$r(\varepsilon=10^{-4})$" if M_COMPACT
                      else r"recovery radius $r(\varepsilon=10^{-4})$")

        # The censored cells all sit at r = c0/2, the half-width of I: there the
        # discrepancy is below eps across the *whole* admissible interval, so no
        # mode weight in I is distinguishable from w.  That is mode blindness in
        # E2's own language, not a failed measurement -- so the saturation line
        # is drawn and named rather than left as an unexplained plateau.
        a, b = mode_weight_interval(w)
        ax.axhline(b - w, color=MUTED, lw=0.7, ls=(0, (1, 2)), zorder=0)
        ax.set_ylim(top=(b - w) * 1.75)

        # no "censored" legend entry: three series contribute censored points
        # with three different markers, so a single proxy would mismatch them
        ax.legend(loc="lower left", ncol=1, fontsize=6.0, labelspacing=0.25,
                  handlelength=1.4, borderpad=0.25) if M_COMPACT else \
            ax.legend(loc="lower left", ncol=1, labelspacing=0.3)
        # a single decade of major ticks reads as an unlabelled axis
        ax.yaxis.set_minor_formatter(
            plt.FuncFormatter(lambda v, _: f"{v:g}" if v in (0.02, 0.05, 0.2) else "")
        )
        if not M_COMPACT:
            corner_note(ax, rf"$N={N_c}$,  $m_{{\rm base}}={m_base_c}$",
                        xy=(0.98, 0.97), va="top")

    # one size for all three, so they can be re-laid-out as subfigures
    panels = [
        Panel("a", r"Blindness scaling (Law P, $\theta=0.8$)", draw_a,
              M_PANEL, M_TITLE_SIZE),
        Panel("b", f"Sensitivity ratio ({len(curves)} cells)", draw_b,
              M_PANEL, M_TITLE_SIZE),
        Panel("c", "Intervention", draw_c, M_PANEL, M_TITLE_SIZE),
    ]
    render("M", panels, _grid(1, 3), (3 * M_PANEL[0], M_PANEL[1] + 0.2),
           out_dir, png, tight_kw=dict(pad=0.9, w_pad=1.8))


# ==========================================================================
# Table M
# ==========================================================================


def table_M(art: dict, out_dir: Path) -> pd.DataFrame:
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
    tab.to_csv(out_dir / "table_M.csv", index=False)
    (out_dir / "table_M.tex").write_text(_table_M_tex(tab), encoding="utf-8")
    print("  wrote table_M.csv / table_M.tex")
    return tab


def _table_M_tex(tab: pd.DataFrame) -> str:
    lines = [
        r"% Table M -- measured vs predicted (Law P, w = 1/2).",
        r"% Uncertainties are NUMERICAL (fit window / KL scheme / precision),",
        r"% not statistical: E1 and E2 contain no sampling noise.",
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
# Appendix figures
# ==========================================================================


def figure_A1(art: dict, out_dir: Path, png: bool = False) -> None:
    """Profiles: rows = N, cols = regime (pinned | null), + a |w - lambda| strip."""
    profiles, summary = art["profiles"], art["summary"]
    w = MAIN_W
    regimes = [("pinned", MAIN_LAW), ("null", NULL_LAW)]
    Ns = [int(n) for n in
          sorted(_sel(summary, law=MAIN_LAW[0], param=MAIN_LAW[1], w=w).N.unique())]
    cmap = m_cmap()
    letters = "abcdefghijklmnop"

    def make_profile(N: int, law: str, param: float, show_legend: bool):
        def draw(ax: Axes) -> None:
            prof = _sel(profiles, law=law, param=param, w=w, N=N, grid="I")
            if prof.empty:
                ax.set_axis_off()
                return
            ms = sorted({m for m in (4, 16, N // 4, N // 2, (3 * N) // 4)
                         if m in set(prof.m.unique())})
            norm = plt.Normalize(np.log(4), np.log(max(ms)))
            lo_v, hi_v = np.inf, -np.inf
            for m in ms:
                g = prof[prof.m == m].sort_values("lambda")
                # D(w) = 0 exactly (test T8); keeping that point would stretch
                # the log axis by 300 decades and hide everything else.
                keep = g["lambda"].to_numpy() != w
                vals = np.exp(g.logD.to_numpy()[keep])
                lo_v, hi_v = min(lo_v, vals.min()), max(hi_v, vals.max())
                ax.plot(g["lambda"].to_numpy()[keep], vals,
                        color=cmap(norm(np.log(max(m, 4)))), lw=1.1, label=f"$m={m}$")
            ax.set_ylim(lo_v * 0.25, hi_v * 6.0)
            ax.axvline(w, color=MUTED, lw=0.7, ls=(0, (1, 2)))
            a, b = float(prof["lambda"].min()), float(prof["lambda"].max())
            for edge in (a, b):
                ax.axvspan(edge - 0.012, edge + 0.012, color=GRID, alpha=0.7, lw=0)
            ax.set_yscale("log")
            ax.set_xlabel(r"$\lambda$")
            ax.set_ylabel(r"$\mathfrak{D}_{N,m}(\lambda)$")
            if show_legend:
                # m is shared across the row; a light frame keeps the legend
                # readable where it sits over curves
                ax.legend(fontsize=6.2, ncol=2, loc="lower center",
                          columnspacing=0.8, handlelength=1.2, frameon=True,
                          framealpha=0.88, edgecolor="none", facecolor=SURFACE)

        return draw

    def make_strip(law: str, param: float):
        def draw(ax: Axes) -> None:
            lam = np.linspace(*_sel(profiles, law=law, param=param, w=w, grid="I")
                              ["lambda"].agg(["min", "max"]).values, 201)
            ax.plot(lam, np.abs(w - lam), color=CAT[1], lw=1.4)
            ax.axvline(w, color=MUTED, lw=0.7, ls=(0, (1, 2)))
            ax.set_xlabel(r"$\lambda$")
            ax.set_ylabel(r"$D_{\rm TV}$", fontsize=7.5)

        return draw

    panels: list[Panel] = []
    k = 0
    for N in Ns:
        for regime, (law, param) in regimes:
            panels.append(Panel(letters[k], f"$N={N}$, {regime}",
                                make_profile(N, law, param, regime == "pinned"),
                                (3.7, 2.6), title_weight="normal",
                                show_letter=False))
            k += 1
    for regime, (law, param) in regimes:
        panels.append(Panel(letters[k], f"total variation, {regime}",
                            make_strip(law, param), (3.7, 1.8),
                            title_weight="normal", show_letter=False))
        k += 1

    def compose(fig):
        gs = fig.add_gridspec(len(Ns) + 1, 2,
                              height_ratios=[1.0] * len(Ns) + [0.40],
                              hspace=0.72, wspace=0.32)
        axes = [fig.add_subplot(gs[row, col])
                for row in range(len(Ns)) for col in range(2)]
        axes += [fig.add_subplot(gs[len(Ns), col]) for col in range(2)]
        return axes

    render("A1", panels, compose, (7.0, 1.85 * len(Ns) + 1.5), out_dir, png)


def figure_A2(art: dict, out_dir: Path, png: bool = False) -> None:
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
            name = "Law P" if law == "product" else "Law C"
            ax.errorbar(g.N, g.rate_hat, yerr=yerr, marker=MARKERS[j], ms=4.2,
                        capsize=2.0, elinewidth=0.9,
                        label=f"{name}, {sym}$={param:g}$", **st)
            ref = (-0.5 * math.log(1 - param**2)) if law == "product" else (
                (lambda ms: -0.5 * math.log(1 - ms**2) if ms > 0 else None)(
                    curie_weiss_magnetisation(param))
            )
            if ref is not None:
                ax.axhline(ref, color=st["color"], lw=0.8, ls=(0, (5, 2)), alpha=0.55)
                # For Law C this is a *reference* obtained by substituting the
                # spontaneous magnetisation into the Chernoff formula; the paper
                # makes no prediction there and nothing is fitted to it.
                tag = r"\mathrm{rate}^\star" if law == "product" else r"\mathrm{ref.}"
                ax.annotate(rf"${tag}={ref:.3f}$",
                            xy=(0.02, ref), xycoords=("axes fraction", "data"),
                            ha="left", va="bottom", fontsize=6.8, color=st["color"])
        ax.set_xscale("log")
        ax.set_xticks(sorted(fits.N.dropna().unique()))
        ax.xaxis.set_minor_locator(plt.NullLocator())   # minor log ticks collide
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("$N$")
        ax.set_ylabel(r"fitted decay rate $\hat c_N$")
        ax.legend(loc="lower right", fontsize=7)

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
                        lw=0, label=rf"$\varrho={frac:g}$")
            xs = np.array([g.fit_lo.iloc[0], g.fit_hi.iloc[0]])
            ax.plot(xs, g.intercept.iloc[0] + g.slope.iloc[0] * xs,
                    color=CAT[j % len(CAT)], lw=1.2)
        ax.set_xlabel("$N$")
        ax.set_ylabel(r"$\log\Delta_{N,\lfloor\varrho N\rfloor}$")
        ax.legend(loc="lower left", fontsize=7)

    panels = [
        Panel("a", "measured rate vs prediction", draw_a, bumped(5.0, 3.3)),
        Panel("b", "decay at fixed visible fraction", draw_b, bumped(4.0, 3.3)),
    ]
    render("A2", panels, _grid(1, 2, gridspec_kw={"width_ratios": [1.55, 1.0]}),
           (8.4, 3.3), out_dir, png, tight_kw={})


def figure_A3(art: dict, out_dir: Path, png: bool = False) -> None:
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
                    label=rf"$s={s}$")
            ax.plot(g.pi, g.kappa_fit, color=CAT[j % len(CAT)], lw=0,
                    marker=MARKERS[j], ms=4.2, mfc="none")
        ref = g_all[(g_all.s == 0) & (g_all.pi > 0)].sort_values("pi")
        if not ref.empty:
            p0, k0 = float(ref.pi.iloc[-1]), float(ref.kappa_exact.iloc[-1])
            pp = np.array([float(ref.pi.min()), p0])
            ax.plot(pp, k0 * (pp / p0), color=INK, lw=0.9, ls=(0, (4, 2)), alpha=0.6)
            ax.annotate("slope $1$", xy=(0.06, 0.82), xycoords="axes fraction",
                        fontsize=8.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\pi_s(\mu)$")
        ax.set_ylabel(r"$\varkappa(\pi,s)$")
        h, l = ax.get_legend_handles_labels()
        h += [Line2D([], [], color=MUTED, lw=1.3),
              Line2D([], [], color=MUTED, lw=0, marker="o", ms=4.2, mfc="none")]
        l += ["exact", "numerical fit"]
        ax.legend(h, l, loc="lower right", ncol=2, columnspacing=0.9, fontsize=8.0)
        corner_note(ax, note, xy=(0.03, 0.97), ha="left", va="top", fontsize=7.5)

    def draw_b(ax: Axes) -> None:
        for j, s in enumerate(sorted(g_all.s.unique())):
            g = g_all[(g_all.s == s) & (np.isclose(g_all.pi, 1e-2))
                      & (~g_all.censored)].sort_values("eps")
            if g.empty:
                continue
            ax.plot(g.eps, g.r_eps, color=CAT[j % len(CAT)], marker=MARKERS[j],
                    ms=3.8, lw=1.3, label=rf"$s={s}$")
        g = g_all[(g_all.s == 0) & (np.isclose(g_all.pi, 1e-2)) & (~g_all.censored)]
        if not g.empty:
            e0 = float(g.eps.max())
            r0 = float(g[g.eps == e0].r_eps.iloc[0])
            ee = np.array([float(g.eps.min()), e0])
            ax.plot(ee, r0 * (ee / e0) ** 0.5, color=INK, lw=0.9, ls=(0, (4, 2)),
                    alpha=0.6)
            ax.annotate("slope $+1/2$", xy=(0.06, 0.82), xycoords="axes fraction",
                        fontsize=8.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\varepsilon$")
        ax.set_ylabel(r"$r(\varepsilon)$")
        ax.legend(loc="lower right", fontsize=8.0)
        corner_note(ax, note, xy=(0.03, 0.97), ha="left", va="top", fontsize=7.5)

    def draw_c(ax: Axes) -> None:
        prof = _sel(profiles, law=law, param=param, w=w, N=N, grid="I")
        base = prof[prof.m == m_base].sort_values("lambda")
        lam = base["lambda"].to_numpy()
        keep = lam != w                  # D(w) = 0 exactly; see figure_A1
        vals = np.exp(base.logD.to_numpy()[keep])
        lo_v, hi_v = vals.min(), vals.max()
        ax.plot(lam[keep], vals, color=MUTED, lw=1.4, ls=(0, (5, 2)),
                label=rf"unboosted ($m={m_base}$)")
        pi = 1e-2
        for j, s in enumerate([0, 1, 4]):
            bs = prof[prof.m == s].sort_values("lambda")
            if bs.empty:
                continue
            mixed = np.logaddexp(math.log1p(-pi) + base.logD.to_numpy(),
                                 math.log(pi) + bs.logD.to_numpy())
            vals = np.exp(mixed[keep])
            lo_v, hi_v = min(lo_v, vals.min()), max(hi_v, vals.max())
            ax.plot(lam[keep], vals, color=CAT[j], lw=1.2,
                    ls=[(0, ()), (0, (5, 1.5)), (0, (1, 1.4))][j], label=rf"$s={s}$")
        ax.set_yscale("log")
        ax.set_ylim(lo_v * 0.2, hi_v * 20.0)
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"$\mathfrak{D}_{\rm boost}(\lambda)$")
        ax.legend(loc="lower left", fontsize=8.0)
        corner_note(ax, note, xy=(0.97, 0.55), ha="right", va="center", fontsize=7.5)

    A3_PANEL = bumped(3.8, 3.1)
    panels = [
        Panel("a", "curvature", draw_a, A3_PANEL, M_TITLE_SIZE),
        Panel("b", r"radius vs tolerance ($\pi=10^{-2}$)", draw_b, A3_PANEL, M_TITLE_SIZE),
        Panel("c", r"curvature restoration ($\pi=10^{-2}$)", draw_c, A3_PANEL, M_TITLE_SIZE),
    ]
    render("A3", panels, _grid(1, 3), (9.4, 3.1), out_dir, png, tight_kw={})


def figure_A4(art: dict, out_dir: Path, png: bool = False) -> None:
    """Low-visibility uncertainty: ``log U_{N,s}`` vs ``s`` with the exact anchor."""
    uscale = art["uscale"]
    if not _has(uscale):
        return
    law, param, w = MAIN_LAW[0], MAIN_LAW[1], MAIN_W
    g_all = _sel(uscale, law=law, param=param, w=w)
    Ns = sorted(g_all.N.unique())

    def draw_a(ax: Axes) -> None:
        for N in Ns:
            g = _sel(g_all, N=int(N)).sort_values("s")
            c = n_color(N, Ns)
            ax.plot(g.s, g.logU, color=c, marker="o", ms=4.0, lw=1.3, label=f"$N={N}$")
            ss = np.linspace(0, float(g.s.max()), 50)
            ax.plot(ss, np.log(g.u0_hat.iloc[0]) - g.L_hat.iloc[0] * ss, color=c,
                    lw=0.9, ls=(0, (4, 2)), alpha=0.7)

        ax.plot([0], [math.log(w * (1 - w))], marker="*", ms=11, color=CAT[1], zorder=5)
        ax.annotate(rf"exact anchor $U_{{N,0}}=w(1-w)={w*(1-w):.2f}$",
                    xy=(0, math.log(w * (1 - w))), xytext=(10, -4),
                    textcoords="offset points", fontsize=8.5, color=CAT[1], va="top")

        pc = ProductConstants(param, int(Ns[-1]))
        ss = np.linspace(0, float(g_all.s.max()), 50)
        ax.plot(ss, math.log(pc.u0) - pc.L * ss, color=INK, lw=1.1,
                ls=(0, (6, 2, 1, 2)), alpha=0.75,
                label=rf"certified $u_0e^{{-Ls}}$, $L=L_0+\eta_N={pc.L:.3f}$")
        ax.set_xlabel("$s$ (visible coordinates under the boost)")
        ax.set_ylabel(r"$\log U_{N,s}$")
        ax.legend(loc="lower left", fontsize=8.5)

    panels = [Panel("a", "low-visibility residual mode uncertainty", draw_a,
                    bumped(5.2, 3.5), M_TITLE_SIZE)]
    render("A4", panels, _grid(1, 1), bumped(5.2, 3.5), out_dir, png, tight_kw={})


def figure_A5(art: dict, out_dir: Path, png: bool = False) -> None:
    """Cross-mode flows alongside the blindness scaling, and under boosting."""
    summary, boost = art["summary"], art["boost"]
    law, param, w = MAIN_LAW[0], MAIN_LAW[1], MAIN_W
    sub = _sel(summary, law=law, param=param, w=w).sort_values(["N", "m"])
    Ns = sorted(sub.N.unique())

    def draw_a(ax: Axes) -> None:
        for N in Ns:
            g = _sel(sub, N=int(N))
            c = n_color(N, Ns)
            ax.plot(g.m, g.logFplus, color=c, lw=1.3, marker="o", ms=2.6,
                    label=f"$N={N}$")
            ax.plot(g.m, g.logDelta, color=c, lw=0.85, ls=(0, (1, 1.4)), alpha=0.85)
        ax.set_xlabel("$m$")
        ax.set_ylabel("log value (nats)")
        ax.legend(loc="lower left", ncol=2)

    have_boost = _has(boost)
    if have_boost:
        N_b = int(max(boost.N.unique()))
        gb = _sel(boost, law=law, param=param, w=w, N=N_b)
        m_base_b = int(sorted(gb.m_base.unique())[-1])
        gb = _sel(gb, m_base=m_base_b, eps=float(gb.eps.min()))

    def draw_b(ax: Axes) -> None:
        if not have_boost:
            ax.text(0.5, 0.5, "E2 not run", ha="center", va="center",
                    color=MUTED, transform=ax.transAxes)
            ax.set_axis_off()
            return
        for j, s in enumerate(sorted(gb.s.unique())):
            g = gb[(gb.s == s) & (gb.pi > 0)].sort_values("pi")
            ax.plot(g.pi, np.exp(g.logFplus_boost), color=CAT[j % len(CAT)],
                    marker=MARKERS[j], ms=3.8, lw=1.2, label=rf"$s={s}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$\pi_s(\mu)$")
        ax.set_ylabel(r"$F^+_{\rm boost}$")
        ax.legend(loc="lower right")
        corner_note(ax, rf"$N={N_b}$,  $m_{{\rm base}}={m_base_b}$",
                    xy=(0.03, 0.97), ha="left", va="top")

    panels = [
        Panel("a", r"flows $F^+_{N,m}$ (solid) vs $\Delta_{N,m}$ (dotted)",
              draw_a, bumped(4.3, 3.2)),
        Panel("b", "boosted flow", draw_b, bumped(4.3, 3.2)),
    ]
    render("A5", panels, _grid(1, 2), (8.2, 3.2), out_dir, png, tight_kw={})


def figure_A6(art: dict, out_dir: Path, png: bool = False) -> None:
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
        norm = plt.Normalize(np.log(4), np.log(max(ms)))
        lo_v, hi_v = np.inf, -np.inf
        for m in ms:
            g = prof[prof.m == m].sort_values("lambda")
            keep = g["lambda"].to_numpy() != w   # D(w) = 0 exactly; see figure_A1
            vals = np.exp(g.logD.to_numpy()[keep])
            lo_v, hi_v = min(lo_v, vals.min()), max(hi_v, vals.max())
            ax.plot(g["lambda"].to_numpy()[keep], vals,
                    color=cmap(norm(np.log(max(m, 4)))), lw=1.2, label=f"$m={m}$")
        ax.axvline(w, color=MUTED, lw=0.7, ls=(0, (1, 2)))
        ax.set_yscale("log")
        ax.set_ylim(lo_v * 0.25, hi_v * 6.0)
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"$\mathfrak{D}_{N,m}(\lambda)$")
        ax.legend(fontsize=6.6, loc="lower left", ncol=2)

    def draw_b(ax: Axes) -> None:
        curves = list(_rho_curves(profiles, summary, w, law, param))
        lo = min(float(r.min()) for _, _, _, r in curves)
        hi = max(float(r.max()) for _, _, _, r in curves)
        ax.axhline(bands["rho_limit"], color=CAT[1], lw=1.0, ls=(0, (4, 2)),
                   alpha=0.85, zorder=1)
        for Nc, m, lam, rho in curves:
            ax.plot(lam, rho, color=n_color(Nc, Ns), lw=0.55, alpha=0.75, zorder=2)
        ax.plot([w], [bands["rho_limit"]], marker="*", ms=9, color=CAT[1], zorder=5)
        span = hi - lo
        ax.set_ylim(lo - 0.35 * span, hi + 0.35 * span)
        ax.annotate(
            rf"$C_I={bands['C_I']:.3g}$  (${bands['C_I']/hi:.0f}\times$ above)",
            xy=(0.5, 0.995), xytext=(0.5, 0.905), xycoords="axes fraction",
            textcoords="axes fraction", ha="center", va="center",
            fontsize=6.8, color=CAT[0],
            arrowprops=dict(arrowstyle="-|>", color=CAT[0], lw=0.9,
                            shrinkA=1.5, shrinkB=0),
        )
        ax.annotate(
            rf"$c_I={bands['c_I']:.1e}$  (${lo/bands['c_I']:.0f}\times$ below)",
            xy=(0.5, 0.005), xytext=(0.5, 0.095), xycoords="axes fraction",
            textcoords="axes fraction", ha="center", va="center",
            fontsize=6.8, color=CAT[0],
            arrowprops=dict(arrowstyle="-|>", color=CAT[0], lw=0.9,
                            shrinkA=1.5, shrinkB=0),
        )
        ax.set_xlabel(r"$\lambda$")
        ax.set_ylabel(r"$\rho_{N,m}(\lambda)$")

    panels = [
        Panel("a", rf"profiles, $w=0.9$, $N={N}$", draw_a, bumped(4.3, 3.2)),
        Panel("b", rf"$\rho$ band, $\rho(w)={bands['rho_limit']:.1f}$", draw_b,
              bumped(4.3, 3.2)),
    ]
    render("A6", panels, _grid(1, 2), (8.4, 3.2), out_dir, png, tight_kw={})


def figure_A7(art: dict, out_dir: Path, png: bool = False) -> None:
    """Numerics audit: mpmath-vs-float64 error, and slope sensitivity."""
    audit, numerics, fits = art["audit"], art["numerics"], art["fits"]

    def draw_a(ax: Axes) -> None:
        if not _has(audit):
            ax.text(0.5, 0.5, "run scripts/audit_numerics.py", ha="center",
                    va="center", color=MUTED, transform=ax.transAxes)
            ax.set_axis_off()
            return
        for j, (impl, g) in enumerate(audit.groupby("impl")):
            lbl = ("branched (spec sec. 2.3, $T=30$)" if impl == "branched"
                   else "stable (default)")
            ax.scatter(g.logD_mp, np.abs(g.abs_err_log).clip(1e-18),
                       s=13, color=CAT[j % len(CAT)], marker=MARKERS[j],
                       alpha=0.8, label=lbl)
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
            ax.scatter(np.full(len(spread), i), np.abs(spread).clip(1e-18),
                       s=15, color=CAT[i % len(CAT)], alpha=0.75)
        ax.set_xticks(range(len(knobs)))
        ax.set_xticklabels(knobs, rotation=20, ha="right", fontsize=6.8)
        ax.set_yscale("log")
        ax.set_ylabel(r"$|\Delta\hat c_N|$")

    panels = [
        Panel("a", "float64 vs mpmath", draw_a, bumped(4.3, 3.2)),
        Panel("b", "slope sensitivity per knob", draw_b, bumped(4.3, 3.2)),
    ]
    render("A7", panels, _grid(1, 2), (8.4, 3.2), out_dir, png, tight_kw={})


def _e3_result_draw(art: dict, corpora, main_text: bool = False):
    r"""The E3 result panel, shared by A8(b) and the main-text `figure_M_E3`.

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
    rd, rdf = art["realdata"], art["realdata_fits"]

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
        if not (main_text and M_COMPACT):
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
            c = E3_COLORS.get(corpus, CAT[0])
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
                            color=c, marker=E3_MARKERS[arm], ms=3.6, lw=1.2,
                            capsize=1.5, elinewidth=0.7, zorder=3)
                if len(cens):
                    ax.plot(cens.V, cens.mmse_hat, color=c, marker=E3_MARKERS[arm],
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
                                xytext=(20, -8), textcoords="offset points",
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
                            fontsize=7.5, color=E3_COLORS["code_prose"])
            sec = rdf[(rdf.corpus == "bilingual") & (rdf.arm == "large")]
            if len(sec):
                ax.annotate(rf"$\hat\kappa={sec.kappa_hat.iloc[0]:.4f}$",
                            xy=(0.98, 0.78), xycoords="axes fraction", ha="right",
                            fontsize=7.5, color=E3_COLORS["bilingual"])

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(xmin, xmax)
        ax.set_xlabel(r"visible tokens $|V|$")
        ax.set_ylabel(r"$\widehat{\mathrm{mmse}}(Z\mid X_V)$")

        h = [Line2D([], [], color=E3_COLORS.get(c, CAT[0]), lw=2.0) for c in corpora]
        lab = [E3_LABELS.get(c, c) for c in corpora]
        if not (main_text and M_COMPACT):   # the caption explains the band
            h.append(Patch(facecolor=MUTED, alpha=0.3, lw=0))
            lab.append(r"band: $\widehat{\mathrm{mmse}}$ to Brier")
        # only the classifier arm can be censored, so the proxy carries that
        # arm's marker; an "o" proxy would name a symbol that appears nowhere
        if bool(rd.floored.any()):
            h.append(Line2D([], [], color=MUTED, marker=E3_MARKERS["large"],
                            lw=0, mfc="none", mew=1.0))
            lab.append("censored")
        ax.legend(h, lab, fontsize=fs - 0.4, loc="lower left", labelspacing=0.3,
                  handlelength=1.6, frameon=True, framealpha=0.85,
                  edgecolor="none", facecolor=SURFACE)

    return draw


def figure_M_E3(art: dict, out_dir: Path, png: bool = False) -> None:
    """The one E3 panel for the main text: the result plus its own validity check."""
    rd = art["realdata"]
    if not _has(rd):
        return
    corpora = list(dict.fromkeys(rd.corpus))
    # same size as figure_M's panels: this is the main-text E3 counterpart.
    fig, ax = plt.subplots(figsize=M_PANEL)
    _e3_result_draw(art, corpora, main_text=True)(ax)
    # No drawn-in title, for the same reason standalone panels have none: this
    # goes into a float whose \caption names it.
    fig.tight_layout()
    save(fig, out_dir / "figure_M_E3", png=png, tight=False)
    print("  figure_M_E3: single main-text panel")


def figure_A8(art: dict, out_dir: Path, png: bool = False) -> None:
    """E3 details: reliability, the Brier check, backoff order, corpus metadata."""
    rd, calib, stats = art["realdata"], art["realdata_calib"], art["realdata_stats"]
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
            col = E3_COLORS.get(corpus, CAT[0])
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

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"visible tokens $|V|$")
        ax.set_ylabel(r"Brier $/\ \widehat{\mathrm{mmse}}$")
        ax.annotate(rf"censored above ${thresh:g}\times$", xy=(0.03, thresh),
                    xycoords=ax.get_yaxis_transform(), ha="left", va="bottom",
                    fontsize=6.2, color=SECONDARY)
        # sits above the unit line at the second-largest |V|, where the total
        # slack has already lifted off it and left the gap clear
        x_lab = sorted(large.V.unique())[-2] if len(large.V.unique()) > 1 else 1.0
        ax.annotate("calibrated", xy=(x_lab, 1.0), xytext=(0, 5),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=6.2, color=SECONDARY)

        h = [Line2D([], [], color=E3_COLORS.get(c, CAT[0]), lw=2.0)
             for c in corpora]
        lab = [E3_LABELS.get(c, c) for c in corpora]
        h += [Line2D([], [], color=MUTED, lw=1.4, marker="o", ms=4.0),
              Line2D([], [], color=MUTED, lw=1.0, ls=(0, (4, 1.6)), marker="s",
                     ms=3.0, mfc="none"),
              Line2D([], [], color=MUTED, lw=0, marker="o", ms=4.0, mfc="none")]
        lab += ["total slack", "calibration only", "censored"]
        ax.legend(h, lab, fontsize=6.0, loc="upper left", ncol=2,
                  columnspacing=0.9, labelspacing=0.3)

    draw_b = _e3_result_draw(art, corpora, main_text=False)

    def draw_c(ax: Axes) -> None:
        """Realised backoff order -- the small arm's effective resolution."""
        g_all = rd[rd.arm == "small"]
        width = 0.38
        for i, corpus in enumerate(corpora):
            g = g_all[g_all.corpus == corpus].sort_values("V")
            if g.empty:
                continue
            x = np.arange(len(g)) + (i - 0.5 * (len(corpora) - 1)) * width
            ax.bar(x, g.eff_order, width=width * 0.92,
                   color=E3_COLORS.get(corpus, CAT[0]), alpha=0.85,
                   label=E3_LABELS.get(corpus, corpus))
        ref = g_all[g_all.corpus == corpora[0]].sort_values("V")
        ax.set_xticks(np.arange(len(ref)))
        ax.set_xticklabels([str(int(v)) for v in ref.V])
        ax.set_yticks(range(0, int(np.nanmax(g_all.eff_order)) + 2))
        ax.set_xlabel(r"$|V|$")
        ax.set_ylabel("realized backoff order")
        ax.legend(fontsize=6.2, loc="upper left")
        # one line, not two: at this panel width the longer form reaches across
        # into the legend
        corner_note(ax, f"capped below {int(stats.backoff_threshold.iloc[0])} counts"
                    if _has(stats) else "", xy=(0.97, 0.97), ha="right", va="top")

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
            c = E3_COLORS.get(corpus, CAT[0])
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
        h += [Line2D([], [], color=E3_COLORS.get(c, CAT[0]), lw=2.0) for c in corpora]
        lab += [E3_LABELS.get(c, c) for c in corpora]
        ax.legend(h, lab, fontsize=6.2, loc="upper right", ncol=1,
                  labelspacing=0.24, handlelength=1.2, handletextpad=0.5,
                  borderpad=0.3, frameon=True, framealpha=0.9,
                  edgecolor="none", facecolor=SURFACE)

    def draw_f(ax: Axes) -> None:
        """Which functional form fits: the exponential Asm. 4.1 assumes, or a
        power law?  Fitted on the cells uncensored for *every* feature set, so
        the model classes are compared like with like."""
        form = art["realdata_form"]
        if not _has(form) or "fit_range" not in form.columns:
            ax.text(0.5, 0.5, "re-run E3 for the form comparison", ha="center",
                    va="center", color=MUTED, transform=ax.transAxes)
            ax.set_axis_off()
            return
        f = form[(form.fit_range == "common") & (form.arm == "large")]
        rows = list(f.groupby(["corpus", "features"], sort=False))
        x = np.arange(len(rows))
        wid = 0.36
        for j, (model, col) in enumerate((("exponential", CAT[1]),
                                          ("power_law", CAT[0]))):
            vals = [float(g[g.model == model].R2.iloc[0]) for _, g in rows]
            ax.bar(x + (j - 0.5) * wid, vals, width=wid * 0.9, color=col,
                   alpha=0.85, label=model.replace("_", " "))
        ax.set_xticks(x)
        ax.set_xticklabels(
            [E3_LABELS.get(c, c) + "\n" + fs.replace("_", "+")
             for (c, fs), _ in rows], fontsize=7.0)
        ax.set_ylim(0.0, 1.18)
        ax.set_ylabel(r"fit $R^2$")
        ax.axhline(1.0, color=MUTED, lw=0.6, ls=(0, (1, 2)))
        ax.legend(fontsize=7.5, loc="lower left")
        expo = [float(g[g.model == "power_law"].rate_or_exponent.iloc[0])
                for _, g in rows]
        pw_r2 = [float(g[g.model == "power_law"].R2.iloc[0]) for _, g in rows]
        for xi, e, r in zip(x, expo, pw_r2):
            ax.annotate(rf"$V^{{{e:.2f}}}$", xy=(xi + 0.5 * wid, r + 0.02),
                        ha="center", va="bottom", fontsize=7.2, color=CAT[0])

    panels = [
        # Ordered result -> the two analyses that qualify it -> the two
        # diagnostics that license it (one per estimator arm).  All panels are
        # emitted at one size so they can be re-laid-out freely as subfigures;
        # the composed 2x3 is a contact sheet, not a layout prescription.
        # Corpora / licences / protocol are not a figure -- they live in the
        # README and the paper's appendix.
        Panel("a", "residual mode MMSE on real corpora", draw_b, A8_PANEL, M_TITLE_SIZE),
        Panel("b", "functional form of the decay", draw_f, A8_PANEL, M_TITLE_SIZE),
        Panel("c", "model class: does a richer classifier change the law?",
              draw_e, A8_NARROW, M_TITLE_SIZE),
        Panel("d", "slack in the upper bound, and its source", draw_a, A8_NARROW),
        Panel("e", "realized backoff order — posterior arm", draw_c, A8_NARROW),
    ]
    render("A8", panels, _grid(2, 3), (3 * A8_PANEL[0], 2 * A8_PANEL[1]),
           out_dir, png, tight_kw={})


# ==========================================================================
# Figure A9 -- the sampling cost of low-visibility mass (E4)
# ==========================================================================


def figure_A9(art: dict, out_dir: Path, png: bool = False) -> None:
    """E4: what the full-mask channel costs, in examples and in loss variance.

    Panels (a)-(c) are the *detection* cost -- how many masked examples it takes
    to see the discrepancy E1/E2 certify is nonzero.  Panels (d)-(e) are the
    *raw-loss* cost -- the per-example loss that enters the training objective.
    Colour means ``s`` in every panel that carries more than one schedule.
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
                    marker=MARKERS[j], ms=3.4, label=rf"$s={s}$")
        ref = end[(end.s == 0) & (end.pi > 0)].sort_values("pi")
        pp = ref.pi.to_numpy()
        _guide(ax, pp, float(ref.n_star.iloc[0]), float(pp[0]), -1.0,
               "slope $-1$")
        # An untruncated diffusion schedule puts only Theta(1/N) on the full
        # mask; that is the operating point the paper's Sec. 5.2 is about.
        ax.axvline(1.0 / N, color=NULL_ACCENT, lw=1.0, ls=(0, (1, 1.6)), zorder=1)
        ax.annotate(r"$\pi_0 = 1/N$", xy=(1.0 / N, 0.97),
                    xycoords=ax.get_xaxis_transform(), rotation=90,
                    ha="right", va="top", fontsize=7.5, color=NULL_ACCENT)
        blind = float(end[(end.s == 0) & (end.pi == 0)].log10_n_star.iloc[0])
        corner_note(ax, "mode-blind ($\\pi_0=0$):\n"
                        rf"$n^\star \approx 10^{{{blind:.0f}}}$",
                    xy=(0.97, 0.97), ha="right", va="top", fontsize=7.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"low-visibility schedule mass $\pi_s(\mu)$")
        ax.set_ylabel(r"$n^\star$ (masked examples)")
        ax.legend(loc="lower left", ncol=2, columnspacing=0.9, fontsize=8.0)

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
                    marker=MARKERS[j], ms=3.4, label=rf"$s={s}$")
        ref = sub[sub.s == 0].sort_values("lambda_err")
        xx = ref.lambda_err.to_numpy()
        _guide(ax, xx, float(ref.n_star.iloc[0]), float(xx[0]), -2.0,
               "slope $-2$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"mode-weight error $|\lambda - w|$")
        ax.set_ylabel(r"$n^\star$ (masked examples)")
        ax.legend(loc="lower left", ncol=2, columnspacing=0.9, fontsize=8.0)
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
        """The high-variance term of Sec. 5.2, priced exactly."""
        g = end[(end.s == 0) & (end.pi > 0)].sort_values("pi")
        base = float(end[(end.s == 0) & (end.pi == 0)].raw_sd.iloc[0])
        ax.plot(g.pi, g.raw_sd, color=CAT[0], lw=1.5, marker="o", ms=3.4,
                label=r"total  $\mathrm{sd}[L]$")
        ax.plot(g.pi, np.sqrt(g.raw_var_between.to_numpy()), color=CAT[1],
                lw=1.4, ls=(0, (4, 1.6)), marker="s", ms=3.4,
                label="full-mask channel alone")
        ax.axhline(base, color=MUTED, lw=0.9, ls=(0, (1, 1.6)), zorder=1)
        ax.annotate(rf"no full-mask channel ({base:.0f} nats)",
                    xy=(0.97, base), xycoords=ax.get_yaxis_transform(),
                    ha="right", va="bottom", fontsize=6.4, color=SECONDARY)
        cross = g[g.raw_var_schedule_frac >= 0.5]
        if not cross.empty:
            ax.annotate("schedule term overtakes\nthe within-mask term",
                        xy=(float(cross.pi.iloc[0]), float(cross.raw_sd.iloc[0])),
                        xytext=(0.30, 0.72), textcoords="axes fraction",
                        fontsize=6.4, color=SECONDARY, ha="left",
                        arrowprops=dict(arrowstyle="-", lw=0.7, color=MUTED))
        ax.set_xscale("log")
        ax.set_xlabel(r"full-mask schedule mass $\pi_0$")
        ax.set_ylabel(r"sd of raw loss $L$ (nats)")
        ax.legend(loc="upper left", fontsize=6.8)
        corner_note(ax, note + ",  $s=0$", xy=(0.97, 0.03), ha="right", va="bottom")

    panels = [
        # detection cost (a-c), then raw-loss cost (d-e).  All one size, so the
        # composed 2x3 is a contact sheet rather than a layout prescription.
        Panel("a", "examples needed to detect the discrepancy", draw_a, A8_PANEL, M_TITLE_SIZE),
        Panel("b", "signal grows in $\\pi_0$, noise only in $\\sqrt{\\pi_0}$",
              draw_b, A8_PANEL, M_TITLE_SIZE),
        Panel("c", "cost of resolving a smaller mode-weight error", draw_c, A8_PANEL, M_TITLE_SIZE),
        Panel("d", "raw loss per example, by mask size", draw_d, A8_PANEL),
        Panel("e", "variance of the raw per-example loss", draw_e, A8_PANEL),
    ]
    render("A9", panels, _grid(2, 3), (3 * A8_PANEL[0], 2 * A8_PANEL[1]),
           out_dir, png, tight_kw={})


# ==========================================================================
# A10: E5 -- optimisation sanity check
# ==========================================================================

#: pi is ordered, so E5's schedules take a sequential ramp rather than four
#: categorical slots.  Anchored on the paper's CAT[0] violet and run out through
#: blue and teal to a light green, so it reads as part of the same palette as
#: every other figure.  Adjacent-pair CVD dE 11.9 (deutan), and every step
#: clears the blind red comfortably (worst dE 8.0 deutan, 29.6 normal vision).
#: Two residuals, both relieved by the legend and the per-series markers: the
#: teal-to-green adjacency is dE 14.9 against a floor of 15, and the lightest
#: step is below 3:1 on white.  Green is unavoidable -- with blind on red, any
#: warm fourth hue collides with it outright (dE 13.7).
E5_PI_RAMP: tuple[str, ...] = ("#4a3aa7", "#2077b0", "#12a08c", "#8fc740")


def figure_A10(art: dict, out_dir: Path, png: bool = False) -> None:
    """E5: training curves, mode-weight recovery, and final error vs pi_0."""
    training = art["training"]
    if training is None or training.empty:
        raise ValueError("no E5 training data")

    w_true = float(training["w"].iloc[0])

    # The blind schedule is a different kind of object from the rest -- it is
    # the mechanism-absent control -- so it takes the palette red and a dashed
    # stroke, and is excluded from the ramp the others share.  Those others
    # differ only in pi, an *ordered* quantity, so they take the one-hue N ramp
    # (dark = least mass) rather than categorical slots that would imply four
    # unrelated conditions.  Amber is deliberately absent: with blind on red,
    # every warm fourth hue collided with it (normal-vision dE 13.7).
    pi_of = training.groupby("schedule")["pi"].first().to_dict()
    blind_names = sorted(s for s in pi_of if pi_of[s] <= 0)
    informative = sorted((s for s in pi_of if pi_of[s] > 0), key=lambda s: pi_of[s])
    sched_order = blind_names + informative

    sched_colors: dict[str, str] = {}
    sched_markers: dict[str, str] = {}
    sched_styles: dict[str, object] = {}
    for s in blind_names:
        sched_colors[s] = CAT[1]
        sched_markers[s] = "x"
        sched_styles[s] = (0, (5, 2))
    for i, s in enumerate(informative):
        sched_colors[s] = E5_PI_RAMP[min(i, len(E5_PI_RAMP) - 1)]
        sched_markers[s] = MARKERS[i % len(MARKERS)]
        sched_styles[s] = (0, ())

    def draw_a(ax: Axes) -> None:
        """(a) Validation loss vs step, against each schedule's Bayes floor.

        On a linear axis this panel is worse than useless: four of the five
        curves pile onto zero, and the one that does not (``pi=1e-1``, at
        $0.069$) looks like the worst fit when it is in fact the *irreducible*
        loss of that schedule.  A fraction ``pi`` of its examples are fully
        masked, and a fully masked example carries $H(w)=\\log 2$ nats that no
        model can remove, so the floor is ``pi log 2`` -- which the measured
        curves reach to within $5\\%$.  Plotting on a log axis with those floors
        drawn makes the panel say what it should: every run converged, and the
        level differences are schedule structure, not fit quality.
        """
        for sched in sched_order:
            sub = training[training["schedule"] == sched]
            seeds = sub["seed"].unique()
            for seed in seeds:
                run = sub[sub["seed"] == seed].sort_values("step")
                ax.plot(run["step"], run["val_loss"], color=sched_colors[sched],
                        ls=sched_styles[sched],
                        alpha=0.18 if len(seeds) > 1 else 1.0, lw=0.7)
            mean = sub.groupby("step")["val_loss"].mean().reset_index()
            ax.plot(mean["step"], mean["val_loss"], color=sched_colors[sched],
                    ls=sched_styles[sched], lw=1.5, label=sched)
            floor = float(sub["pi"].iloc[0]) * math.log(2.0)
            if floor > 0:
                ax.axhline(floor, color=sched_colors[sched], lw=0.7,
                           ls=(0, (1, 2)), alpha=0.9, zorder=0)
        ax.set_yscale("log")
        ax.set_xlabel("training step")
        ax.set_ylabel("validation loss (nats)")
        corner_note(ax, r"dotted: Bayes floor $\pi_0\log 2$",
                    xy=(0.97, 0.03), ha="right", va="bottom")
        ax.legend(fontsize=6.5, loc="upper right", frameon=True,
                  framealpha=0.88, edgecolor="none", facecolor=SURFACE)

    def draw_b(ax: Axes) -> None:
        r"""(b) Implied mode weight $\hat w$ vs step.

        A seed-mean is drawn only where the seeds agree.  Under the blind
        schedule they do not: each run collapses onto one mode or the other,
        so the five endpoints straddle the range and their mean sits at a
        value no single run visits.  Averaging there would manufacture a
        smooth curve out of a coin flip, so the blind runs are shown
        individually at full opacity instead.
        """
        spread_tol = 0.10
        for sched in sched_order:
            sub = training[training["schedule"] == sched]
            seeds = sub["seed"].unique()
            final = sub[sub["step"] == sub["step"].max()]["w_hat"]
            agree = len(seeds) < 2 or (float(final.max()) - float(final.min())) < spread_tol
            for seed in seeds:
                run = sub[sub["seed"] == seed].sort_values("step")
                ax.plot(run["step"], run["w_hat"], color=sched_colors[sched],
                        ls=sched_styles[sched],
                        alpha=0.25 if agree else 0.85,
                        lw=0.7 if agree else 1.0,
                        label=sched if (not agree and seed == seeds[0]) else None)
            if agree:
                mean = sub.groupby("step")["w_hat"].mean().reset_index()
                ax.plot(mean["step"], mean["w_hat"], color=sched_colors[sched],
                        ls=sched_styles[sched], lw=1.5, label=sched)
        ax.axhline(w_true, color=MUTED, lw=0.8, ls="--", zorder=0)
        # every schedule converges onto w, so this label always sits under a
        # bundle of curves; it needs its own background to stay readable
        ax.annotate(f"$w={w_true:g}$", xy=(0.985, w_true),
                    xycoords=("axes fraction", "data"), xytext=(0, 3),
                    textcoords="offset points", fontsize=6.5, color=SECONDARY,
                    ha="right", va="bottom", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE,
                              ec="none", alpha=0.9))
        ax.set_xlabel("training step")
        ax.set_ylabel(r"implied $\hat{w}$")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=6.5, loc="upper right", frameon=True,
                  framealpha=0.88, edgecolor="none", facecolor=SURFACE)

    def draw_c(ax: Axes) -> None:
        """(c) Final |w_hat - w| vs pi_0."""
        # get the final w_hat for each (schedule, seed)
        last_step = training.groupby(["schedule", "seed"])["step"].max().reset_index()
        finals = training.merge(last_step, on=["schedule", "seed", "step"])

        pi_vals = []
        errors = []
        err_sds = []
        colors = []
        markers_list = []
        labels = []

        for sched in sched_order:
            sub = finals[finals["schedule"] == sched]
            pi = float(sub["pi"].iloc[0])
            w_hats = sub["w_hat"].values
            err = np.mean(np.abs(w_hats - w_true))
            # sample sd over seeds, not population sd: n=5
            sd = float(np.std(np.abs(w_hats - w_true), ddof=1)) if len(w_hats) > 1 else 0.0
            if pi > 0:
                pi_vals.append(pi)
                errors.append(err)
                err_sds.append(sd)
                colors.append(sched_colors[sched])
                markers_list.append(sched_markers[sched])
                labels.append(sched)

        if pi_vals:
            # The lower cap must stay strictly positive: at pi=1e-1 the sd
            # exceeds the mean, and a log axis silently clips a non-positive
            # lower bound down to the bottom spine -- which reads as an
            # enormous uncertainty rather than as a bar that ran off scale.
            lower = [min(s, e * 0.9) for e, s in zip(errors, err_sds)]
            for p, e, lo, s, c, mk, lab in zip(
                pi_vals, errors, lower, err_sds, colors, markers_list, labels
            ):
                ax.errorbar(p, e, yerr=[[lo], [s]], fmt=mk, color=c,
                            markersize=5, capsize=3, lw=1.2, label=lab)

            pis = np.array(sorted(pi_vals))
            if len(pis) >= 2:
                order = np.argsort(pi_vals)
                errs = np.asarray(errors)[order]
                fitted = float(np.polyfit(np.log10(pis), np.log10(errs), 1)[0])
                ref_err = errors[pi_vals.index(max(pi_vals))]
                slope_line = ref_err * (pis / max(pi_vals)) ** (-0.5)
                ax.plot(pis, slope_line, color=INK, ls=(0, (4, 2)), lw=0.9,
                        alpha=0.6, zorder=1,
                        label=rf"$-1/2$ (fitted ${fitted:.2f}$)")

        # The blind schedule sits at pi_0 = 0 and cannot be placed on a log
        # axis, but it is the comparison the panel exists to make -- so it is
        # drawn as a reference level rather than exiled to a corner note.
        blind = finals[finals["schedule"] == "blind"]
        if not blind.empty:
            blind_err = float(np.mean(np.abs(blind["w_hat"].values - w_true)))
            blind_col = sched_colors.get(blind["schedule"].iloc[0], CAT[1])
            ax.axhline(blind_err, color=blind_col, lw=1.2, ls=(0, (5, 2)),
                       zorder=1)
            ax.annotate(rf"blind ($\pi_0=0$): ${blind_err:.2f}$",
                        xy=(0.03, blind_err), xycoords=("axes fraction", "data"),
                        xytext=(0, 3), textcoords="offset points",
                        fontsize=6.4, color=blind_col, ha="left", va="bottom")

        if pi_vals:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(r"low-visibility mass $\pi_0(\mu)$")
            ax.set_ylabel(r"$|\hat{w} - w|$")
            ax.legend(fontsize=6.5, loc="lower left", frameon=True,
                      framealpha=0.88, edgecolor="none", facecolor=SURFACE)

    panels = [
        Panel("a", "training loss", draw_a, A8_PANEL, M_TITLE_SIZE),
        Panel("b", "mode-weight recovery", draw_b, A8_PANEL, M_TITLE_SIZE),
        Panel("c", r"final $|\hat{w} - w|$ vs.\ $\pi_0$", draw_c, A8_PANEL, M_TITLE_SIZE),
    ]
    render("A10", panels, _grid(1, 3), (3 * A8_PANEL[0], A8_PANEL[1]),
           out_dir, png, tight_kw={})


# ==========================================================================
# Driver
# ==========================================================================


def make_all_figures(results_dir: Path | None = None, out_dir: Path | None = None,
                     only: list[str] | None = None, png: bool = False) -> None:
    apply_style()
    results_dir = Path(results_dir or RESULTS_DIR)
    out_dir = Path(out_dir or FIGURES_DIR)
    art = _load(results_dir)

    todo = {
        "M": lambda: (figure_M(art, out_dir, png), table_M(art, out_dir)),
        "M_E3": lambda: figure_M_E3(art, out_dir, png),
        "A1": lambda: figure_A1(art, out_dir, png),
        "A2": lambda: figure_A2(art, out_dir, png),
        "A3": lambda: figure_A3(art, out_dir, png),
        "A4": lambda: figure_A4(art, out_dir, png),
        "A5": lambda: figure_A5(art, out_dir, png),
        "A6": lambda: figure_A6(art, out_dir, png),
        "A7": lambda: figure_A7(art, out_dir, png),
        "A8": lambda: figure_A8(art, out_dir, png),
        "A9": lambda: figure_A9(art, out_dir, png),
        "A10": lambda: figure_A10(art, out_dir, png),
    }
    for name, fn in todo.items():
        if only and name not in only:
            continue
        try:
            fn()
        except Exception as exc:  # keep going: a missing artifact must not
            print(f"  [warn] figure {name} skipped: {type(exc).__name__}: {exc}")
