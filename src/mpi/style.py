"""Shared plotting style.

Colour choices use the paper's violet/red visual identity for discrete series:

* every ordered or categorical sequence starts from the first palette entry,
  so panels with the same number of series share the same accents;
* corpus and model identities use those same positions consistently;
* genuinely continuous color bars use a custom diverging map whose endpoints
  match the paper's DarkKlein and LinkBurgundy hyperlink colours.

Every series is also distinguished by marker and/or dash pattern, so identity
never rests on colour alone.
"""

from __future__ import annotations

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

__all__ = [
    "apply_style", "CALM_PALETTE", "CONTINUOUS_CMAP", "N_COLORS", "CAT", "INK", "MUTED", "GRID", "SURFACE", "NULL_GREY",
    "CORPUS_COLORS", "CORPUS_LABELS", "CORPUS_MARKERS", "n_color", "m_cmap", "law_style",
    "annotate_panel", "corner_note", "save",
]

# -- fixed colormaps --------------------------------------------------------
# User-approved paper palette, kept explicit so the mapping cannot drift with
# plotting-library defaults.  The first two entries exactly match mpiViolet
# and mpiRed in the main figure; the remaining hues are calm supporting
# accents chosen to stay distinct from them and from one another.
CALM_PALETTE = (
    "#4A3AA7", "#E34948", "#2A7F82", "#C58A32", "#3568A8",
    "#4F8A61", "#A95878", "#80634E", "#66717E", "#898781",
)
CONTINUOUS_CMAP = LinearSegmentedColormap.from_list(
    "mpi_diverging",
    ("#122A82", "#4A3AA7", "#F3F1EC", "#E34948", "#8A1538"),
    N=256,
)

# -- ink & chrome -----------------------------------------------------------

INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
# Pure white: figures sit on the page, not on a tinted card.  Deliberate
# region shading (axvspan / axhspan / fill_between) is drawn *over* this,
# so the regions that must stay distinguishable still are.
SURFACE = "#ffffff"
#: null control as one series among several, where it must recede
NULL_GREY = CALM_PALETTE[9]
# -- categorical slots in canonical order ----------------------------------
CAT = list(CALM_PALETTE)

# -- discrete dimension levels use the same fixed order ---------------------
N_RAMP = list(CALM_PALETTE)

# -- corpus identity within the same shared family ---------------------------
# Colour identifies the *corpus*, marker the *estimator arm*, so the reader
# sees at a glance that the corpus study is a different kind of claim -- measurement on real
# data, not exact synthetic computation.
CORPUS_COLORS = {
    "code_prose": CALM_PALETTE[0],
    "bilingual": CALM_PALETTE[1],
    "markov": CALM_PALETTE[2],
}
CORPUS_LABELS = {
    "code_prose": "code vs prose",
    "bilingual": "German vs English",
    "markov": "Markov fixture",
}
CORPUS_MARKERS = {"small": "D", "large": "v"}

MARKERS = ["o", "s", "^", "D", "v", "P"]


def apply_style() -> None:
    """Paper-figure rcParams: thin marks, recessive chrome, no chartjunk."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "axes.labelcolor": INK,
            "axes.edgecolor": "#b0afa8",
            "axes.linewidth": 0.5,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.grid": False,
            "grid.color": GRID,
            "grid.linewidth": 0.35,
            "grid.alpha": 1.0,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": SECONDARY,
            "ytick.labelcolor": SECONDARY,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": False,
            "xtick.minor.top": False,
            "ytick.right": False,
            "ytick.minor.right": False,
            "legend.fontsize": 7.5,
            "legend.frameon": False,
            "legend.handlelength": 1.8,
            "lines.linewidth": 1.4,
            "lines.markersize": 3.6,
            "figure.dpi": 130,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def n_color(N: int, all_N) -> str:
    """Colour for a given ``N``: its position in the fixed palette order."""
    all_N = sorted(set(int(v) for v in all_N))
    idx = all_N.index(int(N))
    return N_RAMP[idx % len(N_RAMP)]


N_COLORS = N_RAMP


def m_cmap():
    """The shared categorical map for the discrete visibility levels."""
    return ListedColormap(CALM_PALETTE, name="mpi_categorical")


def law_style(law: str, param: float, pinned: bool) -> dict:
    """Line style for the four displayed laws in the paper palette."""
    if not pinned:
        return {"color": NULL_GREY, "linestyle": (0, (1, 1.6)), "linewidth": 1.1}
    if law == "product":
        # theta=0.8 is the paper's primary conditioned-product setting.
        return {"color": CAT[0] if param >= 0.7 else CAT[4], "linestyle": "-"}
    return {"color": CAT[2], "linestyle": (0, (5, 1.5))}


def annotate_panel(ax, letter: str, title: str, fontsize: float | None = None,
                   fontweight: str | None = None,
                   show_letter: bool = True) -> None:
    """Centred panel title, so it sits over its own plot area rather than the figure."""
    text = f"({letter}) {title}" if show_letter else title
    ax.set_title(text, loc="center", pad=6, color=INK,
                 fontsize=fontsize, fontweight=fontweight)


def corner_note(ax, text: str, xy=(0.98, 0.02), ha="right", va="bottom",
                fontsize: float = 6.4) -> None:
    """Muted in-axes note for context a standalone panel would otherwise lose
    (the run parameters that used to live in a figure-level suptitle)."""
    ax.annotate(text, xy=xy, xycoords="axes fraction", ha=ha, va=va,
                fontsize=fontsize, color=MUTED)


def save(fig, path, close: bool = True, png: bool = False,
         tight: bool = True) -> None:
    """Write ``path.pdf`` (and ``path.png`` only when ``png`` is set).

    PDF is the deliverable; PNG exists solely as a preview channel behind
    ``make_figures.py --png``.

    ``tight=False`` writes the figure at exactly its declared ``figsize``
    instead of cropping to the drawn content.  Standalone panels use it so that
    every panel of a figure comes out at an identical size and can be dropped
    into a subfigure row without one of them sitting a millimetre proud of the
    others; ``fig.tight_layout()`` has already arranged the content inside.
    """
    from pathlib import Path

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # bbox_inches=None would fall back to rcParams["savefig.bbox"], which is
    # "tight" here; passing the figure's own bbox is what actually pins the
    # output to the declared figsize.
    bbox = "tight" if tight else fig.bbox_inches
    fig.savefig(path.with_suffix(".pdf"), bbox_inches=bbox)
    if png:
        fig.savefig(path.with_suffix(".png"), bbox_inches=bbox)
    if close:
        plt.close(fig)
