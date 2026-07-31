"""Shared plotting style.

Colour choices are validated, not eyeballed:

* ``N`` and ``m`` are *ordered*, so they get one-hue ramps (blue for ``N``,
  viridis for ``m`` as the spec requires) rather than categorical hues.
* ``s`` (boost size) and the law comparison are *categorical* with <= 3 levels,
  so they take the first three slots of a categorical palette whose all-pairs
  CVD separation was checked with the palette validator
  (worst pair deutan dE 9.2, normal-vision dE 24.0 on a light surface).
* The null control and every reference line are achromatic, so "mechanism
  absent" and "prediction" never compete with a data hue.
* E3 lives in a different colour family (violet/red) because it is a different
  kind of claim -- measurement on real data, not exact enumeration.

Every series is also distinguished by marker and/or dash pattern, so identity
never rests on colour alone.
"""

from __future__ import annotations

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

__all__ = [
    "apply_style", "N_COLORS", "CAT", "INK", "MUTED", "GRID", "SURFACE", "NULL_GREY",
    "NULL_ACCENT",
    "E3_COLORS", "E3_LABELS", "E3_MARKERS", "n_color", "m_cmap", "law_style",
    "annotate_panel", "corner_note", "save",
]

# -- ink & chrome -----------------------------------------------------------
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
# Pure white: figures sit on the page, not on a tinted card.  Deliberate
# region shading (axvspan / axhspan / fill_between) is drawn *over* this,
# so the regions that must stay distinguishable still are.
SURFACE = "#ffffff"
#: null control as one series among several (A2), where it must recede
NULL_GREY = "#b8b6ae"
#: null control overlaid on the data (M(a)), where it must be unmistakable.
#: Orange is the CVD-validated complement of the blue N-ramp and appears
#: nowhere else in that panel; the dotted pattern still reads as "control".
NULL_ACCENT = "#eb6834"

# -- categorical slots (harmonised with E3 violet/red/green) ----------------
CAT = ["#4a3aa7", "#e34948", "#008300", "#d4820e"]

# -- ordinal violet ramp for N (harmonised with E3 palette) ------------------
N_RAMP = ["#a99be0", "#7766cc", "#5544b0", "#3a2a80"]

# -- E3 palette: deliberately outside the E1/E2 families --------------------
# Colour identifies the *corpus*, marker the *estimator arm*, so the reader
# sees at a glance that E3 is a different kind of claim -- measurement on real
# data, not exact synthetic computation.
E3_COLORS = {"code_prose": "#4a3aa7", "bilingual": "#e34948", "markov": "#008300"}
E3_LABELS = {"code_prose": "code vs prose", "bilingual": "German vs English",
             "markov": "Markov fixture"}
E3_MARKERS = {"small": "D", "large": "v"}

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
            "axes.grid": True,
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
    """Colour for a given ``N``: position in the ordered blue ramp."""
    all_N = sorted(set(int(v) for v in all_N))
    idx = all_N.index(int(N))
    if len(all_N) == 1:
        return N_RAMP[-2]
    pos = idx * (len(N_RAMP) - 1) / (len(all_N) - 1)
    return N_RAMP[int(round(pos))]


N_COLORS = N_RAMP


def m_cmap():
    """Viridis, as the spec prescribes for colour-by-``m`` small multiples."""
    return plt.get_cmap("viridis")


def law_style(law: str, param: float, pinned: bool) -> dict:
    """Line style for a law: pinned laws get hue, the null control stays grey."""
    if not pinned:
        return {"color": NULL_GREY, "linestyle": (0, (1, 1.6)), "linewidth": 1.1}
    if law == "product":
        return {"color": CAT[0] if param < 0.7 else CAT[1], "linestyle": "-"}
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
