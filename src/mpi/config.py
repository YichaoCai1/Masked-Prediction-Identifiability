"""Grids, run matrix, predicted constants, and paths.

Everything a run depends on lives here so that a single object
(:class:`RunConfig`) can be hashed and written into every output table
(spec section 6, "Determinism").
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
FIGURES_DIR = REPO_ROOT / "figures"
CACHE_DIR = REPO_ROOT / ".cache"


# --------------------------------------------------------------------------
# Law specifications
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LawSpec:
    """A data law: ``law`` in {"product", "cw"} plus its single parameter."""

    law: str          # "product" (Law P) or "cw" (Law C, Curie-Weiss)
    param: float      # theta for Law P, beta for Law C
    pinned: bool      # True if the law has two-mode structure (fits are meaningful)
    label: str        # human-readable label for figures

    @property
    def param_name(self) -> str:
        return "theta" if self.law == "product" else "beta"

    @property
    def key(self) -> str:
        return f"{self.law}-{self.param:g}"


#: Law P, the primary law: every constant below is a *prediction*, not a fit.
LAWS_P: tuple[LawSpec, ...] = (
    LawSpec("product", 0.6, True, r"Law P, $\theta=0.6$"),
    LawSpec("product", 0.8, True, r"Law P, $\theta=0.8$"),
)

#: Law C, the robustness / null-control law.
LAWS_C: tuple[LawSpec, ...] = (
    LawSpec("cw", 1.5, True, r"Law C, $\beta=1.5$ (metastable)"),
    LawSpec("cw", 0.5, False, r"Law C, $\beta=0.5$ (null control)"),
)

#: Optional Curie-Weiss sweep (spec section 1.3, "optional").
LAWS_C_SWEEP: tuple[LawSpec, ...] = (
    LawSpec("cw", 0.9, False, r"Law C, $\beta=0.9$"),
    LawSpec("cw", 1.1, True, r"Law C, $\beta=1.1$"),
)

ALL_LAWS: tuple[LawSpec, ...] = LAWS_P + LAWS_C


def law_by_key(key: str) -> LawSpec:
    for spec in ALL_LAWS + LAWS_C_SWEEP:
        if spec.key == key:
            return spec
    raise KeyError(f"unknown law key {key!r}")


# --------------------------------------------------------------------------
# Predicted constants (spec section 1.1) -- Law P only
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ProductConstants:
    """Closed-form constants of Prop. `prop:product-instantiation`.

    All are predictions from the paper; nothing here is fitted.
    """

    theta: float
    N: int

    @property
    def L0(self) -> float:
        return math.log((1.0 + self.theta) / (1.0 - self.theta))

    @property
    def alpha(self) -> float:
        return self.theta / (2.0 * (1.0 + self.theta))

    @property
    def eta_N(self) -> float:
        return 2.0 * math.exp(-self.theta**2 * self.N / 8.0)

    @property
    def c0(self) -> float:
        return 0.5

    @property
    def kappa(self) -> float:
        return self.L0 * self.theta / 2.0

    @property
    def c1(self) -> float:
        return self.theta**2 / 16.0

    @property
    def c(self) -> float:
        """Combined certified decay rate ``min{c1, kappa}``."""
        return min(self.c1, self.kappa)

    @property
    def s_pin(self) -> int:
        return math.ceil(16.0 * math.log(2.0) / self.theta**2)

    @property
    def u0(self) -> float:
        return 0.25

    @property
    def L(self) -> float:
        return self.L0 + self.eta_N

    @property
    def s_ov(self) -> int:
        return math.floor(self.alpha * self.N)

    @property
    def rate_star(self) -> float:
        """The *true* decay rate: minus the log Bhattacharyya coefficient
        between the two mode-conditional coordinate marginals."""
        return -0.5 * math.log(1.0 - self.theta**2)

    @property
    def certified_window(self) -> tuple[int, int]:
        return (self.s_pin, self.s_ov)

    @property
    def window_empty(self) -> bool:
        return self.s_pin > self.s_ov

    def as_dict(self) -> dict:
        return {
            "theta": self.theta,
            "N": self.N,
            "L0": self.L0,
            "alpha": self.alpha,
            "eta_N": self.eta_N,
            "c0": self.c0,
            "kappa": self.kappa,
            "c1": self.c1,
            "c": self.c,
            "s_pin": self.s_pin,
            "u0": self.u0,
            "L": self.L,
            "s_ov": self.s_ov,
            "rate_star": self.rate_star,
            "window_empty": self.window_empty,
        }


def curie_weiss_magnetisation(beta: float, tol: float = 1e-14) -> float:
    """Positive solution of ``m = tanh(beta m)``; 0 when ``beta <= 1``."""
    if beta <= 1.0:
        return 0.0
    lo, hi = 1e-12, 1.0 - 1e-15
    # f(m) = tanh(beta m) - m is positive just above 0 and negative at 1.
    for _ in range(400):
        mid = 0.5 * (lo + hi)
        if math.tanh(beta * mid) - mid > 0.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def reference_rate(spec: LawSpec) -> float | None:
    """Predicted (Law P) or reference (Law C) exponential decay rate.

    For Law C this is a *reference* line obtained by substituting the
    spontaneous magnetisation ``m*`` for ``theta`` in the Chernoff formula.
    It is not a claim of the paper and nothing is fitted to it.
    """
    if spec.law == "product":
        return -0.5 * math.log(1.0 - spec.param**2)
    m_star = curie_weiss_magnetisation(spec.param)
    if m_star <= 0.0:
        return None
    return -0.5 * math.log(1.0 - m_star**2)


# --------------------------------------------------------------------------
# Sensitivity band constants (Lem. `lem:sensitivity`, functions of I alone)
# --------------------------------------------------------------------------


def logit(u: float) -> float:
    return math.log(u / (1.0 - u))


def mode_weight_interval(w: float) -> tuple[float, float]:
    """``I = [c0/2, 1 - c0/2]`` with ``c0 = min(w, 1-w)``."""
    c0 = min(w, 1.0 - w)
    return (c0 / 2.0, 1.0 - c0 / 2.0)


def band_constants(w: float) -> dict[str, float]:
    """``(c_I, C_I)`` of Lem. `lem:sensitivity`, plus the exact limit rho(w)."""
    a, b = mode_weight_interval(w)
    delta_I = logit(b) - logit(a)
    # u(1-u) is concave, so its minimum over [a, b] is attained at an endpoint.
    m_I = min(a * (1.0 - a), b * (1.0 - b))
    return {
        "a": a,
        "b": b,
        "delta_I": delta_I,
        "m_I": m_I,
        "c_I": 4.0 * math.exp(-2.0 * delta_I),
        "C_I": math.exp(2.0 * delta_I) / m_I,
        "rho_limit": 1.0 / (w * (1.0 - w)),
    }


# --------------------------------------------------------------------------
# Grids (spec section 1.3)
# --------------------------------------------------------------------------

N_GRID: tuple[int, ...] = (127, 255, 511, 1023)
W_GRID: tuple[float, ...] = (0.5, 0.9)

#: Always-computed small visible sizes (needed for the full-mask boundary and
#: for the low-visibility uncertainty fit).
SMALL_M: tuple[int, ...] = (0, 1, 2, 4)

_DYADIC_M: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)

#: Fixed visible fractions used for the "decay at fixed fraction of N" fit.
RHO_FRACTIONS: tuple[float, ...] = (0.25, 0.5, 0.75)

#: Low-visibility intervention grids.
PI_GRID: tuple[float, ...] = (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
S_BOOST_GRID: tuple[int, ...] = (0, 1, 2, 4)
EPS_GRID: tuple[float, ...] = (1e-8, 1e-6, 1e-4, 1e-2)

#: Branch threshold for the three-branch binary-KL evaluation (spec section 2.3).
BRANCH_T_DEFAULT: float = 30.0
BRANCH_T_SENSITIVITY: tuple[float, ...] = (25.0, 30.0, 35.0)

#: Default binary-KL evaluation scheme.  ``"stable"`` is the threshold-free
#: ``expm1``/``log1p`` form; ``"branched"`` is the three-branch scheme of spec
#: section 2.3.  Measured against mpmath at 60 digits, ``"stable"`` is accurate
#: to ~1e-14 in ``log D`` uniformly, whereas ``"branched"`` degrades to
#: 1e-10...1e-7 because its middle branch cancels catastrophically just inside
#: ``|x| = T``.  See README, "Deviations from the specification".
KL_IMPL_DEFAULT: str = "stable"

#: The (scheme, T) pairs whose spread forms leg (b) of the numerical CI.
KL_IMPL_SENSITIVITY: tuple[tuple[str, float], ...] = (
    ("stable", 30.0),
    ("branched", 25.0),
    ("branched", 30.0),
    ("branched", 35.0),
)

#: Minimum number of points in an OLS decay window.
FIT_MIN_WINDOW: int = 5


def visible_sizes(N: int) -> np.ndarray:
    """``M_N`` of spec section 1.3, unioned with ``{0, 1, 2, 4}``, sorted."""
    sizes = {m for m in _DYADIC_M if 1 <= m <= N - 1}
    sizes |= {N // 4, N // 2, (3 * N) // 4}
    sizes |= set(SMALL_M)
    return np.array(sorted(s for s in sizes if 0 <= s <= N - 1), dtype=int)


def lambda_grid_I(w: float, n: int = 201) -> np.ndarray:
    """``Lambda_I``: ``n`` equispaced points on ``I``, with ``lambda = w`` exact.

    The point nearest ``w`` is snapped onto ``w`` so that the grid contains the
    true mode weight exactly (needed for test T8 and for the curvature fit).
    """
    a, b = mode_weight_interval(w)
    grid = np.linspace(a, b, n)
    grid[int(np.argmin(np.abs(grid - w)))] = w
    return grid


def lambda_grid_wide(n: int = 197) -> np.ndarray:
    """``Lambda_wide``: visualisation-only grid on ``[0.01, 0.99]``."""
    return np.linspace(0.01, 0.99, n)


# --------------------------------------------------------------------------
# Run configuration + provenance
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    """Everything that determines the numbers in the output tables."""

    laws: tuple[str, ...] = tuple(spec.key for spec in ALL_LAWS)
    N_grid: tuple[int, ...] = N_GRID
    w_grid: tuple[float, ...] = W_GRID
    branch_T: float = BRANCH_T_DEFAULT
    kl_impl: str = KL_IMPL_DEFAULT
    n_lambda_I: int = 201
    n_lambda_wide: int = 197
    pi_grid: tuple[float, ...] = PI_GRID
    s_boost_grid: tuple[int, ...] = S_BOOST_GRID
    eps_grid: tuple[float, ...] = EPS_GRID
    fit_min_window: int = FIT_MIN_WINDOW
    bisect_tol: float = 1e-10

    def law_specs(self) -> list[LawSpec]:
        return [law_by_key(k) for k in self.laws]

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)

    @property
    def config_hash(self) -> str:
        blob = json.dumps(self.as_dict(), sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def git_hash(default: str = "nogit") -> str:
    try:
        out = subprocess.run(
            [
                "git",
                "-c",
                f"safe.directory={REPO_ROOT.as_posix()}",
                "rev-parse",
                "--short",
                "HEAD",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:  # pragma: no cover - git may be absent
        pass
    return default


def provenance(cfg: RunConfig) -> dict:
    """Provenance stamp written into every output table."""
    return {
        "config_hash": cfg.config_hash,
        "git_hash": git_hash(),
        "branch_T": cfg.branch_T,
        "kl_impl": cfg.kl_impl,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
