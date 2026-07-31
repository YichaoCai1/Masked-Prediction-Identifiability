"""E5 -- optimisation sanity check: does the population prediction survive SGD?

E1--E4 are population statements.  E5 trains a small model on data drawn from
Law P under different mask schedules and reads out the implied mode weight,
showing that the mode-blindness / recovery dichotomy predicted by the theory
manifests in actual gradient-based optimisation.

The model is minimal by design.  Law P is exchangeable, so ``p(X_K | X_V)``
depends on ``X_V`` only through the visible magnetisation ``r``.  By the
information decomposition (Prop. info-decomp), the mode-weight signal is
carried entirely by the binary cross-entropy of the mode posterior
``beta_V(r) = p(Z = +1 | X_V)``.  The within-mode conditional
``p(X_K | Z, X_V)`` is a product law whose parameters are known and
schedule-independent, so it contributes no gradient on the mode weight.

We therefore train a small MLP that predicts ``logit beta(r, m)`` from the
sufficient statistics ``(r, m)`` and evaluate the implied mode weight
``w_hat = sigma(logit beta(0, 0))`` at the trivial fibre.

Produces ``training``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor

from .config import LawSpec, RunConfig, RESULTS_DIR
from .tableio import write_table

__all__ = [
    "Schedule",
    "LawPSampler",
    "ModePredictor",
    "pick_device",
    "train_run",
    "run_e5",
]


def pick_device(prefer: str | None = None) -> torch.device:
    """Device for E5; ``prefer`` (``--e5-device``) overrides the default.

    The default is CPU *even when a GPU is present*, which is measured rather
    than assumed: the model is a 2-128-128-1 MLP on batches of 512, small
    enough that per-step kernel-launch overhead dominates the arithmetic.  On
    a 3000-step benchmark CUDA came out at 18.5s against 17.5s for CPU, so
    moving to the GPU costs a little and buys nothing.  Pass ``cuda``
    explicitly if a larger model or batch makes that trade different.
    """
    return torch.device(prefer) if prefer else torch.device("cpu")


# --------------------------------------------------------------------------
# Schedule specification
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Schedule:
    """A two-point mask schedule: ``(1-pi) delta_{m_base} + pi delta_s``.

    ``pi = 0`` is the mode-blind baseline (only the base visible size is used).
    ``pi > 0`` mixes in low-visibility examples at visible size ``s``.
    """

    name: str
    m_base: int          # base visible size (e.g. N//2)
    s: int               # boost visible size (0 = full mask)
    pi: float            # probability of drawing the boost size

    @property
    def label(self) -> str:
        if self.pi <= 0.0:
            return f"blind ($s={self.s}$, $\\pi_0=0$)"
        return f"$s={self.s}$, $\\pi_0={self.pi:g}$"


# --------------------------------------------------------------------------
# Data sampler
# --------------------------------------------------------------------------


class LawPSampler:
    """On-the-fly sampler from Law P with a given mask schedule.

    Each call to ``sample(batch_size)`` returns a batch of
    ``(r, m, z)`` tuples where:
    - ``z in {0, 1}`` is the mode (1 = positive, 0 = negative)
    - ``m`` is the visible set size drawn from the schedule
    - ``r`` is the visible magnetisation ``sum(X_V)``

    Because the model only needs ``(r, m)`` as input and ``z`` as target,
    we never materialise the full sequence ``X`` -- we draw ``r`` directly
    from its conditional distribution given ``(z, m)``, which for Law P is
    ``r = 2a - m`` where ``a ~ Binomial(m, (1 + tau*theta)/2)`` and
    ``tau = 2z - 1``.
    """

    def __init__(self, theta: float, w: float, N: int, schedule: Schedule,
                 rng: np.random.Generator | None = None):
        self.theta = theta
        self.w = w
        self.N = N
        self.schedule = schedule
        self.rng = rng or np.random.default_rng()

        # precompute Bernoulli parameters for each mode
        self.p_plus = (1.0 + theta) / 2.0   # P(X_i = +1 | Z = +1)
        self.p_minus = (1.0 - theta) / 2.0  # P(X_i = +1 | Z = -1)

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor]:
        """Return ``(features, targets)`` where features = (r, m) and target = z."""
        rng = self.rng

        # 1. Draw mode z ~ Bernoulli(w)
        z = rng.binomial(1, self.w, size=batch_size)  # 1 = positive mode

        # 2. Draw visible size m from the schedule
        use_boost = rng.random(batch_size) < self.schedule.pi
        m = np.where(use_boost, self.schedule.s, self.schedule.m_base)

        # 3. Draw visible magnetisation r from the conditional
        #    a ~ Binomial(m, p_tau) where tau = 2z - 1
        p_coin = np.where(z == 1, self.p_plus, self.p_minus)
        a = rng.binomial(m, p_coin)
        r = 2 * a - m

        # 4. Determine the true mode from the FULL sequence's sign.
        #    We need to know whether the model can infer Z from X_V alone.
        #    The target IS z (the mode label), since the model predicts
        #    beta_V(r) = P(Z=+1 | X_V).  At population, the Bayes-optimal
        #    beta_V is analytically known; we train the model to learn it.

        # features: (r / N, m / N) -- normalised for numerical stability
        feat = torch.tensor(
            np.column_stack([r / self.N, m / self.N]),
            dtype=torch.float32,
        )
        target = torch.tensor(z, dtype=torch.float32)
        m_tensor = torch.tensor(m, dtype=torch.float32)
        return feat, target, m_tensor


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


class ModePredictor(nn.Module):
    """Small MLP: (r/N, m/N) → logit β̂(r, m).

    Two hidden layers of 128 units with ReLU activation, outputting a single
    logit.  The architecture is deliberately over-parameterised relative to
    the true posterior (which is a monotone function of ``r`` for each ``m``),
    so any failure to recover ``w`` is due to the schedule, not capacity.
    """

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x).squeeze(-1)

    def w_hat(self) -> float:
        """Read out the implied mode weight at the trivial fibre (r=0, m=0)."""
        with torch.no_grad():
            # the probe must live wherever the weights do
            device = next(self.parameters()).device
            logit = self.forward(torch.zeros(1, 2, device=device))
            return float(torch.sigmoid(logit).item())


# --------------------------------------------------------------------------
# Single training run
# --------------------------------------------------------------------------


@dataclass
class TrainLog:
    """Log of a single training run."""

    schedule: str
    seed: int
    steps: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    w_hats: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "schedule": self.schedule,
            "seed": self.seed,
            "step": self.steps,
            "loss": self.losses,
            "w_hat": self.w_hats,
            "val_loss": self.val_losses,
        })


def train_run(
    sampler: LawPSampler,
    schedule_name: str,
    seed: int,
    n_steps: int = 50_000,
    batch_size: int = 512,
    lr: float = 1e-3,
    log_every: int = 500,
    val_size: int = 10_000,
    verbose: bool = True,
    device: torch.device | str | None = None,
) -> TrainLog:
    """Train one model and return the training log."""
    device = torch.device(device) if device is not None else pick_device()
    torch.manual_seed(seed)
    model = ModePredictor().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    log = TrainLog(schedule=schedule_name, seed=seed)

    # Fixed validation batch (same seed offset so it's consistent).  Moved once,
    # outside the loop: it is 10k rows and never changes.
    val_rng = np.random.default_rng(seed + 99999)
    val_sampler = LawPSampler(
        sampler.theta, sampler.w, sampler.N, sampler.schedule, val_rng
    )
    val_feat, val_target, _ = val_sampler.sample(val_size)
    val_feat, val_target = val_feat.to(device), val_target.to(device)

    for step in range(1, n_steps + 1):
        feat, target, _ = sampler.sample(batch_size)
        feat, target = feat.to(device), target.to(device)
        logit = model(feat)
        loss = criterion(logit, target)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if step % log_every == 0 or step == 1:
            with torch.no_grad():
                val_logit = model(val_feat)
                val_loss = criterion(val_logit, val_target).item()
                w_hat = model.w_hat()

            log.steps.append(step)
            log.losses.append(loss.item())
            log.w_hats.append(w_hat)
            log.val_losses.append(val_loss)

            if verbose and step % (log_every * 10) == 0:
                print(f"    step {step:>6d}  loss={loss.item():.4f}  "
                      f"val={val_loss:.4f}  w_hat={w_hat:.4f}")

    return log


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

# Default schedule grid
DEFAULT_SCHEDULES: tuple[Schedule, ...] = (
    Schedule("blind",       m_base=511, s=4, pi=0.0),
    Schedule("pi=1e-4",     m_base=511, s=0, pi=1e-4),
    Schedule("pi=1e-3",     m_base=511, s=0, pi=1e-3),
    Schedule("pi=1e-2",     m_base=511, s=0, pi=1e-2),
    Schedule("pi=1e-1",     m_base=511, s=0, pi=1e-1),
)


def run_e5(
    cfg: RunConfig | None = None,
    theta: float = 0.8,
    w: float = 0.5,
    N: int = 1023,
    schedules: Sequence[Schedule] | None = None,
    n_seeds: int = 5,
    n_steps: int = 50_000,
    batch_size: int = 512,
    lr: float = 1e-3,
    log_every: int = 500,
    results_dir: Path | None = None,
    verbose: bool = True,
    device: torch.device | str | None = None,
) -> dict[str, pd.DataFrame]:
    """Run E5 over all schedules and seeds; write ``training``.

    ``cfg`` is carried only for the provenance stamp: E5's grid is fixed to
    Law P at ``N=1023`` because ``DEFAULT_SCHEDULES`` hardcodes ``m_base=511``,
    so it deliberately does not follow ``cfg``'s ``N``/``w`` sweeps.  The E5
    settings that do vary are written into the table as their own columns.
    """
    if schedules is None:
        schedules = DEFAULT_SCHEDULES
    cfg = cfg or RunConfig()
    results_dir = results_dir or RESULTS_DIR
    device = torch.device(device) if device is not None else pick_device()
    if verbose:
        print(f"  E5 device: {device}")

    t0 = time.time()
    all_logs: list[pd.DataFrame] = []

    for sched in schedules:
        if verbose:
            print(f"  E5 schedule {sched.name!r}  "
                  f"(m_base={sched.m_base}, s={sched.s}, pi={sched.pi})")
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            sampler = LawPSampler(theta, w, N, sched, rng)
            log = train_run(
                sampler, sched.name, seed,
                n_steps=n_steps, batch_size=batch_size,
                lr=lr, log_every=log_every, verbose=verbose,
                device=device,
            )
            df = log.to_dataframe()
            df["theta"] = theta
            df["w"] = w
            df["N"] = N
            df["m_base"] = sched.m_base
            df["s"] = sched.s
            df["pi"] = sched.pi
            # E5's own settings: cfg's provenance stamp describes the exact
            # experiments, not this one, so the table has to say what it ran
            df["n_steps"] = n_steps
            df["batch_size"] = batch_size
            df["lr"] = lr
            df["device"] = str(device)
            all_logs.append(df)

    training = pd.concat(all_logs, ignore_index=True)

    # write_table, not to_parquet: every other artifact carries config_hash /
    # git_hash / library versions, and a figure has to be traceable to its run
    write_table(training, "training", cfg, results_dir)
    if verbose:
        print(f"  E5 done in {time.time()-t0:.1f}s  "
              f"({len(schedules)} schedules × {n_seeds} seeds)")

    return {"training": training}
