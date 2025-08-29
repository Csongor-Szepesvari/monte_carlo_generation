"""
Benchmark configuration: scenarios, grids, and helpers to enumerate runs.

This module centralizes the experimental grid for comparing methods that
estimate the sum of top-k order statistics under various distribution regimes.

Scopes covered:
- Scenario grids over total samples (n), accuracy targets (epsilon), and seeds
- Policy for deriving k from n
- Distribution regimes (balanced and imbalanced in mean/variance)
- Partitioning parameters to match monte_carlo_generation defaults

Intended usage (example):
    from benchmark_config import enumerate_runs, DEFAULT_GRID
    for cfg in enumerate_runs(DEFAULT_GRID):
        # cfg is a dict with fields ready for a benchmarking harness
        ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, Iterator, List, Optional


# ----------------------------- k policy helpers -----------------------------

def compute_k(total_n: int, policy: str) -> int:
    """Compute k given a policy string.

    Supported policies:
    - "fraction:<p>": k = max(1, floor(p * n)) e.g., "fraction:0.25"
    - "fixed:<k>": k = given integer, clipped to [1, n]
    """
    if policy.startswith("fraction:"):
        fraction = float(policy.split(":", 1)[1])
        k = int(total_n * fraction)
        return max(1, min(k, total_n))
    if policy.startswith("fixed:"):
        k = int(policy.split(":", 1)[1])
        return max(1, min(k, total_n))
    raise ValueError(f"Unsupported k policy: {policy}")


# --------------------------- distribution regimes ---------------------------

@dataclass(frozen=True)
class DistributionRegime:
    name: str
    mu: List[float]
    sigma: List[float]


DEFAULT_REGIMES: List[DistributionRegime] = [
    DistributionRegime(
        name="balanced_mean_balanced_var",
        mu=[0.0, 0.0, 0.0, 0.0],
        sigma=[1.0, 1.0, 1.0, 1.0],
    ),
    DistributionRegime(
        name="mean_skew_one_dominant",
        mu=[1.0, 0.0, 0.0, 0.0],
        sigma=[1.0, 1.0, 1.0, 1.0],
    ),
    DistributionRegime(
        name="variance_skew_one_high",
        mu=[0.0, 0.0, 0.0, 0.0],
        sigma=[2.0, 1.0, 1.0, 1.0],
    ),
    DistributionRegime(
        name="two_modes_competitive",
        mu=[0.8, 0.7, 0.0, 0.0],
        sigma=[1.0, 1.0, 1.0, 1.0],
    ),
    DistributionRegime(
        name="mixed_var_competitive",
        mu=[0.5, 0.0, 0.0, 0.0],
        sigma=[1.5, 1.0, 0.8, 0.6],
    ),
]


# ------------------------------ grid definition -----------------------------

@dataclass(frozen=True)
class Grid:
    n_values: List[int] = field(default_factory=lambda: [20, 40, 60, 100, 200, 400])
    epsilon_values: List[float] = field(default_factory=lambda: [0.05, 0.02, 0.01])
    k_policy: str = "fraction:0.25"  # aligns with monte_carlo_generation default
    seeds: List[int] = field(default_factory=lambda: list(range(5)))
    regimes: List[DistributionRegime] = field(default_factory=lambda: DEFAULT_REGIMES)

    # Partitioning parameters (used by monte_carlo_generation)
    num_partitions: int = 10
    bias_fraction: float = 0.5
    bias_factor: float = 2.0


DEFAULT_GRID = Grid()


# ---------------------------- enumeration utilities ----------------------------

def enumerate_runs(grid: Grid = DEFAULT_GRID) -> Iterator[Dict]:
    """Yield concrete run configurations from the cartesian product of the grid.

    Each yielded dict contains:
    - regime_name, mu, sigma
    - total_n, k, epsilon, seed
    - num_partitions, bias_fraction, bias_factor

    These dicts are intended to be consumed by a benchmarking harness that can
    call various estimators (adaptive MC, UCB/BR, NN) under identical settings.
    """
    for regime in grid.regimes:
        for total_n in grid.n_values:
            k = compute_k(total_n, grid.k_policy)
            for epsilon in grid.epsilon_values:
                for seed in grid.seeds:
                    yield {
                        "regime_name": regime.name,
                        "mu": list(regime.mu),
                        "sigma": list(regime.sigma),
                        "total_n": int(total_n),
                        "k": int(k),
                        "epsilon": float(epsilon),
                        "seed": int(seed),
                        "num_partitions": int(grid.num_partitions),
                        "bias_fraction": float(grid.bias_fraction),
                        "bias_factor": float(grid.bias_factor),
                    }


def summarize_grid(grid: Grid = DEFAULT_GRID) -> Dict[str, int]:
    """Return simple counts for quick sanity checks."""
    return {
        "num_regimes": len(grid.regimes),
        "num_n": len(grid.n_values),
        "num_epsilons": len(grid.epsilon_values),
        "num_seeds": len(grid.seeds),
        "total_runs": len(grid.regimes)
        * len(grid.n_values)
        * len(grid.epsilon_values)
        * len(grid.seeds),
    }


__all__ = [
    "DistributionRegime",
    "Grid",
    "DEFAULT_REGIMES",
    "DEFAULT_GRID",
    "compute_k",
    "enumerate_runs",
    "summarize_grid",
]


