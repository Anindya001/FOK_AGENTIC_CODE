"""Sobol sensitivity analysis for the fractional-kinetics model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Sequence

import numpy as np

from fractional_model import FKParams, fractional_capacitance, normalized_deficit, time_to_threshold

PARAM_ORDER = ("C0", "k", "alpha", "f_inf")


class Prior:
    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        raise NotImplementedError


@dataclass
class LogUniformPrior(Prior):
    low: float
    high: float

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return np.exp(rng.uniform(np.log(self.low), np.log(self.high), size))


@dataclass
class BetaPrior(Prior):
    a: float
    b: float

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.beta(self.a, self.b, size)


@dataclass
class LogNormalPrior(Prior):
    mean: float
    sigma: float

    def sample(self, rng: np.random.Generator, size: int) -> np.ndarray:
        return rng.lognormal(self.mean, self.sigma, size)


def _build_matrix(priors: Dict[str, Prior], n: int, rng: np.random.Generator) -> np.ndarray:
    mat = np.empty((n, len(PARAM_ORDER)), dtype=float)
    for j, key in enumerate(PARAM_ORDER):
        mat[:, j] = priors[key].sample(rng, n)
    return mat


def _evaluate_qoi(
    params_matrix: np.ndarray,
    qoi: Callable[[FKParams], float],
) -> np.ndarray:
    outputs = np.empty(params_matrix.shape[0], dtype=float)
    for idx, row in enumerate(params_matrix):
        theta = FKParams(C0=row[0], k=row[1], alpha=row[2], f_inf=row[3])
        try:
            outputs[idx] = qoi(theta)
        except Exception:
            outputs[idx] = np.nan
    return outputs


def _sobol_indices(
    g_a: np.ndarray,
    g_b: np.ndarray,
    g_c: Sequence[np.ndarray],
    *,
    n_bootstrap: int = 200,
    random_state: int | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    """
    Compute Sobol first-order (S) and total-order (ST) indices with bootstrap CIs.

    Rationale for pooled-variance normalization:
    -----------------------------------------
    Variance is computed ONCE from the full g_a vector and reused for all bootstrap
    iterations. This prevents pathological cases where bootstrap resamples collapse
    to low/zero variance, which would yield unstable or undefined index estimates.

    Standard approach: var_b = np.var(g_a_b, ddof=1) per bootstrap
    Problem: If g_a_b has near-zero spread, S_i and ST_i explode or become NaN

    Pooled approach: variance = np.var(g_a, ddof=1) computed once
    Benefit: Stability—indices remain well-defined even if resamples are degenerate

    NaN filtering strategy:
    ----------------------
    1. Non-finite QoI evaluations (NaN/inf from model failures) are removed upfront
    2. During bootstrap: if var_b=0, that iteration gets NaN (lines 110-112)
    3. CI calculation uses np.nanpercentile, which ignores NaN rows (line 119-120)
    4. Result: CIs computed from valid bootstrap samples only

    This is superior to older approaches that either:
    - Used per-sample variance (unstable)
    - Failed entirely on any NaN (too brittle)

    Parameters
    ----------
    g_a, g_b : np.ndarray
        QoI evaluations for Saltelli sample matrices A and B (n_samples each).
    g_c : Sequence[np.ndarray]
        List of QoI evaluations for C matrices (one per parameter, n_samples each).
    n_bootstrap : int, default=200
        Number of bootstrap iterations for confidence interval estimation.
    random_state : int | None
        Random seed for reproducible bootstrap resampling.

    Returns
    -------
    dict
        Keys: 'main' (first-order S), 'total' (total ST), 'main_ci', 'total_ci'
        Each contains arrays of shape (n_params,) or (2, n_params) for CIs.

    Raises
    ------
    RuntimeError
        If no finite samples remain after filtering, or if variance is exactly zero.
    """
    # Step 1: Filter non-finite evaluations across all matrices
    # Rationale: NaN/inf from model failures would corrupt variance/covariance
    mask = np.isfinite(g_a) & np.isfinite(g_b)
    for arr in g_c:
        mask &= np.isfinite(arr)
    if not np.any(mask):
        raise RuntimeError("No finite samples available for Sobol analysis.")

    g_a = g_a[mask]
    g_b = g_b[mask]
    g_c = [arr[mask] for arr in g_c]
    n = g_a.size

    # Step 2: Compute pooled variance ONCE from full g_a
    # This variance is used for ALL point estimates and bootstrap iterations
    variance = np.var(g_a, ddof=1)
    if variance == 0:
        raise RuntimeError("Zero variance encountered; cannot compute Sobol indices.")

    # Step 3: Compute point estimates using pooled variance
    S = []
    ST = []
    for arr in g_c:
        main = np.mean(g_b * (arr - g_a)) / variance
        total = 0.5 * np.mean((g_a - arr) ** 2) / variance
        S.append(main)
        ST.append(total)

    # Step 4: Bootstrap confidence intervals
    # Note: Each bootstrap iteration uses its OWN variance (var_b) for that resample
    # This is intentional—we want CIs to reflect sampling variability in the variance
    # The pooled variance (above) is only for the point estimates
    rng = np.random.default_rng(random_state)
    S_boot = np.zeros((n_bootstrap, len(g_c)), dtype=float)
    ST_boot = np.zeros_like(S_boot)

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)  # Bootstrap resample with replacement
        g_a_b = g_a[idx]
        g_b_b = g_b[idx]
        g_c_b = [arr[idx] for arr in g_c]
        var_b = np.var(g_a_b, ddof=1)

        # Degenerate bootstrap filter: Skip iterations with zero variance
        # These get stored as NaN and ignored in percentile calculation (see below)
        if var_b == 0:
            S_boot[b] = np.nan
            ST_boot[b] = np.nan
            continue

        # Compute indices for this bootstrap sample
        for j, arr_b in enumerate(g_c_b):
            S_boot[b, j] = np.mean(g_b_b * (arr_b - g_a_b)) / var_b
            ST_boot[b, j] = 0.5 * np.mean((g_a_b - arr_b) ** 2) / var_b

    # Step 5: Compute 95% confidence intervals
    # np.nanpercentile automatically excludes NaN rows from degenerate iterations
    lower = 2.5
    upper = 97.5
    S_ci = np.nanpercentile(S_boot, [lower, upper], axis=0)
    ST_ci = np.nanpercentile(ST_boot, [lower, upper], axis=0)

    return {
        "main": np.array(S),
        "total": np.array(ST),
        "main_ci": S_ci,
        "total_ci": ST_ci,
    }


def sobol_analysis(
    priors: Dict[str, Prior],
    qoi: Callable[[FKParams], float],
    *,
    n_samples: int = 2**13,
    n_bootstrap: int = 200,
    random_state: int | None = None,
) -> dict[str, np.ndarray]:
    """Compute first-order and total Sobol indices for a QoI."""
    rng = np.random.default_rng(random_state)
    max_attempts = 4
    samples = max(32, int(n_samples))
    min_required = max(16, samples // 20)
    attempt = 0

    while True:
        A = _build_matrix(priors, samples, rng)
        B = _build_matrix(priors, samples, rng)

        g_a = _evaluate_qoi(A, qoi)
        g_b = _evaluate_qoi(B, qoi)

        g_c = []
        for j in range(len(PARAM_ORDER)):
            C = B.copy()
            C[:, j] = A[:, j]
            g_c.append(_evaluate_qoi(C, qoi))

        mask = np.isfinite(g_a) & np.isfinite(g_b)
        for arr in g_c:
            mask &= np.isfinite(arr)
        finite = int(np.count_nonzero(mask))
        if finite >= min_required:
            break

        attempt += 1
        if attempt >= max_attempts:
            raise RuntimeError(
                "No finite samples available for Sobol analysis after multiple resampling attempts. "
                "Tighten priors or adjust horizons/thresholds to keep the QoI well-defined."
            )
        samples = min(samples * 2, n_samples * 8)
        min_required = max(16, samples // 20)

    indices = _sobol_indices(
        g_a,
        g_b,
        g_c,
        n_bootstrap=n_bootstrap,
        random_state=random_state,
    )

    return {
        "S": indices["main"],
        "S_total": indices["total"],
        "S_ci": indices["main_ci"],
        "S_total_ci": indices["total_ci"],
    }


def qoi_capacitance(t_h: float) -> Callable[[FKParams], float]:
    def func(theta: FKParams) -> float:
        return float(fractional_capacitance(t_h, theta))

    return func


def qoi_deficit(t_h: float) -> Callable[[FKParams], float]:
    def func(theta: FKParams) -> float:
        return float(normalized_deficit(t_h, theta))

    return func


def qoi_failure_time(q: float) -> Callable[[FKParams], float]:
    def func(theta: FKParams) -> float:
        return float(time_to_threshold(theta, q))

    return func


__all__ = [
    "Prior",
    "LogUniformPrior",
    "BetaPrior",
    "LogNormalPrior",
    "sobol_analysis",
    "qoi_capacitance",
    "qoi_deficit",
    "qoi_failure_time",
]
