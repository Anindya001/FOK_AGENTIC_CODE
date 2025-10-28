"""Diagnostics and performance metrics for FK forecasts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np

from fractional_estimation import FKFitResult, fit_fractional_model
from fractional_model import FKParams, fractional_capacitance

try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None  # type: ignore


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    diff = np.asarray(actual) - np.asarray(predicted)
    return float(np.sqrt(np.mean(diff**2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    diff = np.asarray(actual) - np.asarray(predicted)
    return float(np.mean(np.abs(diff)))


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    return float(100.0 * np.mean(np.abs((actual - predicted) / actual)))


def information_criteria(
    residuals: np.ndarray,
    sigma: float,
    n_params: int,
) -> dict[str, float]:
    m = residuals.size
    ll = -0.5 * m * np.log(2 * np.pi * sigma**2) - 0.5 / sigma**2 * np.dot(residuals, residuals)
    aic = 2 * n_params - 2 * ll
    bic = np.log(m) * n_params - 2 * ll
    return {"loglik": float(ll), "AIC": float(aic), "BIC": float(bic)}


def waic(log_lik_samples: np.ndarray) -> float:
    """Compute WAIC from log-likelihood samples (shape: n_draws x n_obs)."""
    log_lik_samples = np.asarray(log_lik_samples)
    lppd = np.sum(np.log(np.mean(np.exp(log_lik_samples), axis=0)))
    p_waic = np.sum(np.var(log_lik_samples, axis=0))
    return float(-2.0 * (lppd - p_waic))


@dataclass
class ResidualDiagnostics:
    shapiro_p: float | None
    runs_p: float | None


def residual_diagnostics(residuals: np.ndarray) -> ResidualDiagnostics:
    if stats is None:  # pragma: no cover
        return ResidualDiagnostics(shapiro_p=None, runs_p=None)
    shapiro = stats.shapiro(residuals) if residuals.size >= 3 else (None, None)
    signs = np.sign(residuals - np.mean(residuals))
    non_zero = signs[signs != 0]
    if non_zero.size < 2:
        runs_p = None
    else:
        runs = 1 + np.sum(non_zero[1:] != non_zero[:-1])
        n_pos = np.sum(non_zero > 0)
        n_neg = np.sum(non_zero < 0)
        if n_pos == 0 or n_neg == 0:
            runs_p = None
        else:
            total = n_pos + n_neg
            mean_r = 1 + (2 * n_pos * n_neg) / total
            var_r = (
                2 * n_pos * n_neg * (2 * n_pos * n_neg - n_pos - n_neg)
                / (total**2 * (total - 1))
            )
            if var_r <= 0:
                runs_p = None
            else:
                z = (runs - mean_r) / math.sqrt(var_r)
                runs_p = float(math.erfc(abs(z) / math.sqrt(2)))
    return ResidualDiagnostics(
        shapiro_p=shapiro[1] if shapiro else None,
        runs_p=runs_p,
    )


@dataclass
class PrequentialResult:
    times: np.ndarray
    errors: np.ndarray


def prequential_forecast(
    times: Iterable[float],
    values: Iterable[float],
    *,
    min_window: int = 8,
    forecast_horizon: int = 1,
) -> PrequentialResult:
    """Forward-chaining forecast errors using FK fits."""
    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    n = t.size
    errors = []
    times_out = []
    for end in range(min_window, n - forecast_horizon):
        fit_subset = fit_fractional_model(t[:end], y[:end])
        params = fit_subset.params
        future_time = t[end + forecast_horizon - 1]
        pred = fractional_capacitance(future_time, params)
        errors.append(y[end + forecast_horizon - 1] - pred)
        times_out.append(future_time)
    if not errors:
        return PrequentialResult(times=np.array([]), errors=np.array([]))
    return PrequentialResult(times=np.array(times_out), errors=np.array(errors))


__all__ = [
    "rmse",
    "mae",
    "mape",
    "information_criteria",
    "waic",
    "ResidualDiagnostics",
    "residual_diagnostics",
    "PrequentialResult",
    "prequential_forecast",
]
