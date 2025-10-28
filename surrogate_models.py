"""Surrogate degradation models for comparison with FK."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

try:
    from scipy.optimize import least_squares
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "scipy is required for surrogate fitting. Install scipy>=1.7."
    ) from exc


ArrayLike = Iterable[float] | np.ndarray


def _split_series(times: np.ndarray, values: np.ndarray, train_ratio: float) -> dict[str, np.ndarray]:
    n_total = times.size
    train_idx = max(int(train_ratio * n_total), 6)
    train_idx = min(train_idx, n_total - 2)
    fit_t = times[:train_idx]
    fit_y = values[:train_idx]
    forecast_t = times[train_idx:]
    forecast_y = values[train_idx:]
    return {
        "fit_t": fit_t,
        "fit_y": fit_y,
        "forecast_t": forecast_t,
        "forecast_y": forecast_y,
        "train_idx": train_idx,
    }


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_true - y_pred
    return float(np.sqrt(np.mean(diff**2)))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(100.0 * np.mean(np.abs((y_true - y_pred) / y_true)))


def fit_classical(times: ArrayLike, values: ArrayLike, train_ratio: float = 0.7) -> dict[str, object]:
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    order = np.argsort(times)
    times = times[order]
    values = values[order]
    splits = _split_series(times, values, train_ratio)
    t_fit = splits["fit_t"]
    y_fit = splits["fit_y"]

    def residuals(theta: np.ndarray) -> np.ndarray:
        c_inf = theta[0]
        delta = theta[1]
        lam = theta[2]
        lam = max(lam, 1e-6)
        pred = c_inf + delta * np.exp(-lam * t_fit)
        return pred - y_fit

    c0 = float(y_fit[0])
    c_inf_guess = float(np.percentile(y_fit, 20))
    delta_guess = c0 - c_inf_guess
    lam_guess = 1.0 / max(t_fit[-1] - t_fit[0], 1.0)
    x0 = np.array([c_inf_guess, delta_guess, lam_guess], dtype=float)

    result = least_squares(residuals, x0, method="trf")
    c_inf, delta, lam = result.x
    lam = max(lam, 1e-6)

    pred_all = c_inf + delta * np.exp(-lam * times)
    train_pred = pred_all[: splits["train_idx"]]
    forecast_pred = pred_all[splits["train_idx"] :]

    metrics = {
        "rmse_train": _rmse(values[: splits["train_idx"]], train_pred),
        "mape_train": _mape(values[: splits["train_idx"]], train_pred),
    }
    if splits["forecast_t"].size:
        metrics["rmse_forecast"] = _rmse(values[splits["train_idx"] :], forecast_pred)
        metrics["mape_forecast"] = _mape(values[splits["train_idx"] :], forecast_pred)

    return {
        "params": {
            "C_inf": float(c_inf),
            "Delta": float(delta),
            "lambda": float(lam),
        },
        "metrics": metrics,
        "time": times,
        "prediction": pred_all,
        "train_idx": splits["train_idx"],
    }


def fit_kww(times: ArrayLike, values: ArrayLike, train_ratio: float = 0.7) -> dict[str, object]:
    times = np.asarray(times, dtype=float)
    values = np.asarray(values, dtype=float)
    order = np.argsort(times)
    times = times[order]
    values = values[order]
    splits = _split_series(times, values, train_ratio)
    t_fit = splits["fit_t"]
    y_fit = splits["fit_y"]

    def residuals(theta: np.ndarray) -> np.ndarray:
        c_inf = theta[0]
        tau = np.exp(theta[1])
        beta = 1.0 / (1.0 + np.exp(-theta[2]))
        c0 = theta[3]
        pred = c_inf + (c0 - c_inf) * np.exp(-np.power(np.maximum(t_fit, 0.0) / tau, beta))
        return pred - y_fit

    c0_guess = float(y_fit[0])
    c_inf_guess = float(np.percentile(y_fit, 10))
    tau_guess = max(np.median(np.diff(t_fit)), 1.0)
    beta_guess = 0.7
    x0 = np.array([c_inf_guess, math.log(tau_guess), math.log(beta_guess / (1 - beta_guess)), c0_guess])

    result = least_squares(residuals, x0, method="trf")
    c_inf = result.x[0]
    tau = math.exp(result.x[1])
    beta = 1.0 / (1.0 + math.exp(-result.x[2]))
    c0 = result.x[3]

    pred_all = c_inf + (c0 - c_inf) * np.exp(-np.power(np.maximum(times, 0.0) / tau, beta))
    train_pred = pred_all[: splits["train_idx"]]
    forecast_pred = pred_all[splits["train_idx"] :]

    metrics = {
        "rmse_train": _rmse(values[: splits["train_idx"]], train_pred),
        "mape_train": _mape(values[: splits["train_idx"]], train_pred),
    }
    if splits["forecast_t"].size:
        metrics["rmse_forecast"] = _rmse(values[splits["train_idx"] :], forecast_pred)
        metrics["mape_forecast"] = _mape(values[splits["train_idx"] :], forecast_pred)

    return {
        "params": {
            "C_inf": float(c_inf),
            "tau": float(tau),
            "beta": float(beta),
            "C0": float(c0),
        },
        "metrics": metrics,
        "time": times,
        "prediction": pred_all,
        "train_idx": splits["train_idx"],
    }


__all__ = ["fit_classical", "fit_kww"]
