"""Prequential conformal prediction for fractional-kinetics forecasts."""

from __future__ import annotations

import numpy as np


def _median_and_mad(samples: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(samples, axis=0)
    mad = np.median(np.abs(samples - median), axis=0)
    mad = np.where(mad > 0, mad, eps)
    return median, mad


def conformal_intervals(
    calibration_obs: np.ndarray,
    calibration_samples: np.ndarray,
    test_samples: np.ndarray,
    *,
    alpha: float = 0.1,
    eps: float = 1e-9,
) -> dict[str, np.ndarray]:
    """Compute conformal prediction intervals using calibration samples."""
    if calibration_samples.ndim != 2:
        raise ValueError("calibration_samples must be 2-D (n_samples, n_calibration).")
    if test_samples.ndim != 2:
        raise ValueError("test_samples must be 2-D (n_samples, n_test).")
    if calibration_samples.shape[0] != test_samples.shape[0]:
        raise ValueError("Calibration and test sample counts must match.")
    if calibration_samples.shape[1] != calibration_obs.size:
        raise ValueError("Mismatch between calibration observations and samples.")

    centers_cal, scales_cal = _median_and_mad(calibration_samples, eps)
    scores = np.abs(calibration_obs - centers_cal) / (scales_cal + eps)
    m = scores.size
    sorted_scores = np.sort(scores)
    quantile_level = min(1.0, max(0.0, (1.0 - alpha) * (1.0 + 1.0 / (m + 1.0))))
    rank = int(np.ceil(quantile_level * m)) - 1
    rank = min(max(rank, 0), m - 1)
    q_hat = sorted_scores[rank]

    centers_test, scales_test = _median_and_mad(test_samples, eps)
    lower = centers_test - q_hat * scales_test
    upper = centers_test + q_hat * scales_test

    return {
        "center": centers_test,
        "scale": scales_test,
        "lower": lower,
        "upper": upper,
        "q_hat": q_hat,
        "alpha": alpha,
    }


__all__ = ["conformal_intervals"]
