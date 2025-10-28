"""High-level fractional-kinetics forecasting pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from core import load_dataset
from fractional_conformal import conformal_intervals
from fractional_diagnostics import (
    PrequentialResult,
    information_criteria,
    mae,
    mape,
    prequential_forecast,
    residual_diagnostics,
    rmse,
    waic,
)
from fractional_estimation import FKFitResult, fit_fractional_model
from fractional_model import fractional_capacitance
from fractional_prediction import bootstrap_bias_correction
from fractional_sensitivity import (
    BetaPrior,
    LogNormalPrior,
    LogUniformPrior,
    qoi_capacitance,
    qoi_deficit,
    qoi_failure_time,
    sobol_analysis,
)
from fractional_uq import (
    PredictiveResults,
    failure_time_samples,
    laplace_draws,
    mcmc_draws,
    posterior_predictive,
)


@dataclass
class FractionalConfig:
    train_ratio: float = 0.7
    calibration_fraction: float = 0.2
    confidence: float = 0.9
    n_draws: int = 2000
    bootstrap_draws: int = 512
    thresholds: Sequence[float] = (0.8, 0.7)
    run_sensitivity: bool = False
    sensitivity_horizons: Sequence[float] = (200.0,)
    sobol_samples: int = 2048
    sobol_bootstrap: int = 200
    random_state: Optional[int] = None
    use_mcmc: bool = False
    mcmc_draws: int = 1000
    mcmc_burn_in: int = 300
    mcmc_step_scale: float = 0.5


def _split_series(times: np.ndarray, values: np.ndarray, config: FractionalConfig) -> dict[str, np.ndarray]:
    if not 0.4 <= config.train_ratio <= 0.95:
        raise ValueError("train_ratio must lie in [0.4, 0.95].")
    n_total = times.size
    train_idx = max(int(config.train_ratio * n_total), 8)
    train_idx = min(train_idx, n_total - 2)
    cal_size = max(1, int(config.calibration_fraction * train_idx))
    fit_size = train_idx - cal_size
    if fit_size < 4:
        raise ValueError("Not enough data for fitting after calibration split.")
    return {
        "fit_t": times[:fit_size],
        "fit_y": values[:fit_size],
        "cal_t": times[fit_size:train_idx],
        "cal_y": values[fit_size:train_idx],
        "forecast_t": times[train_idx:],
        "forecast_y": values[train_idx:],
        "train_t": times[:train_idx],
        "train_y": values[:train_idx],
        "train_idx": train_idx,
        "fit_size": fit_size,
    }


def _predictive_bands(results: PredictiveResults, start_idx: int) -> dict[str, np.ndarray]:
    epistemic = np.full(results.time.shape, np.nan, dtype=float)
    total = np.full_like(epistemic, np.nan)
    if start_idx < results.time.size:
        low_high = np.quantile(results.mu_samples[:, start_idx:], [0.025, 0.975], axis=0)
        epistemic[start_idx:] = low_high[0]
        epistemic_high = np.full_like(epistemic, np.nan)
        epistemic_high[start_idx:] = low_high[1]
    else:
        epistemic_high = np.copy(epistemic)
    if start_idx < results.time.size:
        low_high_total = np.quantile(results.total_samples[:, start_idx:], [0.025, 0.975], axis=0)
        total[start_idx:] = low_high_total[0]
        total_high = np.full_like(total, np.nan)
        total_high[start_idx:] = low_high_total[1]
    else:
        total_high = np.copy(total)
    return {
        "epistemic_low": epistemic,
        "epistemic_high": epistemic_high,
        "total_low": total,
        "total_high": total_high,
    }


def _build_priors(fit: FKFitResult) -> dict[str, object]:
    theta = fit.params
    return {
        "C0": LogNormalPrior(mean=np.log(theta.C0), sigma=0.2),
        "k": LogUniformPrior(low=max(theta.k / 5.0, 1e-8), high=theta.k * 5.0),
        "alpha": BetaPrior(a=2 + theta.alpha * 5, b=2 + (1 - theta.alpha) * 5),
        "f_inf": BetaPrior(a=2 + theta.f_inf * 5, b=2 + (1 - theta.f_inf) * 5),
    }


class FractionalPipelineError(RuntimeError):
    """Raised when the fractional pipeline encounters an unrecoverable failure."""


class FractionalPICPCore:
    def __init__(self, config: Optional[FractionalConfig] = None):
        self.config = config or FractionalConfig()

    def run_forecast(
        self,
        data: pd.DataFrame | str,
        *,
        time_column: Optional[str] = None,
        target_column: Optional[str] = None,
    ) -> dict[str, object]:
        try:
            return self._run_forecast_inner(
                data,
                time_column=time_column,
                target_column=target_column,
            )
        except FractionalPipelineError:
            raise
        except Exception as exc:
            raise FractionalPipelineError(f"Fractional pipeline failed: {exc}") from exc

    def _run_forecast_inner(
        self,
        data: pd.DataFrame | str,
        *,
        time_column: Optional[str] = None,
        target_column: Optional[str] = None,
    ) -> dict[str, object]:
        frame, time_col, target_col = load_dataset(data, time_column, target_column)
        frame = frame[[time_col, target_col]].dropna()
        times = frame[time_col].to_numpy(dtype=float)
        values = frame[target_col].to_numpy(dtype=float)
        order = np.argsort(times)
        times = times[order]
        values = values[order]

        splits = _split_series(times, values, self.config)

        try:
            fit_result = fit_fractional_model(splits["train_t"], splits["train_y"])
        except Exception as exc:
            raise FractionalPipelineError(f"Fractional model fit failed: {exc}") from exc
        params_hat = fit_result.params
        bias_summary: Optional[dict[str, float]] = None
        if self.config.bootstrap_draws > 0 and fit_result.residuals.size:
            try:
                bias_summary = bootstrap_bias_correction(
                    fit_result.residuals,
                    n_bootstrap=self.config.bootstrap_draws,
                    random_state=self.config.random_state,
                )
            except Exception:
                bias_summary = None

        try:
            params_draws, sigma_draws = laplace_draws(
                fit_result,
                n_draws=self.config.n_draws,
                random_state=self.config.random_state,
            )
        except Exception as exc:
            raise FractionalPipelineError(f"Laplace sampling failed: {exc}") from exc
        mcmc_output: Optional[dict[str, object]] = None
        if self.config.use_mcmc:
            try:
                mcmc_params, mcmc_sigma, acceptance_rate = mcmc_draws(
                    fit_result,
                    splits["train_t"],
                    splits["train_y"],
                    n_draws=self.config.mcmc_draws,
                    burn_in=self.config.mcmc_burn_in,
                    step_scale=self.config.mcmc_step_scale,
                    random_state=self.config.random_state,
                )
            except Exception as exc:  # pragma: no cover - optional path
                mcmc_output = {"error": str(exc)}
            else:
                mcmc_output = {
                    "acceptance_rate": float(acceptance_rate),
                    "sigma_draws": mcmc_sigma.tolist(),
                    "params": {
                        "C0": [float(theta.C0) for theta in mcmc_params],
                        "k": [float(theta.k) for theta in mcmc_params],
                        "alpha": [float(theta.alpha) for theta in mcmc_params],
                        "f_inf": [float(theta.f_inf) for theta in mcmc_params],
                    },
                }

        t_plot = times
        try:
            predictive = posterior_predictive(
                params_draws,
                sigma_draws,
                t_plot,
                random_state=self.config.random_state,
            )
        except Exception as exc:
            raise FractionalPipelineError(f"Predictive sampling failed: {exc}") from exc
        bias_factor = None
        mean_bias_corrected: Optional[np.ndarray] = None
        if bias_summary is not None:
            bias_factor = bias_summary.get("bias_factor")
            if isinstance(bias_factor, (float, int)) and np.isfinite(bias_factor) and bias_factor > 0:
                mean_bias_corrected = predictive.mean / float(bias_factor)

        train_idx = splits["train_idx"]
        fit_size = splits["fit_size"]
        forecast_count = splits["forecast_t"].size
        forecast_start = train_idx
        cal_size = splits["cal_t"].size

        bands = _predictive_bands(predictive, forecast_start)

        if cal_size > 0 and forecast_count > 0:
            calibration_samples = predictive.total_samples[:, fit_size:train_idx]
            test_samples = predictive.total_samples[:, train_idx : train_idx + forecast_count]
            try:
                conformal = conformal_intervals(
                    splits["cal_y"],
                    calibration_samples,
                    test_samples,
                    alpha=1.0 - self.config.confidence,
                )
            except Exception as exc:
                raise FractionalPipelineError(f"Conformal calibration failed: {exc}") from exc
            conformal_low = np.full(predictive.time.shape, np.nan, dtype=float)
            conformal_high = np.full_like(conformal_low, np.nan)
            conformal_low[forecast_start : forecast_start + forecast_count] = conformal["lower"]
            conformal_high[forecast_start : forecast_start + forecast_count] = conformal["upper"]
        else:
            conformal = None
            conformal_low = np.full(predictive.time.shape, np.nan, dtype=float)
            conformal_high = np.full_like(conformal_low, np.nan)

        hybrid_low = np.full_like(conformal_low, np.nan)
        hybrid_high = np.full_like(conformal_high, np.nan)
        if forecast_count > 0:
            total_low = bands["total_low"]
            total_high = bands["total_high"]
            start = forecast_start
            end = forecast_start + forecast_count
            total_lo_slice = total_low[start:end]
            total_hi_slice = total_high[start:end]
            conf_lo_slice = conformal_low[start:end]
            conf_hi_slice = conformal_high[start:end]
            combined_low = np.nanmin(np.vstack([conf_lo_slice, total_lo_slice]), axis=0)
            combined_high = np.nanmax(np.vstack([conf_hi_slice, total_hi_slice]), axis=0)
            hybrid_low[start:end] = combined_low
            hybrid_high[start:end] = combined_high

        train_pred = fractional_capacitance(splits["train_t"], params_hat)
        forecast_pred = fractional_capacitance(splits["forecast_t"], params_hat)

        metrics = {
            "rmse_train": rmse(splits["train_y"], train_pred),
            "mae_train": mae(splits["train_y"], train_pred),
            "mape_train": mape(splits["train_y"], train_pred),
        }
        metrics["sigma_log"] = float(fit_result.sigma_log)
        if forecast_count:
            metrics["rmse_forecast"] = rmse(splits["forecast_y"], forecast_pred)
            metrics["mae_forecast"] = mae(splits["forecast_y"], forecast_pred)
            metrics["mape_forecast"] = mape(splits["forecast_y"], forecast_pred)
            if conformal is not None:
                lower = conformal["lower"]
                upper = conformal["upper"]
                coverage = np.mean(
                    (splits["forecast_y"] >= lower) & (splits["forecast_y"] <= upper)
                )
                metrics["conformal_coverage"] = float(coverage)
        if bias_factor is not None:
            metrics["bias_factor"] = float(bias_factor)
        if fit_result.hessian_cond is not None:
            metrics["hessian_condition"] = float(fit_result.hessian_cond)
        if fit_result.monotonic is not None:
            metrics["monotonic_training"] = bool(fit_result.monotonic)
        if mcmc_output and isinstance(mcmc_output, dict) and "acceptance_rate" in mcmc_output:
            metrics["mcmc_acceptance"] = float(mcmc_output["acceptance_rate"])
        info = information_criteria(fit_result.residuals, fit_result.sigma, 4)

        log_y_train = np.log(splits["train_y"])
        mu_samples_train = predictive.mu_samples[:, :train_idx]
        log_mu_samples = np.log(np.clip(mu_samples_train, 1e-15, np.inf))
        log_lik_samples = (
            -log_y_train[None, :]
            - np.log(sigma_draws)[:, None]
            - 0.5 * np.log(2 * np.pi)
            - ((log_y_train[None, :] - log_mu_samples) ** 2) / (2 * (sigma_draws[:, None] ** 2))
        )
        waic_value = waic(log_lik_samples)

        residual_info = residual_diagnostics(fit_result.residuals)
        prequential = prequential_forecast(times, values) if forecast_count else PrequentialResult(np.array([]), np.array([]))

        try:
            failure_samples = failure_time_samples(params_draws, self.config.thresholds)
        except Exception as exc:
            raise FractionalPipelineError(f"Failure time sampling failed: {exc}") from exc
        failure_quantiles = {
            q: {
                "q05": float(np.nanpercentile(samples, 5)),
                "median": float(np.nanpercentile(samples, 50)),
                "q95": float(np.nanpercentile(samples, 95)),
            }
            for q, samples in failure_samples.items()
        }
        mean_bias_corrected_list = (
            mean_bias_corrected.tolist() if mean_bias_corrected is not None else None
        )

        sensitivity_results = None
        if self.config.run_sensitivity:
            priors = _build_priors(fit_result)
            sensitivity_results = {}

            def _safe_sobol(name: str, func) -> None:
                try:
                    sensitivity_results[name] = sobol_analysis(
                        priors,
                        func,
                        n_samples=self.config.sobol_samples,
                        n_bootstrap=self.config.sobol_bootstrap,
                        random_state=self.config.random_state,
                    )
                except Exception as exc:
                    sensitivity_results[name] = {"error": str(exc)}

            for horizon in self.config.sensitivity_horizons:
                _safe_sobol(f"Y({horizon})", qoi_capacitance(horizon))
                _safe_sobol(f"Delta({horizon})", qoi_deficit(horizon))
            for q in self.config.thresholds:
                _safe_sobol(f"T({q})", qoi_failure_time(q))

        return {
            "success": True,
            "time_column": time_col,
            "target_column": target_col,
            "confidence": self.config.confidence,
            "train_ratio": self.config.train_ratio,
            "data": {
                "times": times.tolist(),
                "values": values.tolist(),
                "train_count": int(splits["train_t"].size),
                "calibration_count": int(splits["cal_t"].size),
                "forecast_count": int(forecast_count),
            },
            "fit": {
                "params": {
                    "C0": params_hat.C0,
                    "k": params_hat.k,
                    "alpha": params_hat.alpha,
                    "f_inf": params_hat.f_inf,
                },
                "sigma": fit_result.sigma,
                "sigma_log": float(fit_result.sigma_log),
                "hessian_condition": fit_result.hessian_cond,
                "monotonic": fit_result.monotonic,
                "covariance": fit_result.covariance.tolist(),
                "residuals": fit_result.residuals.tolist(),
                "success": fit_result.success,
                "message": fit_result.message,
            },
            "forecast": {
                "time": predictive.time.tolist(),
                "mean": predictive.mean.tolist(),
                "mean_bias_corrected": mean_bias_corrected_list,
                "epistemic_low": bands["epistemic_low"].tolist(),
                "epistemic_high": bands["epistemic_high"].tolist(),
                "total_low": bands["total_low"].tolist(),
                "total_high": bands["total_high"].tolist(),
                "conformal_low": conformal_low.tolist(),
                "conformal_high": conformal_high.tolist(),
                "hybrid_low": hybrid_low.tolist(),
                "hybrid_high": hybrid_high.tolist(),
            },
            "metrics": {
                **metrics,
                **info,
                "WAIC": waic_value,
                "residual_shapiro_p": residual_info.shapiro_p,
                "residual_runs_p": residual_info.runs_p,
            },
            "failure_time": {
                "thresholds": list(self.config.thresholds),
                "quantiles": failure_quantiles,
            },
            "posterior": {
                "sigma_draws": sigma_draws.tolist(),
                "mcmc": mcmc_output,
            },
            "sensitivity": sensitivity_results,
            "prequential": {
                "times": prequential.times.tolist(),
                "errors": prequential.errors.tolist(),
            },
            "bias_correction": bias_summary,
        }


__all__ = ["FractionalPICPCore", "FractionalConfig", "FractionalPipelineError"]
