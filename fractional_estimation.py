"""Parameter estimation for the fractional-kinetics degradation model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np

from fractional_model import FKParams, ensure_monotonic, fractional_capacitance

try:
    from scipy.optimize import least_squares
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "scipy is required for fractional parameter estimation. "
        "Please install scipy>=1.7 to continue."
    ) from exc


ArrayLike = Iterable[float] | np.ndarray


def _logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    eps = 1e-12
    p = min(max(p, eps), 1 - eps)
    return math.log(p / (1.0 - p))


@dataclass
class FKFitResult:
    params: FKParams
    sigma: float
    covariance: np.ndarray
    transformed: np.ndarray
    residuals: np.ndarray
    success: bool
    message: str
    cost: float
    grad_norm: float
    nfev: int
    njev: int
    sigma_log: float
    hessian_cond: Optional[float]
    monotonic: Optional[bool]


def _prepare_inputs(times: ArrayLike, values: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    if t.shape != y.shape:
        raise ValueError("times and values must share the same shape.")
    if t.ndim != 1:
        raise ValueError("times must be one-dimensional.")
    order = np.argsort(t)
    t_sorted = t[order]
    y_sorted = y[order]
    if np.any(y_sorted <= 0):
        raise ValueError("Capacitance values must be strictly positive.")
    return t_sorted, y_sorted


def _initial_guess(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    c0_guess = float(np.max(y))
    tail_count = max(3, min(10, y.size // 4))
    tail_mean = float(np.mean(y[-tail_count:]))
    f_inf_guess = min(max(tail_mean / c0_guess, 1e-3), 0.95)
    alpha_guess = 0.7
    mid_value = f_inf_guess + (1.0 - f_inf_guess) / math.e
    mid_cap = c0_guess * mid_value
    idx = np.searchsorted(y[::-1], mid_cap, side="left")
    if idx == 0:
        k_guess = 1e-3
    else:
        t_mid = float(t[-idx - 1])
        k_guess = max(1e-6, ((1.0 - mid_value) * math.gamma(1.0 + alpha_guess) / t_mid ** alpha_guess))
    return np.array(
        [
            math.log(k_guess),
            _logit(alpha_guess),
            _logit(f_inf_guess),
            math.log(c0_guess),
        ],
        dtype=float,
    )


def _unpack_params(xi: Sequence[float]) -> FKParams:
    kappa, a, b, c0 = xi
    k = math.exp(kappa)
    alpha = _logistic(a)
    f_inf = _logistic(b)
    C0 = math.exp(c0)
    return FKParams(C0=C0, k=k, alpha=alpha, f_inf=f_inf)


def fit_fractional_model(
    times: ArrayLike,
    values: ArrayLike,
    *,
    loss: str = "linear",
    huber_c: float = 1.345,
) -> FKFitResult:
    """Fit FK parameters to log-capacitance data via least squares."""
    t, y = _prepare_inputs(times, values)
    log_y = np.log(y)

    def residuals(xi: np.ndarray) -> np.ndarray:
        params = _unpack_params(xi)
        mu = fractional_capacitance(t, params)
        log_mu = np.log(np.clip(mu, 1e-15, np.inf))
        return log_y - log_mu

    x0 = _initial_guess(t, y)
    result = least_squares(
        residuals,
        x0,
        method="trf",
        loss=loss,
        f_scale=huber_c,
        max_nfev=2000,
    )

    theta_hat = _unpack_params(result.x)
    res = result.fun
    m = res.size
    p = result.x.size
    dof = max(1, m - p)
    sigma_hat = math.sqrt(float(np.dot(res, res)) / dof)

    hessian_cond: Optional[float] = None
    if result.jac is not None and result.jac.size:
        jtj = result.jac.T @ result.jac
        try:
            cov = sigma_hat**2 * np.linalg.inv(jtj)
        except np.linalg.LinAlgError:
            cov = sigma_hat**2 * np.linalg.pinv(jtj)
        try:
            hessian_cond = float(np.linalg.cond(jtj))
        except np.linalg.LinAlgError:
            hessian_cond = None
    else:
        cov = np.full((p, p), np.nan, dtype=float)

    grad_norm = float(np.linalg.norm(result.grad)) if result.grad is not None else float("nan")
    try:
        monotonic_flag = bool(ensure_monotonic(t, theta_hat))
    except Exception:
        monotonic_flag = None

    return FKFitResult(
        params=theta_hat,
        sigma=sigma_hat,
        covariance=cov,
        transformed=result.x.copy(),
        residuals=res,
        success=bool(result.success),
        message=result.message,
        cost=float(result.cost),
        grad_norm=grad_norm,
        nfev=int(result.nfev),
        njev=int(getattr(result, "njev", 0)),
        sigma_log=float(math.log(max(sigma_hat, 1e-12))),
        hessian_cond=hessian_cond,
        monotonic=monotonic_flag,
    )


__all__ = ["FKFitResult", "fit_fractional_model"]
