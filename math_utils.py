"""
Numerical utilities for PICP with robust fallbacks.
All functions use numpy arrays; fallbacks to mpmath for precision.
"""

from __future__ import annotations

from typing import Iterable

import math
import numpy as np

try:
    from scipy.special import mittag_leffler as _scipy_mittag_leffler, gammaln  # type: ignore
except Exception:  # pragma: no cover
    _scipy_mittag_leffler = None
    gammaln = None  # type: ignore

try:  # Optional high-precision fallback
    import mpmath
except Exception:  # pragma: no cover - used only when available
    mpmath = None  # type: ignore

__all__ = ["mittag_leffler", "mittag_leffler_two_param", "safe_exp_decay", "stable_log_transform"]


def _to_numpy_array(values: Iterable[float] | np.ndarray) -> np.ndarray:
    """Convert *values* to a 1-D numpy array of floats."""
    arr = np.asarray(values, dtype=float)
    return arr.copy() if arr.ndim > 1 else arr.reshape(-1)


def mittag_leffler(alpha: float, z: Iterable[float] | np.ndarray, tol: float = 1e-10) -> np.ndarray:
    r"""Mittag-Leffler function :math:`E_\\alpha(z)` with domain checks.

    Parameters
    ----------
    alpha
        Order parameter. Must be strictly positive.
    z
        Evaluation points. Real-valued iterable/array.
    tol
        Numerical tolerance used to zero-out tiny imaginary components and to
        configure arbitrary-precision fallbacks.

    Returns
    -------
    np.ndarray
        Array of the same shape as :paramref:`z` containing real-valued
        :math:`E_\alpha(z)` evaluations.

    Raises
    ------
    ValueError
        If ``alpha <= 0`` or ``tol <= 0``.
    RuntimeError
        If neither SciPy nor mpmath backends are available.
    """
    if alpha <= 0:
        raise ValueError("alpha must be strictly positive")
    if tol <= 0:
        raise ValueError("tol must be strictly positive")

    z_arr = np.asarray(z, dtype=float)
    original_shape = np.asarray(z).shape
    if z_arr.ndim == 0:
        z_arr = z_arr.reshape(1)

    finite_mask = np.isfinite(z_arr)
    result = np.full_like(z_arr, np.nan, dtype=float)

    eval_points = z_arr[finite_mask]

    if eval_points.size == 0:
        return result.reshape(original_shape)

    def _post_process(values: np.ndarray | list[float]) -> np.ndarray:
        processed = np.array(values, dtype=np.complex128)
        processed = np.real_if_close(processed, tol=tol)
        processed = processed.astype(float, copy=False)
        processed[~np.isfinite(processed)] = np.nan
        return processed

    backend_success = False

    if _scipy_mittag_leffler is not None:
        try:
            with np.errstate(over="raise", invalid="raise"):
                ml_values = _scipy_mittag_leffler(eval_points, alpha, 1.0)
            result[finite_mask] = _post_process(ml_values)
            backend_success = True
        except FloatingPointError:
            backend_success = False
        except Exception:
            backend_success = False
        except FloatingPointError:
            backend_success = False
        except Exception:
            backend_success = False

    if not backend_success:
        if mpmath is not None:
            mp_ctx = mpmath.mp  # type: ignore[attr-defined]
            default_dps = mp_ctx.dps
            try:
                mp_ctx.dps = max(default_dps, int(-np.log10(tol)) + 6)
                alpha_mp = mp_ctx.mpf(alpha)

                def _mittag_series_mp(z_val: "mpmath.mpf") -> "mpmath.mpf":
                    total = mp_ctx.mpf(1)
                    term = mp_ctx.mpf(1)
                    k = 1
                    while abs(term) > tol and k < 600:
                        gamma_arg = alpha_mp * k + 1
                        denom = mpmath.gamma(gamma_arg)
                        term = (z_val ** k) / denom
                        total += term
                        k += 1
                    return total

                def _mp_eval(value: float) -> float:
                    z_val = mp_ctx.mpf(value)
                    if hasattr(mpmath, "mittag_leffler"):
                        ml = mpmath.mittag_leffler(alpha_mp, 1, z_val)  # type: ignore[attr-defined]
                    else:
                        ml = _mittag_series_mp(z_val)
                    if abs(mpmath.im(ml)) <= tol:
                        return float(mpmath.re(ml))
                    return float(ml)

                mp_values = [_mp_eval(float(val)) for val in eval_points.tolist()]
                result[finite_mask] = _post_process(mp_values)
            finally:
                mp_ctx.dps = default_dps
        else:
            series_values = []
            for val in eval_points:
                term = 1.0
                total = 1.0
                k = 1
                while abs(term) > tol and k < 200:
                    if gammaln is not None:
                        denominator = np.exp(gammaln(alpha * k + 1.0))
                    else:
                        denominator = math.gamma(alpha * k + 1.0)
                    term = (val ** k) / denominator
                    total += term
                    k += 1
                series_values.append(total)
            result[finite_mask] = _post_process(series_values)

    tiny_mask = np.abs(result) < tol
    result[tiny_mask] = 0.0

    return result.reshape(original_shape)


def mittag_leffler_two_param(
    alpha: float,
    beta: float,
    z: Iterable[float] | np.ndarray,
    tol: float = 1e-10,
) -> np.ndarray:
    """Evaluate the two-parameter Mittag-Leffler function :math:`E_{\\alpha,\\beta}(z)`."""
    if alpha <= 0:
        raise ValueError("alpha must be strictly positive")
    if beta <= 0:
        raise ValueError("beta must be strictly positive")
    if tol <= 0:
        raise ValueError("tol must be strictly positive")

    z_arr = np.asarray(z, dtype=float)
    original_shape = np.asarray(z).shape
    if z_arr.ndim == 0:
        z_arr = z_arr.reshape(1)

    finite_mask = np.isfinite(z_arr)
    result = np.full_like(z_arr, np.nan, dtype=float)
    eval_points = z_arr[finite_mask]
    if eval_points.size == 0:
        return result.reshape(original_shape)

    def _post_process(values: np.ndarray | list[float]) -> np.ndarray:
        processed = np.array(values, dtype=np.complex128)
        processed = np.real_if_close(processed, tol=tol)
        processed = processed.astype(float, copy=False)
        processed[~np.isfinite(processed)] = np.nan
        return processed

    backend_success = False
    if _scipy_mittag_leffler is not None:
        try:
            with np.errstate(over="raise", invalid="raise"):
                ml_values = _scipy_mittag_leffler(eval_points, alpha, beta)
            result[finite_mask] = _post_process(ml_values)
            backend_success = True
        except FloatingPointError:
            backend_success = False
        except Exception:
            backend_success = False

    if not backend_success:
        if mpmath is not None:
            mp_ctx = mpmath.mp  # type: ignore[attr-defined]
            default_dps = mp_ctx.dps
            try:
                mp_ctx.dps = max(default_dps, int(-np.log10(tol)) + 6)
                alpha_mp = mp_ctx.mpf(alpha)
                beta_mp = mp_ctx.mpf(beta)

                def _mp_eval(value: float) -> float:
                    z_val = mp_ctx.mpf(value)
                    ml = mpmath.mittag_leffler(alpha_mp, beta_mp, z_val)  # type: ignore[attr-defined]
                    if abs(mpmath.im(ml)) <= tol:
                        return float(mpmath.re(ml))
                    return float(ml)

                mp_values = [_mp_eval(float(val)) for val in eval_points.tolist()]
                result[finite_mask] = _post_process(mp_values)
            finally:
                mp_ctx.dps = default_dps
        else:
            series_values = []
            for val in eval_points:
                if gammaln is not None:
                    total = float(np.exp(-gammaln(beta)))
                else:
                    total = 1.0 / math.gamma(beta)
                term = total
                k = 1
                while abs(term) > tol and k < 200:
                    power = alpha * k + beta
                    if gammaln is not None:
                        denom = np.exp(gammaln(power))
                    else:
                        denom = math.gamma(power)
                    term = (val ** k) / denom
                    total += term
                    k += 1
                series_values.append(total)
            result[finite_mask] = _post_process(series_values)

    tiny_mask = np.abs(result) < tol
    result[tiny_mask] = 0.0
    return result.reshape(original_shape)


def safe_exp_decay(t: Iterable[float] | np.ndarray, tau: float, floor: float = 1e-12) -> np.ndarray:
    r"""Numerically stable exponential decay :math:`\\exp(-t/\\tau)` with flooring.

    Parameters
    ----------
    t
        Time values (any real iterable).
    tau
        Characteristic time scale. Must be strictly positive.
    floor
        Minimum value returned to avoid exact zeros (default ``1e-12``).

    Returns
    -------
    np.ndarray
        Decay values clipped to ``[floor, 1]``.
    """
    if tau <= 0:
        raise ValueError("tau must be strictly positive")
    if floor <= 0:
        raise ValueError("floor must be strictly positive")

    t_arr = np.asarray(t, dtype=float)
    scaled = t_arr / tau

    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        clipped = np.clip(-scaled, -700.0, 700.0)  # avoid overflow in exp
        decay = np.exp(clipped)

    decay = np.clip(decay, floor, 1.0)
    decay[~np.isfinite(decay)] = floor
    return decay.reshape(np.asarray(t).shape)


def stable_log_transform(y: Iterable[float] | np.ndarray, offset: float = 1e-8) -> np.ndarray:
    """Log transform with offset for zero-avoidance.

    Parameters
    ----------
    y
        Input values (iterable / array).
    offset
        Small positive constant added to ``y`` before taking the logarithm.

    Returns
    -------
    np.ndarray
        ``log(y + offset)`` while guarding against non-positive arguments.
    """
    if offset <= 0:
        raise ValueError("offset must be strictly positive")

    y_arr = np.asarray(y, dtype=float)
    adjusted = y_arr + offset
    adjusted[adjusted <= 0] = offset

    with np.errstate(divide="ignore", invalid="ignore"):
        transformed = np.log(adjusted)

    transformed[~np.isfinite(transformed)] = np.log(offset)
    return transformed.reshape(np.asarray(y).shape)
