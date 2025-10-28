# Implementation Blueprint: Fractional-Order FK Pipeline + UI Panels

This document is the canonical spec for upgrading the repo at C:\Users\Anindya\Desktop\P3CODE\AGENTIC with the fractional-kinetics (FK) model, Sobol sensitivity, Laplace/Bayesian UQ, conformal prediction, and the matching PyQt UI. Treat it as the to-do list for building Section 2.1–2.4 of the manuscript.

---
## 1. Backend Modules – Build List

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| ractional_model.py | FK evaluation + derivatives | c_fk(t, theta), dc_dt(t, theta), delta_h(t, theta), solve_Tq(theta, q) |
| ractional_estimation.py | Constrained MLE, log-normal noise | it_fk(data) -> (theta_hat, sigma_hat, cov, diag) |
| ractional_prediction.py | Deterministic outputs | predict_curve(theta_hat, t_grid), ailure_time(theta_hat, q) |
| ractional_uq.py | Laplace/MCMC sampling + predictive | laplace_draws, posterior_predictive |
| ractional_conformal.py | Prequential conformal | conformal_calibrate(calibration_obs, pred_samples, alpha) |
| ractional_sensitivity.py | Sobol indices for QoIs | sobol_indices(qoi, priors, t_h, q_list) |
| ractional_diagnostics.py | Metrics & adequacy | mse, mape, info_criteria, esidual_tests, prequential_cv |
| ractional_core.py | Orchestrator | Glue of all modules, returns result dict |

### 1.1 FK Model Details
- c_fk(t, theta) implements \(C(t) = C_0[f_\infty + (1-f_\infty)E_\alpha(-k t^\alpha)]\).
- Use scipy.special.mittag_leffler for both \(E_\alpha\) and \(E_{\alpha,\alpha}\).
- Vectorised NumPy inputs; add log-space branch for large arguments.
- solve_Tq: Newton with derivative, fallback to bisection; guard 0<_inf<1.
- Monotonicity helper: assert 
p.all(np.diff(c_fk(t_obs)) <= 1e-6); flag diagnostics otherwise.

### 1.2 Estimation
- Observation model: log-normal errors (Eq. 2.3.1).
- Parameter transforms: k = exp(kappa), lpha = logistic(a), _inf = logistic(b), C0 = exp(c0); treat sigma = exp(s).
- Initial guesses: max early value for C0, late plateau for _inf, log-log slope for lpha, 1/e decay for k.
- Optimiser: scipy.optimize.least_squares(..., method="trf"); allow Huber loss toggle.
- Outputs: natural-scale parameters, covariance (Gauss–Newton), gradient norm, Hessian condition number.
- If Hessian ill-conditioned, support optional priors / fixing _inf.

### 1.3 Forecast & Deterministic Outputs
- predict_curve: compute \(\hat C(t)\) on requested grid.
- Failure time solver using solve_Tq per threshold; bisection fallback.
- Provide bootstrap bias correction when sample small.

### 1.4 Uncertainty & Predictive Sampling
- **Laplace draws:** 	heta_s = theta_hat + L @ z; default 2000 draws.
- **Posterior predictive:** for each draw, mu_s(t) (epistemic) and C_star(t) via log-normal noise (total).
- Bands: 2.5/97.5 quantiles of mu_s (epistemic) and C_star (total).
- Failure-time distribution: solve threshold for each sample -> quantiles (5, 50, 95%).
- Optional MCMC (emcee) with priors lpha~Beta(2,2), _inf~Beta(2,5), log k~N(0,5), log C0~N(0,5), sigma~HalfNormal(1).

### 1.5 Conformal Calibration
- Partition time series into train / calibration / test at forecast boundary.
- For each calibration index:
  1. center = median(pred_samples)
  2. scale = MAD(pred_samples)
  3. Score S = abs(obs - center) / (scale + eps).
- q_hat = np.quantile(scores, (1 - alpha) * (1 + 1/(m+1))).
- For future times, compute [center ± q_hat * scale]; deliver as conformal_low/high.
- Guarantee coverage ≥ 1 - alpha - 1/(m+1).

### 1.6 Sobol Sensitivity
- Provide QoIs: Y_h, Delta_h, T_q.
- Parameter priors from ALT data: e.g. k log-uniform, lpha Beta, _inf Beta, C0 log-normal.
- Use Saltelli sampling (6N evaluations). Provide bootstrap CI.
- Return dict with main, 	otal, ci for each parameter and QoI.

### 1.7 Orchestrator (ractional_core.py)
Pseudo flow:
`python
def run_forecast(data, config):
    df, time_col, target_col = load_dataset(...)
    fit = fractional_estimation.fit_fk(df)
    sens = fractional_sensitivity.sobol_indices(...) if config.do_sa else None
    curve = fractional_prediction.predict_curve(fit.theta, config.t_grid)
    failure = fractional_prediction.failure_time(fit.theta, config.thresholds)
    draws = fractional_uq.laplace_draws(fit)
    predictive = fractional_uq.posterior_predictive(draws, config.t_grid)
    conformal = fractional_conformal.conformal_calibrate(calibration_block, predictive.samples)
    diag = fractional_diagnostics.compute_all(df, fit, predictive, conformal)
    return {...}
`

---
## 2. UI / Visualization Specification

Implement three tabs (QTabWidget) plus a diagnostics drawer:

### Tab 1: Overview
- **Table** with parameter estimates & 95% CI (alpha, k, f_inf, C0, sigma).
- **Metrics**: RMSE, MAPE, AIC, BIC, WAIC, log-likelihood.
- **Coverage**: conformal target vs achieved.
- **Residual summary**: Shapiro p-value, runs test p-value, monotonicity flag.

### Tab 2: Sensitivity
- **Controls**: dropdown 	_h (default mission horizon), slider for threshold q (0.6–0.9).
- **Bar charts**:
  - top subplot: first-order S_j by parameter.
  - bottom subplot: total S_j^T by parameter.
- **CI table**: show index ± bootstrap CI.
- **Screening message**: indicate parameters with S^T < 0.01 removed from UQ.

### Tab 3: Forecast & UQ
- **Main Matplotlib figure**:
  - training scatter (light blue dots).
  - FK fit on training window (dark dashed line).
  - forecast boundary (vertical red dashed line).
  - forecast mean (navy solid).
  - epistemic band (teal fill), total band (light teal fill), conformal band (gold outline / hatch).
  - observed tail (orange markers).
  - optional threshold lines (toggleable).
- **Legend**: differentiate all series.
- **Metrics box**: anchored text with RMSE, MAPE, coverage, T_q quantiles.
- **Controls**: checkboxes to toggle bands, slider to adjust forecast horizon, threshold selector.

### Diagnostics Drawer (collapsible side panel)
- Residual histogram + normal curve.
- Normal QQ plot.
- Prequential CV plot (time vs error).
- Table of posterior sample summaries (mean, std, 95% interval) for each parameter.
- Button to export posterior draws / diagnostic report.

### Status & Notifications
- Status label at bottom: Ready → Running analysis… (with busy cursor) → Analysis complete.
- Message box on completion summarising coverage guarantee.

---
## 3. Data Flow (Backend → UI)

1. User clicks “Run analysis”: disable button, set busy cursor, update status label.
2. Call ractional_core.run_forecast(...) with current settings.
3. Store result in _last_result (same pattern as legacy code).
4. Update panels:
   - populate_overview(result)
   - update_sensitivity(result["sensitivity"], current_th, current_q)
   - plot_forecast(result["forecast"], observed_df)
   - update_diagnostics(result["diagnostics"], residuals)
5. Re-enable button, restore cursor, show completion dialog.

Provide hooks so UI controls (e.g., change 	_h or q) re-render charts using cached sensitivity results (no recompute unless requested).

---
## 4. Testing & Validation Checklist

- **Unit tests**: Mittag-Leffler evaluation vs SciPy references; monotonic derivative check.
- **Synthetic regression**: generate FK data with known parameters, verify estimation recovers within tolerance.
- **Sensitivity**: ensure indices sum consistent with total variance; compare with SALib sample runs.
- **UQ**: check sample mean/variance matches Laplace approximation; verify failure-time quantiles using simulated data.
- **Conformal**: on synthetic data, verify empirical coverage ≥ target.
- **UI**: manual smoke test for each tab; ensure toggles and controls update correctly.

---
## 5. Documentation

- Update README / docs with instructions for running fractional pipeline, configuring thresholds, interpreting bands.
- Provide screenshot examples of each UI tab.
- Describe output JSON schema for downstream reporting.

---
By following this blueprint sequentially, another engineer can implement the fractional FK model, complete analytics stack, and refreshed UI consistent with the manuscript’s requirements.
