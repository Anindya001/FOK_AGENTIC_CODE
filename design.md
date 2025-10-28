# Sensitivity & Forecast UI Design

## Analysis Modes
- Combo box `self.mode_combo` allows the user to pick between `Forecast & UQ` and `Sensitivity Study`.
- When `Forecast & UQ` is active:
  - Training fraction, confidence, model combo, and MCMC controls remain interactable.
  - FK/classical/KWW pipelines execute via `AnalysisWorker`.
  - FK-specific options (run-sensitivity checkbox, horizons, thresholds) are exposed.
- When `Sensitivity Study` is active:
  - Forecast controls are disabled to avoid accidental reuse.
  - The Sensitivity Study tab becomes active with its own run/export buttons.
  - Progress bar/status label reset via `_mode_changed` when idle.

## Threading & Progress
- Shared `QThreadPool` (max 2 threads) executes long-running jobs (forecasts, Sobol runs, baseline comparisons).
- `WorkerSignals` funnel success/error back to the main thread.
- `_start_progress` / `_stop_progress` drive an indeterminate `QProgressBar` and status messaging.
- All long jobs disable their initiating controls until callbacks complete.

## Forecast Workflow (`run_analysis`)
1. Guard against concurrent workers; reroute to Sensitivity Study when that mode is selected.
2. Preprocess dataset (drop NaNs, order) and cache arrays in `_pending_*` fields.
3. Branch by `self._current_model`:
   - FK: build `FractionalConfig`, queue `_run_fk_task` with full config, thresholds, horizons.
   - Classical/KWW: queue `_run_classical_task` / `_run_kww_task` with numpy arrays and train ratio.
4. Worker completion:
   - FK handled by `_on_analysis_success`, updating overview tables/plots and enabling exports.
   - Classical/KWW handled by `_on_surrogate_success`, rendering surrogate plots and forecasts.
5. `_on_analysis_error` restores UI state, clears forecasts, and shows the failure message.

## Sensitivity Study Workflow (`_run_sensitivity_study`)
1. Collect FK prior parameters:
   - C₀, k log-normal means/sigmas.
   - α, f∞ beta means + concentration to shape priors.
2. Parse mission horizons, failure thresholds, sample/boot counts.
3. Worker `_run_sensitivity_task` builds priors using `fractional_sensitivity` helpers and computes Sobol indices for:
   - Capacitance Y(h).
   - Deficit Δ(h).
   - Failure time T(q).
4. Results include raw arrays and a log string summarizing FK equation, priors, main/total effects.
5. `_on_sensitivity_study_success` updates the log pane, enables plot export, and redraws bar charts with CIs.
6. Errors return via `_on_sensitivity_study_error` with status reset.

## UI Layout Highlights
- Forecast tab: overview tables, metrics text, forecast figure, export buttons.
- Sensitivity tab (data-backed): QoI combo and Matplotlib canvas for FK run_sensitivity path.
- Sensitivity Study tab (data-free): prior form layout, run/save buttons, log output, stacked Sobol plots.
- Progress bar/status row replaces previous status-only label for uniform feedback.

## Responsiveness Safeguards
- All analyses run asynchronously; buttons disabled during execution.
- Progress bar visible throughout the job; hidden when idle.
- Mode switch resets progress and status when no worker is active.

## File Touchpoints
- `app_ui.py`: orchestrates UI, workers, plotting, progress management.
- `fractional_sensitivity.py`: supplies Sobol sampling utilities reused by the study tab.
- `CHANGELOG.md`: records UX and sensitivity updates.
