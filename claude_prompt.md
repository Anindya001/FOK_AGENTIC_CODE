# Prompt for Claude
You are gpt-5-codex, an OpenAI coding assistant. You are working inside the `FOK_AGENTIC_CODE` repository, which hosts a PyQt5 desktop application for forecasting capacitor degradation, together with numerical backends for fractional-order modeling, Bayesian inference, and Sobol sensitivity analysis. All plotting uses Matplotlib, and Sobol computations live in `fractional_sensitivity.py`.

Your task is to replace the recent Sobol/UI overhaul with a clearer, single-pipeline design that keeps the mathematical improvements (pooled-variance Sobol estimators and degenerate-bootstrap filtering) while removing the confusing "dual Sobol" presentation introduced in commit `3d78888`. Treat the current code as a starting point that needs refinement rather than a ground truth.

## Objectives
1. **Unify Sobol result sourcing**
   * The sensitivity tab currently exposes a mode combo box and two stacked layouts. Instead, keep one canvas and allow the user to switch between "Forecast-backed" (Sobol indices baked into the last FK forecast result) and "Prior sweep" (fresh samples drawn from user-configured priors) without duplicating widgets. Consolidate UI wiring so both options reuse a single rendering path and legend helper.
   * Surface a prominent explanation that the forecast-backed mode anchors priors at the MAP fit, while the prior sweep ignores data entirely. The wording should make it obvious these are complementary views, not contradictory results.
   * Ensure the user can run a prior sweep without leaving the tab, and can reload forecast-backed indices only when the FK forecast has produced them. Remove redundant stacked pages, refresh buttons, and placeholder text left over from the previous iteration.

2. **Refine Sobol plotting UX**
   * Keep `_draw_sobol_panel` (or an equivalent helper) but adjust it to space grouped bars more cleanly, increase legibility of error bars, and ensure legends do not overlap annotations. All Sobol figures (both interactive and exported) must share the 10"×8" size, 0–564 h x-axis, and 30–100 µF y-axis conventions established for prediction plots.
   * Present confidence intervals in the legend with meaningful labels (e.g., “95% CI”) and greyed bands behind the bars if that improves readability; avoid duplicate legend entries.
   * Verify that baseline comparison exports and surrogate plots still apply `_style_prediction_axis` after your refactor.

3. **Preserve and expose backend stability improvements**
   * Retain the pooled-variance normalization and NaN bootstrap filtering in `fractional_sensitivity._sobol_indices`, but add docstrings or inline comments clarifying the rationale so future contributors understand the change.
   * Make sure Sobol runs invoked from the UI thread call into the updated API without duplicating evaluation work. If you refactor function signatures, update `fractional_prediction.py` and any other callers accordingly.

4. **Housekeeping**
   * Remove dead code paths, unused imports, and widget attributes that become obsolete after the consolidation.
   * Keep `compileall` sanity checks passing.
   * Update any user-facing copy or tooltips affected by the restructure to reflect the new workflow.

## Acceptance checklist
- A single sensitivity tab layout handles both forecast-backed and prior-sweep Sobol runs, reusing the same Matplotlib canvas and legend helper.
- Switching between modes updates titles, explanatory text, and enabled controls without flashing or rebuilding widgets.
- Sobol plots consistently use 10"×8" figures with axes fixed to 0–564 hours (x) and 30–100 µF (y), and legends clearly differentiate first-order, total-order, and confidence interval visuals.
- `fractional_sensitivity._sobol_indices` keeps pooled-variance math plus new explanatory comments, and no caller regresses to the older variance estimate.
- Running `python -m compileall app_ui.py fractional_sensitivity.py fractional_prediction.py` succeeds.

Follow the repository’s existing style (type hints, numpy usage, Qt signal wiring). Do not introduce new dependencies beyond what is already in `requirements.txt`.

# Justification
The current sensitivity tab duplicates widgets through a `QStackedWidget`, leaving redundant refresh buttons and creating user confusion about why two Sobol charts disagree; consolidating the UI eliminates that confusion while still supporting both forecast-backed and prior-driven analyses. `_draw_sobol_panel` already draws grouped bars, but its spacing and legend handling remain cramped, so refining the helper improves readability alongside the mandated 10"×8" / (0,30)→(564,100) viewport that other plots already follow. Finally, the pooled-variance estimator in `fractional_sensitivity.py` (lines ~60–120) is mathematically superior but undocumented; preserving it with inline rationale plus ensuring all callers use the stabilized API keeps the numerical benefits without reintroducing variance pathologies.
