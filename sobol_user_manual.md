Sobol Sensitivity Quick Manual
==============================

Overview
--------
- The app exposes **one shared Sobol configuration** (horizons, failure thresholds, Sobol sample count, bootstrap draws).
- Two workflows consume the same settings:
  - **Data-backed Sobol**: runs automatically after an FK forecast when “Compute Sobol indices with FK forecast” is ticked.
  - **Prior-driven study**: explores the model using only priors and the shared Sobol configuration (no dataset required).

Setting Up
----------
1. In the main window, check `Compute Sobol indices with FK forecast` if you want Sobol bars every time the FK model runs. Leave unchecked to skip during forecasting.
2. Click `Edit settings…` to open the Sensitivity tab.
3. Update the **Sobol configuration** group:
   - `Horizons`: positive times for QoIs, comma separated (e.g. `200, 400`).
   - `Failure thresholds`: ratios in (0, 1), comma separated (e.g. `0.8, 0.7`).
   - `Sobol samples`: total Monte Carlo draws (2048 recommended baseline; increase if indices jitter).
   - `Bootstrap draws`: controls CI width (≥200 recommended).
   Any change updates the summary label beside the checkbox.

Data-backed Sobol (uses loaded data)
------------------------------------
1. Load a dataset and choose time/target columns.
2. Ensure `Compute Sobol indices with FK forecast` is checked.
3. Run analysis with the FK model selected.
4. When the forecast finishes, the **Sensitivity** tab displays stacked bars with first-order and total Sobol indices plus confidence intervals. Export via `Save sensitivity plot…`.

Prior-driven Sobol (no data required)
------------------------------------
1. Open the Sensitivity tab with `Edit settings…`.
2. Adjust the **Fractional prior configuration** spins (means and dispersions). These anchor the FK parameters.
3. Verify the shared Sobol configuration above matches the scenarios you want.
4. Click `Run sensitivity study`. Results populate the plot and the text log. Save via `Save sensitivity study plot…`.

Interpreting Results
--------------------
- **First-order S** quantifies individual parameter influence.
- **Total S** includes interactions.
- Bootstrapped 95 % bands appear if bootstrap draws > 0.
- The summary log highlights the strongest contributors per QoI.

Troubleshooting: “No finite samples available for Sobol analysis”
-----------------------------------------------------------------
- The solver attempted multiple resamples but QoI evaluations returned non-finite values.
- Typical causes:
  - Thresholds too aggressive (e.g. very high q) so time-to-threshold diverges.
  - Horizons excessively large, causing numerical underflow in Eα(-k tᵅ).
  - Priors too diffuse, allowing k ≈ 0 or α at the limits.
- Fixes:
  1. Tighten priors (smaller log-sigmas or narrower log-uniform ranges).
  2. Reduce horizons to realistic mission windows.
  3. Use thresholds within 0.5–0.95.
  4. If needed, increase Sobol samples gradually (e.g. 4096) after narrowing the domain.

Best Practices
--------------
- Start with the default settings; inspect the log for extreme parameter values.
- Increase Sobol samples only after ensuring QoIs remain finite.
- Keep bootstrap draws moderate (200–500) for smooth CIs without long runtimes.
