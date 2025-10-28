# Corrected Mathematical Review

**Generated:** 2025-10-28 (Corrected)
**Previous Review:** MATHEMATICAL_REVIEW_REPORT.md (contains errors)

---

## Corrections to Previous Review

### Errors Acknowledged

1. **INCORRECT:** Claimed `_render_surrogate` method was missing
   - **REALITY:** Method exists at app_ui.py:1071 and is fully implemented
   - **IMPACT:** No crash condition exists for surrogate models

2. **INCORRECT:** Claimed baseline comparison variables were undefined
   - **REALITY:** Variables properly defined at app_ui.py:836-838 via `_safe_float()`
   - **IMPACT:** No NameError occurs in baseline comparison

3. **MISSED ISSUE:** Sobol bootstrap NaN row handling is problematic

---

## ACTUAL Issue Found: Sobol Bootstrap NaN Handling

### Location
`fractional_sensitivity.py:109-111`

### Current Code
```python
if var_b == 0:
    S_boot[b] = np.nan    # Sets ENTIRE row to NaN
    ST_boot[b] = np.nan   # Sets ENTIRE row to NaN
    continue
```

### Problem
When bootstrap samples have zero variance, entire rows are set to NaN but remain in the array. This means:
- `np.nanpercentile` computes percentiles ignoring these rows
- If many samples have zero variance, confidence intervals are based on too few valid samples
- No warning is given when this happens
- Could produce misleading confidence intervals

### Example Scenario
If 150 out of 200 bootstrap iterations produce zero variance:
- Only 50 valid samples contribute to confidence intervals
- 95% CI based on 50 samples is much less reliable than advertised
- User has no indication this happened

### Mathematical Impact
Bootstrap confidence intervals require:
```
n_bootstrap >= 200 (typical recommendation)
```

But if many are NaN:
```
n_effective < n_bootstrap
```

The effective sample size could be much smaller, making CIs unreliable.

### Recommended Fix

```python
def _sobol_indices(
    g_a: np.ndarray,
    g_b: np.ndarray,
    g_c: Sequence[np.ndarray],
    *,
    n_bootstrap: int = 200,
    random_state: int | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    # ... existing validation ...

    rng = np.random.default_rng(random_state)
    S_boot = []  # Use lists to accumulate only valid samples
    ST_boot = []

    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        g_a_b = g_a[idx]
        g_b_b = g_b[idx]
        g_c_b = [arr[idx] for arr in g_c]
        var_b = np.var(g_a_b, ddof=1)

        if var_b == 0:
            continue  # Skip this iteration entirely

        # Compute indices for this valid bootstrap sample
        S_b = []
        ST_b = []
        for j, arr_b in enumerate(g_c_b):
            S_b.append(np.mean(g_b_b * (arr_b - g_a_b)) / var_b)
            ST_b.append(0.5 * np.mean((g_a_b - arr_b) ** 2) / var_b)

        S_boot.append(S_b)
        ST_boot.append(ST_b)

    # Check we have enough valid samples
    n_valid = len(S_boot)
    if n_valid < 50:  # Minimum threshold for reliable CI
        raise RuntimeError(
            f"Only {n_valid} out of {n_bootstrap} bootstrap samples were valid "
            f"(zero variance). Confidence intervals are unreliable. "
            f"Consider increasing n_samples or adjusting priors."
        )

    # Convert to arrays and compute percentiles
    S_boot = np.array(S_boot)
    ST_boot = np.array(ST_boot)

    lower = 2.5
    upper = 97.5
    S_ci = np.percentile(S_boot, [lower, upper], axis=0)  # No need for nanpercentile
    ST_ci = np.percentile(ST_boot, [lower, upper], axis=0)

    return {
        "main": np.array(S),
        "total": np.array(ST),
        "main_ci": S_ci,
        "total_ci": ST_ci,
        "n_bootstrap_valid": n_valid,  # Return for diagnostics
    }
```

### Benefits of Fix
1. **Explicit validation:** Raises error if too few valid samples
2. **No silent failures:** User knows when CIs are unreliable
3. **Cleaner logic:** No NaN values in percentile calculation
4. **Diagnostic info:** Returns number of valid bootstrap samples

### Testing
```python
def test_sobol_bootstrap_validation():
    """Test that insufficient valid bootstrap samples raises error."""
    # Create priors that will produce many zero-variance samples
    priors = {
        'C0': LogUniformPrior(99.9, 100.0),  # Very narrow range
        'k': LogUniformPrior(0.001, 0.001001),
        'alpha': BetaPrior(100, 1),  # Concentrated near 1
        'f_inf': BetaPrior(100, 1)
    }

    qoi = qoi_capacitance(t_h=100.0)

    # Should raise RuntimeError due to insufficient valid samples
    with pytest.raises(RuntimeError, match="Only .* bootstrap samples were valid"):
        sobol_analysis(priors, qoi, n_samples=100, n_bootstrap=200)
```

---

## Summary of Corrected Findings

### ✅ What's Actually Correct
1. **Sobol sampling scheme** - Saltelli method properly implemented
2. **Index estimators** - Formulas are mathematically correct
3. **MCMC implementation** - Log-posterior, M-H algorithm all correct
4. **Fractional model** - All mathematical formulations correct
5. **Mittag-Leffler functions** - Proper implementations with fallbacks

### ⚠️ What Needs Fixing
1. **Sobol bootstrap NaN handling** - Needs explicit validation (Lines 109-120)
   - **Priority:** Medium (affects CI reliability in edge cases)
   - **Estimated effort:** 1-2 hours

### ✅ What I Incorrectly Claimed Was Broken
1. ~~`_render_surrogate` method~~ - Exists and works fine
2. ~~Baseline comparison variables~~ - Properly defined
3. ~~Critical UI bugs~~ - Don't actually exist

---

## Revised Priority Assessment

### **Actual Issues:**

**Medium Priority:**
1. **Sobol bootstrap validation** (fractional_sensitivity.py:109-120)
   - Add check for number of valid bootstrap samples
   - Raise error if insufficient valid samples for reliable CIs
   - Estimated effort: 1-2 hours

**Low Priority (Optional Optimizations):**
2. Remove constant terms from MCMC log-posterior (performance only)
3. Add prior normalization constants (for model comparison)
4. Enhanced numerical stability for extreme parameters

### **Non-Issues (My Errors):**
- ~~Missing `_render_surrogate` method~~ ✗
- ~~Undefined baseline comparison variables~~ ✗
- ~~Critical UI crashes~~ ✗

---

## Apology and Lesson Learned

I apologize for the inaccurate review. I should have:
1. More carefully verified claimed "missing" methods with grep
2. Read more context around supposedly undefined variables
3. More carefully analyzed the NaN handling in bootstrap code
4. Not jumped to conclusions about "critical bugs"

The codebase is actually in much better shape than my initial review suggested. The only real issue is the bootstrap validation, which is a relatively minor enhancement rather than a critical bug.

---

**Report Status:** Corrected
**Actual Critical Issues:** 0
**Actual Medium Issues:** 1 (Sobol bootstrap validation)
**Falsely Claimed Issues:** 2 (UI bugs that don't exist)
