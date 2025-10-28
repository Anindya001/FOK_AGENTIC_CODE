# Sobol Analysis Dual-Mode Review & Refactoring Plan

**Generated:** 2025-10-28
**Issue:** Evaluate if "data-backed" and "prior-driven" Sobol modes are contradictory

---

## Executive Summary

**You're right to question this!** The current "data-backed" Sobol mode is **misleading** and not a true posterior-based sensitivity analysis. Here's what's actually happening:

### Current Implementation

| Mode | What it Actually Does | What Users Think It Does |
|------|----------------------|-------------------------|
| **Prior-driven** ✅ | Samples from user-defined priors | Parameter space exploration |
| **"Data-backed"** ⚠️ | Samples from **priors centered on MLE** | Posterior-based sensitivity |

**The problem:** "Data-backed" is NOT sampling from the posterior distribution—it's using the fitted values as prior centers with arbitrary ranges.

---

## Deep Dive: What "Data-Backed" Actually Does

### Location: `fractional_core.py:112-119`

```python
def _build_priors(fit: FKFitResult) -> dict[str, object]:
    theta = fit.params
    return {
        "C0": LogNormalPrior(mean=np.log(theta.C0), sigma=0.2),  # ← Fixed σ=0.2
        "k": LogUniformPrior(low=max(theta.k / 5.0, 1e-8), high=theta.k * 5.0),  # ± 1 order of mag
        "alpha": BetaPrior(a=2 + theta.alpha * 5, b=2 + (1 - theta.alpha) * 5),
        "f_inf": BetaPrior(a=2 + theta.f_inf * 5, b=2 + (1 - theta.f_inf) * 5),
    }
```

### What This Does:
1. Takes MLE point estimates from fit
2. Creates **arbitrary "shrunk" priors** around those estimates
3. Runs Sobol on these tightened priors

### What It Does NOT Do:
- ❌ Use actual posterior uncertainty from MCMC
- ❌ Use covariance matrix from Laplace approximation
- ❌ Reflect true data-informed uncertainty
- ❌ Account for parameter correlations

### Example:
If data shows `k = 0.01 ± 0.002` (tight), but the code uses:
```python
"k": LogUniformPrior(low=0.002, high=0.05)  # ± order of magnitude
```

This is **not** the posterior uncertainty—it's an arbitrary range!

---

## Proper Approaches to Sensitivity Analysis

### Approach 1: Prior-Driven (Current, Correct ✅)
**Use case:** Before seeing data, explore parameter space

**Method:**
- User defines prior ranges for all parameters
- Sobol samples from these priors
- Answers: "Which parameters matter most in general?"

**Example:**
```python
priors = {
    "C0": LogUniformPrior(low=50, high=100),
    "k": LogUniformPrior(low=1e-4, high=1e-2),
    "alpha": BetaPrior(a=2, b=2),  # Uniform on [0,1]
    "f_inf": BetaPrior(a=2, b=2),
}
```

**When useful:** Design phase, no data yet

---

### Approach 2: MLE-Centered (Current "Data-Backed", Misleading ⚠️)
**Use case:** Unclear—not rigorous posterior, not pure prior

**Method:**
- Center priors on MLE estimates
- Use arbitrary spread (e.g., ±5× for k)
- Sobol on these "shrunk" priors

**Problems:**
- Not theoretically justified
- Arbitrary uncertainty ranges
- Ignores actual posterior uncertainty
- Misleading name ("data-backed" implies posterior)

**Recommendation:** **Remove or rename clearly**

---

### Approach 3: True Posterior-Based (Not Implemented ❌)
**Use case:** After fitting, understand which parameters still matter

**Method:**
- Use actual MCMC or Laplace posterior samples
- Each sample is a valid parameter set from posterior
- Compute Sobol on posterior distribution

**Proper implementation:**
```python
def posterior_sobol_analysis(
    posterior_samples: List[FKParams],  # From MCMC
    qoi: Callable,
    n_bootstrap: int = 200
) -> dict:
    """
    True posterior-based Sobol using MCMC samples.

    This respects:
    - Actual posterior uncertainty
    - Parameter correlations
    - Data-informed constraints
    """
    # Build empirical distribution from posterior
    n_posterior = len(posterior_samples)

    # Sobol requires sampling, so we resample from empirical posterior
    def sample_from_posterior(rng, size):
        indices = rng.choice(n_posterior, size=size, replace=True)
        return [posterior_samples[i] for i in indices]

    # Run Saltelli scheme on posterior distribution
    # ... (implementation details)
```

**When useful:** After fitting to data, understand remaining uncertainty

---

## Comparison: Prior vs Posterior Sobol

### Example Scenario:
Fit model to capacitor data, obtain:
- `C0 = 95 ± 1` (very tight—data constrained)
- `k = 0.005 ± 0.004` (wide—poorly constrained)
- `alpha = 0.75 ± 0.15` (moderate)
- `f_inf = 0.3 ± 0.1` (moderate)

### Prior Sobol (before data):
```
All parameters have large sensitivity:
S_C0 = 0.30
S_k = 0.35
S_alpha = 0.25
S_f_inf = 0.10
```

### Posterior Sobol (after data):
```
C0 nearly constrained, k dominates:
S_C0 = 0.05   ← Data constrained this!
S_k = 0.60    ← Still uncertain, drives variability
S_alpha = 0.25
S_f_inf = 0.10
```

**Insight:** After seeing data, C0 matters less (well-constrained), k matters more (still uncertain).

**Current "data-backed" mode CANNOT capture this** because it doesn't use actual posterior.

---

## Recommendations

### Option 1: Keep Prior-Driven Only (Simplest ✅)

**Rationale:**
- Prior Sobol is theoretically sound
- Current "data-backed" is not rigorous
- Posterior Sobol requires significant implementation work

**Implementation:**
1. Remove "Data-backed" mode from UI
2. Keep only "Prior-driven" mode
3. Update documentation

**Effort:** 30 minutes

**Pros:**
- Clean, no misleading features
- One clear mode with clear interpretation

**Cons:**
- Loses ability to do post-data sensitivity (until proper posterior Sobol implemented)

---

### Option 2: Rename "Data-Backed" to "MLE-Centered" (Quick Fix)

**Rationale:**
- Acknowledge what it actually does
- Don't mislead users about posterior

**Implementation:**
```python
self.sens_mode_combo.addItem("MLE-centered (priors around fitted values)", userData="mle")
self.sens_mode_combo.addItem("Prior-driven (explore parameter space)", userData="prior")
```

**Effort:** 15 minutes

**Pros:**
- Honest about what it does
- Quick fix

**Cons:**
- Still not rigorous
- Users may still misunderstand

---

### Option 3: Implement True Posterior Sobol (Rigorous, More Work)

**Rationale:**
- Provide proper posterior-based sensitivity
- Scientifically rigorous

**Implementation:**
1. Add `posterior_sobol_analysis()` function
2. Sample from MCMC/Laplace posterior
3. Run Saltelli scheme on empirical posterior
4. Handle parameter correlations properly

**Effort:** 4-6 hours

**Pros:**
- Scientifically correct
- Valuable feature for post-data analysis

**Cons:**
- Significant implementation time
- Requires careful validation

**Detailed approach:**
```python
def posterior_sobol_from_mcmc(
    posterior_samples: List[FKParams],
    sigma_samples: np.ndarray,
    qoi: Callable,
    n_sobol_samples: int = 2048,
    n_bootstrap: int = 200,
    random_state: Optional[int] = None,
) -> dict:
    """
    Compute Sobol indices on posterior distribution.

    Uses empirical posterior from MCMC as the distribution to sample from.
    """
    rng = np.random.default_rng(random_state)
    n_posterior = len(posterior_samples)

    # Convert posterior samples to matrix
    param_matrix = np.array([
        [s.C0, s.k, s.alpha, s.f_inf]
        for s in posterior_samples
    ])

    # Build empirical CDF for resampling
    def sample_posterior_matrix(n: int) -> np.ndarray:
        indices = rng.choice(n_posterior, size=n, replace=True)
        return param_matrix[indices]

    # Saltelli sampling on posterior
    A = sample_posterior_matrix(n_sobol_samples)
    B = sample_posterior_matrix(n_sobol_samples)

    # Evaluate QoI
    def eval_qoi(matrix: np.ndarray) -> np.ndarray:
        outputs = np.empty(matrix.shape[0])
        for i, row in enumerate(matrix):
            theta = FKParams(C0=row[0], k=row[1], alpha=row[2], f_inf=row[3])
            try:
                outputs[i] = qoi(theta)
            except:
                outputs[i] = np.nan
        return outputs

    g_a = eval_qoi(A)
    g_b = eval_qoi(B)

    # C matrices for each parameter
    g_c = []
    for j in range(4):
        C = B.copy()
        C[:, j] = A[:, j]
        g_c.append(eval_qoi(C))

    # Compute indices (using existing _sobol_indices function)
    return _sobol_indices(g_a, g_b, g_c, n_bootstrap=n_bootstrap, random_state=random_state)
```

---

## My Recommendation: **Option 1 (Remove "Data-Backed")**

### Why:
1. **Theoretical soundness:** Prior Sobol is well-defined and useful
2. **Avoid confusion:** Current "data-backed" misleads users
3. **Simplicity:** One clear mode, one clear interpretation
4. **Future-proof:** Can add proper posterior Sobol later (Option 3) if needed

### What to do:
1. Remove "Data-backed" mode from sensitivity tab
2. Keep only "Prior-driven" mode
3. Update tooltip: "Explore parameter space by sampling from prior distributions"
4. Remove mode switcher combo box (single mode doesn't need switching)
5. Update documentation

---

## Refactoring for Plot Helpers (Your Original Question)

Now let's address your excellent idea about reusable helper functions:

### Proposed Helper Functions

```python
def _format_prediction_axes(ax: plt.Axes,
                            time_label: str = "Time",
                            cap_label: str = "Capacitance",
                            add_units: bool = True) -> None:
    """Apply consistent axis formatting to all prediction plots."""
    ax.set_xlim(0, 564)
    ax.set_ylim(30.0, 100.0)
    ax.set_xlabel(time_label, fontsize=11)
    ylabel = f"{cap_label} (uF)" if add_units and "cap" in cap_label.lower() else cap_label
    ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)


def _add_legend_scatter(handle, label: str, legend_entries: list, seen: set) -> None:
    """Add scatter plot to legend if not duplicate."""
    if handle is not None and label not in seen:
        legend_entries.append((handle, label))
        seen.add(label)


def _add_legend_line(handle, label: str, legend_entries: list, seen: set) -> None:
    """Add line plot to legend if not duplicate."""
    if handle is not None and label not in seen:
        legend_entries.append((handle, label))
        seen.add(label)


def _add_legend_band(handle, label: str, legend_entries: list, seen: set) -> None:
    """Add uncertainty band to legend if not duplicate."""
    if handle is not None and label not in seen:
        legend_entries.append((handle, label))
        seen.add(label)


def _finalize_legend(ax: plt.Axes, legend_entries: list, ncol: int = 2) -> None:
    """Create final legend with consistent styling."""
    if legend_entries:
        handles, labels = zip(*legend_entries)
        ax.legend(
            handles, labels,
            loc="upper right",
            frameon=True,
            framealpha=0.85,
            fontsize=11,
            ncol=ncol,
            columnspacing=1.2
        )


def _add_metrics_box(ax: plt.Axes, metrics: dict, x: float = 0.72, y: float = 0.5) -> None:
    """Add metrics text box with consistent styling."""
    lines = []
    for key, val in metrics.items():
        if val is not None and np.isfinite(val):
            lines.append(f"{key}: {val:.3f}")

    if lines:
        ax.text(
            x, y,
            "\n".join(lines),
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.75, edgecolor="#888888")
        )
```

### Benefits:
1. **Consistency:** All plots use same formatting
2. **Maintainability:** Change once, apply everywhere
3. **Clarity:** Clear separation of concerns
4. **Testability:** Helper functions easy to unit test

---

## Implementation Plan

### Phase A: Remove Confusing "Data-Backed" Mode (30 min)

**Files to modify:** `app_ui.py`

1. Remove mode combo box (lines 367-372)
2. Remove mode switcher logic (lines 699-711)
3. Remove "Data-backed" UI elements
4. Update tooltips to indicate "Prior exploration"
5. Simplify sensitivity tab to single mode

---

### Phase B: Add Plot Helper Functions (1 hour)

**Files to modify:** `app_ui.py`

1. Add helper functions at module level
2. Refactor `_update_forecast_plot()` to use helpers
3. Refactor `_render_surrogate()` to use helpers
4. Refactor baseline comparison plotting to use helpers
5. Test all plot types

---

### Phase C: Relocate Metrics Box (30 min)

**Current issue:** Metrics box at (0.72, 0.5) may overlap legend

**Solution:**
```python
def _smart_metrics_position(legend_loc: str) -> tuple[float, float, str]:
    """Return (x, y, ha) for metrics box that avoids legend."""
    if legend_loc == "upper right":
        return (0.02, 0.02, "left")  # Lower left
    elif legend_loc == "upper left":
        return (0.98, 0.02, "right")  # Lower right
    else:
        return (0.72, 0.5, "left")  # Center right (original)
```

---

## Total Effort Estimate

| Phase | Task | Time |
|-------|------|------|
| A | Remove "data-backed" mode | 30 min |
| B | Add plot helper functions | 1 hour |
| C | Smart metrics positioning | 30 min |
| **Total** | **Complete refactoring** | **2 hours** |

---

## Answer to Your Questions

### 1. "Should Sobol be data or prior only?"

**My answer:** **Prior-driven only** (remove "data-backed")

**Reasoning:**
- Current "data-backed" is not rigorous (doesn't use true posterior)
- Prior Sobol is theoretically sound and useful for design
- True posterior Sobol would require significant work to implement correctly

### 2. "Is dual Sobol contradictory?"

**Yes, as currently implemented:**
- "Prior-driven" ✅ makes sense
- "Data-backed" ❌ is misleading (not true posterior, just shrunk priors)
- Having both creates confusion about what each means

**Solution:** Remove "data-backed", keep prior-driven only

### 3. "What about the helper functions?"

**Excellent idea!** Implementing:
- `_format_prediction_axes()`
- `_add_legend_*()` functions
- `_finalize_legend()`
- `_add_metrics_box()`
- `_smart_metrics_position()`

Will significantly improve code quality and consistency.

---

## Final Recommendation

**Priority 1:** Remove "data-backed" Sobol mode (30 min)
**Priority 2:** Implement plot helper functions (1 hour)
**Priority 3:** Smart metrics positioning (30 min)

**Total time:** 2 hours for significant improvement

**Future work:** Implement true posterior Sobol if needed (4-6 hours)

---

**Ready to proceed?** I can implement all phases if you approve the approach.
