# Plotting Corrections Required

**Generated:** 2025-10-28
**Issue:** Plot axis limits don't match specifications

---

## Required Specifications

1. **Figure size:** 10" × 8" ✅ CORRECT
2. **X-axis (Time):** 0 to 564 hours ❌ NOT SET
3. **Y-axis (Capacitance):** 30 to 100 µF ❌ INVERTED
4. **Origin:** (0, 30) ❌ WRONG (currently inverted Y-axis)

---

## Issues Found

### Issue 1: Y-Axis is Inverted

**Locations:**
- Line 785: `ax.set_ylim(100.0, 30.0)` in baseline comparison plotting
- Line 1717: `ax.set_ylim(100.0, 30.0)` in main forecast plot rendering

**Current behavior:**
```
Y-axis: 100 (bottom) → 30 (top)
Origin: (0, 100) - WRONG
```

**Required behavior:**
```
Y-axis: 30 (bottom) → 100 (top)
Origin: (0, 30) - CORRECT
```

**Impact:** All forecast plots display upside-down with capacitance decreasing upward instead of downward.

---

### Issue 2: X-Axis Limits Not Set

**Locations:**
- Line 784-785: Baseline comparison plotting
- Line 1578+: Main forecast plot rendering (`_update_forecast_plot`)
- Line 1071+: Surrogate model rendering (`_render_surrogate`)

**Current behavior:**
```
X-axis: Auto-scaled based on data
```

**Required behavior:**
```
X-axis: 0 to 564 hours (fixed)
```

**Impact:** Plots zoom to fit data, making comparisons across different datasets inconsistent.

---

### Issue 3: Surrogate Plot Has No Axis Limits

**Location:** Line 1071-1141 (`_render_surrogate` method)

**Current behavior:**
- No `set_xlim` call
- No `set_ylim` call
- Axes auto-scale to data

**Required behavior:**
- X-axis: 0 to 564 hours
- Y-axis: 30 to 100 µF

---

## Corrections Needed

### Correction 1: Fix Y-Axis Orientation

**File:** `app_ui.py`

**Line 785:**
```python
# BEFORE
ax.set_ylim(100.0, 30.0)

# AFTER
ax.set_ylim(30.0, 100.0)
```

**Line 1717:**
```python
# BEFORE
ax.set_ylim(100.0, 30.0)

# AFTER
ax.set_ylim(30.0, 100.0)
```

---

### Correction 2: Add X-Axis Limits

**File:** `app_ui.py`

**Line 785 (add after):**
```python
ax.set_ylim(30.0, 100.0)
ax.set_xlim(0, 564)  # Add this line
```

**Line 1717 (add after):**
```python
ax.set_ylim(30.0, 100.0)
ax.set_xlim(0, 564)  # Add this line
```

---

### Correction 3: Add Axis Limits to Surrogate Plot

**File:** `app_ui.py`

**Line 1131 (add after):**
```python
ax.set_ylabel(self._target_column or "Response")
ax.set_xlim(0, 564)  # Add this line
ax.set_ylim(30.0, 100.0)  # Add this line
ax.grid(True, alpha=0.25)
```

---

## Implementation Summary

### Files to Modify
- `app_ui.py`

### Lines to Change
1. **Line 785:** Change `ax.set_ylim(100.0, 30.0)` → `ax.set_ylim(30.0, 100.0)` and add `ax.set_xlim(0, 564)`
2. **Line 1132:** Add `ax.set_xlim(0, 564)` and `ax.set_ylim(30.0, 100.0)`
3. **Line 1717:** Change `ax.set_ylim(100.0, 30.0)` → `ax.set_ylim(30.0, 100.0)` and add `ax.set_xlim(0, 564)`

### Total Changes
- 3 functions affected
- 6 lines to add/modify

---

## Testing Plan

After corrections:

1. **Visual verification:**
   - Generate forecast plot
   - Verify Y-axis: 30 at bottom, 100 at top
   - Verify X-axis: 0 at left, 564 at right
   - Verify origin is at bottom-left corner (0, 30)

2. **Test cases:**
   - FK model forecast
   - Classical model (surrogate)
   - KWW model (surrogate)
   - Baseline comparison plots

3. **Expected result:**
   - All plots show degradation going downward (capacitance decreasing over time)
   - All plots have consistent axis ranges
   - Origin at (0, 30) in bottom-left corner

---

## Estimated Effort
**Time:** 30 minutes
**Priority:** High (visualization correctness)
