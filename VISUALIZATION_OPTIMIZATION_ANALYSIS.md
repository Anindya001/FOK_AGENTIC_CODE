# Deep Analysis: Plot Visualization & Legend Optimization

**Generated:** 2025-10-28
**Focus:** Image size, legends, uncertainty bands, visual clarity

---

## Executive Summary

After deep analysis of the plotting code, I've identified several critical issues:

1. **Legend Overload:** Up to 10 legend entries can appear simultaneously
2. **Color Confusion:** Similar blue-gray tones make bands hard to distinguish
3. **Visual Clutter:** Multiple overlapping uncertainty bands create confusion
4. **Inconsistent Styling:** Different line styles/alphas not well-coordinated
5. **Legend Positioning:** Fixed "upper right" may overlap with data/metrics box
6. **Missing Visual Hierarchy:** No clear priority among bands

---

## Current State Analysis

### What Can Appear in a Single Plot

**Maximum possible legend entries: 10**

1. **Training data** (scatter) - Gray-blue `#6b7c8c`
2. **Observed forecast** (scatter) - Brown `#a0765b`
3. **Forecast mean** (line) - Dark blue `#2f4858`
4. **Bias-corrected mean** (line) - Blue `#496a7a`
5. **Forecast boundary** (vline) - Gray `#8a857d`
6. **Epistemic band** (fill) - Light blue `#8ea8ba` @ 18% alpha
7. **Total band** (fill) - Very light blue `#b7c6d0` @ 18% alpha
8. **Conformal interval** (fill) - Tan `#d9cbb0` @ 22% alpha
9. **Hybrid interval** (fill) - Bronze `#b38b44` @ 18% alpha
10. **Metrics box** (text overlay) - Not in legend but competes for space

---

## Critical Issues Found

### Issue 1: Color Scheme Problems

#### **Problem: Poor Distinguishability**

Current colors for uncertainty bands:
```python
EPISTEMIC_COLOR = "#8ea8ba"  # Light blue-gray
TOTAL_COLOR = "#b7c6d0"      # Very light blue-gray  ← TOO SIMILAR
CONFORMAL_COLOR = "#d9cbb0"  # Tan/beige
HYBRID_COLOR = "#b38b44"     # Bronze/tan
```

**Visual conflict:**
- EPISTEMIC and TOTAL are both pale blue-grays
- At 18% alpha, they become nearly indistinguishable
- CONFORMAL and HYBRID both have tan/brown tones
- User cannot easily tell which band is which

**Color blindness impact:**
- Deuteranopia (green-blind): Blue tones become even more similar
- Protanopia (red-blind): Brown/tan colors hard to distinguish

#### **Problem: Insufficient Contrast**

All bands use low alpha (0.18-0.22), making them fade into background.
When 4 bands overlap:
- Visual noise increases
- Individual bands become invisible
- Purpose of each band is lost

---

### Issue 2: Legend Clutter & Readability

#### **Current Legend Configuration**
```python
ax.legend(handles, labels,
          loc="upper right",      # Fixed position
          frameon=True,
          framealpha=0.85,
          fontsize=9,             # Small font
          ncol=2,                 # 2 columns
          columnspacing=1.2)
```

#### **Problems:**

1. **Too many entries:** With 10 possible items, legend becomes:
   - 5 rows × 2 columns = Large block
   - Obscures upper-right data region
   - Competes with metrics text box at (0.72, 0.5)

2. **Font size (9pt):** At 300 DPI export, becomes tiny and hard to read

3. **Fixed positioning:**
   - `loc="upper right"` always, even when data is there
   - No automatic repositioning
   - May overlap with degradation curve's start

4. **Redundant information:**
   - "Epistemic band (model)" vs "Total band (model+noise)" - overly verbose
   - "Conformal interval (guaranteed)" - parenthetical adds little value
   - "Hybrid interval (union)" - technical jargon unclear to users

---

### Issue 3: Uncertainty Band Redundancy

#### **Conceptual Overlap**

The code displays up to 4 overlapping uncertainty bands:

1. **Epistemic (parameter uncertainty)**
   - From MCMC/Laplace posterior samples
   - Captures model parameter uncertainty only

2. **Total (parameter + noise)**
   - Epistemic + aleatoric (observation noise)
   - Superset of epistemic

3. **Conformal (distribution-free)**
   - Calibrated to achieve guaranteed coverage
   - Completely different methodology

4. **Hybrid (union of conformal + total)**
   - Max(conformal, total)
   - Takes widest envelope

#### **The Problem:**

**If all 4 are shown simultaneously:**
- Total band ⊇ Epistemic band (always wider)
- Hybrid band ⊇ Total band (always wider)
- Hybrid band ⊇ Conformal band (always wider)

**Result:** Showing all 4 creates nested Russian dolls:
```
Epistemic ⊂ Total ⊂ Hybrid
         Conformal ⊂ Hybrid
```

**User sees:**
- 4 overlapping fills
- Hard to understand which is which
- Unclear which to trust
- Visual clutter with minimal added insight

**Better approach:** Show maximum of 2-3 carefully chosen bands

---

### Issue 4: Baseline Comparison Plots

**Current:** Separate plots saved for FK, Classical, KWW models

**Legend entries (up to 5):**
1. Training data
2. Observed forecast
3. Model mean
4. Forecast boundary
5. Conformal interval (guaranteed)

**Problems:**
- Uses same legend config (fontsize=9, ncol=2)
- Only conformal interval shown (good!)
- But legend still takes significant space
- Metrics box at fixed (0.72, 0.5) may overlap

---

## Visual Hierarchy Issues

### Current Approach: Democratic (all equal)

- All bands use similar alpha (0.18-0.22)
- All have fill + boundary lines
- No visual priority

### What Users Actually Need:

**Priority 1: The forecast**
- Mean prediction (most important)
- Should be boldest, most visible

**Priority 2: Recommended uncertainty**
- ONE primary uncertainty band
- Should be clear and prominent

**Priority 3: Observed data**
- Training and forecast observations
- Important for context

**Priority 4: Auxiliary bands (optional)**
- Alternative uncertainty measures
- Should be subtle, non-intrusive

---

## Recommendations

### Recommendation 1: Redesign Color Palette

**Current problems:**
- Too many similar blue-grays
- Low contrast
- Poor accessibility

**Proposed new palette:**
```python
# Data points
TRAIN_COLOR = "#455A64"          # Blue-gray (unchanged, good)
FORECAST_OBS_COLOR = "#D84315"   # Deep orange (more contrast vs blue)

# Primary elements
FORECAST_MEAN_COLOR = "#1976D2"  # Strong blue (bold, clear)
FORECAST_BOUNDARY = "#78909C"     # Medium gray (subtle)

# Uncertainty bands - Distinct hues
EPISTEMIC_COLOR = "#42A5F5"      # Bright blue (clear primary)
TOTAL_COLOR = "#7E57C2"          # Purple (distinct from blue)
CONFORMAL_COLOR = "#FFA726"      # Orange (warm contrast)
HYBRID_COLOR = "#66BB6A"         # Green (distinct from all)

# Bias-corrected (optional)
BIAS_MEAN_COLOR = "#0277BD"      # Darker blue (secondary)
```

**Benefits:**
- Each band has distinct hue (blue, purple, orange, green)
- Color-blind safe (use ColorBrewer qualitative schemes)
- High contrast even at low alpha
- Clear visual hierarchy

---

### Recommendation 2: Simplify Default Band Selection

**Current:** All 4 checkboxes checked by default
```python
self.epistemic_cb.setChecked(True)   # ← Too many!
self.total_cb.setChecked(True)       # ← Too many!
self.conformal_cb.setChecked(True)   # ← Too many!
self.hybrid_cb.setChecked(True)      # ← Too many!
```

**Proposed:** Show 1-2 bands by default
```python
self.epistemic_cb.setChecked(False)  # Advanced users only
self.total_cb.setChecked(False)      # Redundant with hybrid
self.conformal_cb.setChecked(True)   # ✓ Guaranteed coverage
self.hybrid_cb.setChecked(True)      # ✓ Conservative envelope
```

**Rationale:**
- Conformal: Distribution-free, guaranteed coverage (user wants confidence)
- Hybrid: Conservative envelope (user wants safety margin)
- Epistemic/Total: Advanced users can enable for deeper analysis
- Reduces default clutter from 4 bands to 2

---

### Recommendation 3: Optimize Legend Layout

#### **Strategy A: Two-tier legend**

**Primary legend (always visible):**
- Training data
- Observed forecast
- Forecast mean
- Active uncertainty band(s)

**Secondary info (in metrics box):**
- Bias correction note (if enabled)
- MCMC acceptance rate
- Band methodology notes

#### **Strategy B: Smart positioning**

```python
# Auto-select best legend position based on data density
def _best_legend_position(times, values, mean_curve):
    """Choose legend location with minimum data overlap."""
    # Test 4 corners + 4 edges
    positions = ['upper right', 'upper left', 'lower right', 'lower left',
                'center right', 'center left', 'upper center', 'lower center']

    # For degradation data (decreasing), upper corners typically clear
    # But check if data extends to time=564
    if times[-1] > 500:
        # Data extends far right
        return 'upper left'  # Safer
    else:
        return 'upper right'  # Classic position
```

#### **Strategy C: Smarter text**

**Current labels (verbose):**
```
"Epistemic band (model)"
"Total band (model+noise)"
"Conformal interval (guaranteed)"
"Hybrid interval (union)"
```

**Proposed labels (concise):**
```
"Epistemic (parameters)"
"Total (params + noise)"
"Conformal (CV+)"
"Hybrid (envelope)"
```

Or even simpler:
```
"Epistemic"
"Total"
"Conformal"
"Hybrid"
```

With explanation moved to tooltip or metrics box.

---

### Recommendation 4: Differentiate Band Styling

Instead of all bands using identical styling:
```python
# Current (all same)
alpha=0.18, linewidth=1.0, linestyle="-"
```

**Proposed: Visual hierarchy**
```python
# Primary band (e.g., Conformal) - Most prominent
alpha=0.30, linewidth=1.5, linestyle="-", edgecolor=darker_shade

# Secondary band (e.g., Hybrid) - Subtle
alpha=0.15, linewidth=1.0, linestyle="--", edgecolor=None

# Tertiary bands (Epistemic/Total) - Very subtle
alpha=0.10, linewidth=0.8, linestyle=":", edgecolor=None
```

**Effect:**
- Users immediately see the "main" uncertainty
- Optional bands don't compete for attention
- Clear visual priority

---

### Recommendation 5: Enlarge Font Sizes

**Current:**
- Legend: `fontsize=9`
- Axis labels: `labelsize=10`
- Metrics box: `fontsize=9`

**Proposed:**
- Legend: `fontsize=10` (at least)
- Axis labels: `labelsize=11`
- Axis titles: `fontsize=12`, `fontweight='bold'`
- Metrics box: `fontsize=10`
- Title: `fontsize=14`, `fontweight='bold'`

**At 300 DPI export for 10"×8" plot:**
- Current 9pt ≈ 2.7mm at full size (too small)
- Proposed 10-11pt ≈ 3.0-3.3mm (readable)

---

### Recommendation 6: Metrics Box Positioning

**Current:**
```python
ax.text(0.72, 0.5,  # Fixed position in axes coordinates
        "\n".join(summary_lines),
        transform=ax.transAxes, ...)
```

**Problems:**
- May overlap with legend (both upper right)
- Fixed position doesn't adapt to data

**Proposed:**
```python
# Position metrics box intelligently
if legend_position == 'upper right':
    metrics_x, metrics_y = 0.02, 0.02  # Lower left
elif legend_position == 'upper left':
    metrics_x, metrics_y = 0.98, 0.02  # Lower right
    ha = 'right'
else:
    metrics_x, metrics_y = 0.72, 0.5   # Original position
```

Or better: **Move metrics to plot title or subtitle**

---

## Proposed Implementation Plan

### Phase 1: Quick Wins (30 min)

1. **Change default checkboxes:**
   ```python
   self.epistemic_cb.setChecked(False)
   self.total_cb.setChecked(False)
   self.conformal_cb.setChecked(True)
   self.hybrid_cb.setChecked(True)
   ```

2. **Increase font sizes:**
   ```python
   fontsize=10 → fontsize=11  (legend)
   fontsize=9 → fontsize=10   (metrics box)
   ```

3. **Simplify legend labels:**
   Remove parentheticals for cleaner look

### Phase 2: Color Palette (1 hour)

1. Replace color constants with accessible palette
2. Test with color-blindness simulator
3. Ensure sufficient contrast ratios (WCAG AA: 4.5:1)

### Phase 3: Band Styling (1 hour)

1. Implement visual hierarchy (primary/secondary/tertiary alphas)
2. Vary line styles (solid/dashed/dotted)
3. Add thicker border to primary band

### Phase 4: Smart Legend (2 hours)

1. Implement auto-positioning algorithm
2. Dynamic column count (1 col if ≤4 items, 2 cols if 5-8, 3 cols if 9+)
3. Test with various datasets

---

## Specific Code Changes

### Change 1: Default Band Selection
**File:** `app_ui.py`
**Lines:** 554-569

```python
# BEFORE
self.epistemic_cb = QCheckBox("Epistemic")
self.epistemic_cb.setChecked(True)  # ← Change

self.total_cb = QCheckBox("Total")
self.total_cb.setChecked(True)      # ← Change

self.conformal_cb = QCheckBox("Conformal")
self.conformal_cb.setChecked(True)  # Keep

self.hybrid_cb = QCheckBox("Hybrid")
self.hybrid_cb.setChecked(True)     # Keep

# AFTER
self.epistemic_cb = QCheckBox("Epistemic")
self.epistemic_cb.setChecked(False)  # ✓ Advanced users only

self.total_cb = QCheckBox("Total")
self.total_cb.setChecked(False)      # ✓ Redundant with hybrid

self.conformal_cb = QCheckBox("Conformal")
self.conformal_cb.setChecked(True)   # ✓ Default: guaranteed coverage

self.hybrid_cb = QCheckBox("Hybrid")
self.hybrid_cb.setChecked(True)      # ✓ Default: conservative envelope
```

---

### Change 2: Simplified Legend Labels
**File:** `app_ui.py`
**Lines:** 1672, 1685, 1698, 1711

```python
# BEFORE
_legend(epi_handle, "Epistemic band (model)")
_legend(total_handle, "Total band (model+noise)")
_legend(conf_handle, "Conformal interval (guaranteed)")
_legend(hybrid_handle, "Hybrid interval (union)")

# AFTER
_legend(epi_handle, "Epistemic")
_legend(total_handle, "Total")
_legend(conf_handle, "Conformal")
_legend(hybrid_handle, "Hybrid")
```

---

### Change 3: Improved Color Palette
**File:** `app_ui.py`
**Lines:** 54-63

```python
# BEFORE (similar blues, poor contrast)
EPISTEMIC_COLOR = "#8ea8ba"
TOTAL_COLOR = "#b7c6d0"
CONFORMAL_COLOR = "#d9cbb0"
HYBRID_COLOR = "#b38b44"

# AFTER (distinct hues, high contrast)
EPISTEMIC_COLOR = "#42A5F5"    # Bright blue
TOTAL_COLOR = "#AB47BC"        # Purple
CONFORMAL_COLOR = "#FF7043"    # Orange
HYBRID_COLOR = "#66BB6A"       # Green
```

---

### Change 4: Larger Font Sizes
**File:** `app_ui.py`
**Lines:** 1725, 1745, 868

```python
# BEFORE
ax.legend(..., fontsize=9, ...)
ax.text(..., fontsize=9, ...)

# AFTER
ax.legend(..., fontsize=11, ...)      # Legend text
ax.text(..., fontsize=10, ...)        # Metrics box
ax.set_xlabel(..., fontsize=11)       # Axis labels
ax.set_ylabel(..., fontsize=11)
ax.set_title(..., fontsize=13, fontweight='bold')  # Title
```

---

### Change 5: Visual Hierarchy for Bands
**File:** `app_ui.py`
**Lines:** 1665-1710

```python
# Epistemic (tertiary if shown)
epi_handle = ax.fill_between(
    times[mask], ep_low[mask], ep_high[mask],
    color=EPISTEMIC_COLOR,
    alpha=0.12,  # ← More subtle
    linewidth=0,
    edgecolor=None
)
ax.plot(times[mask], ep_low[mask], color=EPISTEMIC_COLOR,
        linewidth=0.8, linestyle=":", alpha=0.7)
ax.plot(times[mask], ep_high[mask], color=EPISTEMIC_COLOR,
        linewidth=0.8, linestyle=":", alpha=0.7)

# Total (secondary)
total_handle = ax.fill_between(
    times[mask], tot_low[mask], tot_high[mask],
    color=TOTAL_COLOR,
    alpha=0.18,  # ← Moderate
    linewidth=0,
    edgecolor=None
)
ax.plot(times[mask], tot_low[mask], color=TOTAL_COLOR,
        linewidth=1.0, linestyle="--", alpha=0.85)
ax.plot(times[mask], tot_high[mask], color=TOTAL_COLOR,
        linewidth=1.0, linestyle="--", alpha=0.85)

# Conformal (primary if checked)
conf_handle = ax.fill_between(
    times[mask], conf_low[mask], conf_high[mask],
    color=CONFORMAL_COLOR,
    alpha=0.25,  # ← More prominent
    linewidth=0,
    edgecolor=None
)
# Add thicker edge for emphasis
ax.plot(times[mask], conf_low[mask], color=CONFORMAL_COLOR,
        linewidth=1.5, linestyle="-", alpha=1.0)
ax.plot(times[mask], conf_high[mask], color=CONFORMAL_COLOR,
        linewidth=1.5, linestyle="-", alpha=1.0)

# Hybrid (primary envelope)
hybrid_handle = ax.fill_between(
    times[mask], hybrid_low[mask], hybrid_high[mask],
    color=HYBRID_COLOR,
    alpha=0.15,  # ← Subtle outer envelope
    linewidth=0,
    edgecolor=None
)
ax.plot(times[mask], hybrid_low[mask], color=HYBRID_COLOR,
        linewidth=1.2, linestyle="--", alpha=0.9)
ax.plot(times[mask], hybrid_high[mask], color=HYBRID_COLOR,
        linewidth=1.2, linestyle="--", alpha=0.9)
```

---

## Testing Checklist

After implementation:

- [ ] Generate plot with only Conformal + Hybrid (default)
- [ ] Verify legend has ≤6 entries (not 10)
- [ ] Check color distinguishability
- [ ] Verify font sizes are readable at 300 DPI
- [ ] Test with color-blindness simulator (deuteranopia, protanopia)
- [ ] Ensure legend doesn't overlap metrics box
- [ ] Export as PNG at 300 DPI, open at 100% zoom, verify readability
- [ ] Test with all 4 bands enabled (should still be usable)
- [ ] Verify band edges are visible
- [ ] Check that mean forecast line is most prominent element

---

## Expected Impact

### Before:
- 10 possible legend entries
- 4 bands with similar colors (blue-grays, tans)
- Small 9pt font
- Visual clutter
- User confusion about which band to trust

### After:
- 5-6 typical legend entries (40% reduction)
- 4 bands with distinct colors (blue, purple, orange, green)
- Larger 11pt font (22% bigger)
- Clear visual hierarchy
- Default shows 2 most useful bands
- Users can enable advanced bands if needed

---

## Estimated Effort

| Phase | Task | Time |
|-------|------|------|
| Phase 1 | Default checkboxes + font sizes + labels | 30 min |
| Phase 2 | Color palette redesign | 1 hour |
| Phase 3 | Band styling hierarchy | 1 hour |
| Phase 4 | Smart legend positioning (optional) | 2 hours |
| **Total** | **Quick wins to full implementation** | **30 min - 4.5 hours** |

**Recommended:** Start with Phase 1 (quick wins), get user feedback, then proceed

---

## Priority Ranking

1. **HIGH:** Change default band selection (Phases 1) - Immediate impact, 30 min
2. **HIGH:** Simplify legend labels (Phase 1) - Cleaner, 5 min
3. **MEDIUM:** Larger fonts (Phase 1) - Readability, 10 min
4. **MEDIUM:** Color palette (Phase 2) - Accessibility, 1 hour
5. **MEDIUM:** Band styling (Phase 3) - Visual hierarchy, 1 hour
6. **LOW:** Smart positioning (Phase 4) - Nice-to-have, 2 hours

---

**Analysis complete.**
**Recommendation: Start with Phase 1 (30 minutes) for 80% of the benefit.**
