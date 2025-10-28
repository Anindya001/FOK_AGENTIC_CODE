# Fractional-Order Upgrade Blueprint for AEC Degradation Study

## Target Outcome
- Elevate the existing aluminium electrolytic capacitor (AEC) prognostics study to a tier-1 publication by replacing the exponential-with-quadratic degradation model with a mechanistic **fractional-order kinetics** formulation and delivering state-of-the-art uncertainty quantification (UQ).
- Demonstrate both **physical novelty** (fractional PoF modelling) and **statistical rigor** (Monte Carlo + conformal guarantees), validated on the eight capacitor datasets (C1–C8).

## Physics Modelling Roadmap
1. **Derive fractional diffusion model**
   - Start from electrolyte mass balance with anomalous diffusion: apply Caputo derivative of order \(\alpha\) to the transport term.
   - Solve for capacitance decay: \(C(t) = C_\infty + (C_0 - C_\infty) E_\alpha(-(t/\tau)^\alpha)\), where \(E_\alpha\) is the Mittag-Leffler function.
   - Explain physical meaning of \(\alpha\) (0<\(\alpha\)≤1): deviations from classical Fickian evaporation.
2. **Parameter estimation**
   - Parameters: \(\theta = (C_\infty, C_0, \alpha, \tau)\) (or log-transforms).
   - Use robust least squares (Huber) or full Bayesian posterior (Laplace / MCMC) on training window.
   - Compare model evidence (AIC/BIC/WAIC) vs previous quadratic model; report cross-validated RMSE.
3. **Sensitivity analysis**
   - Extend Sobol indices to \(\alpha\), \(\tau\), etc.
   - Provide physical interpretation (e.g., \(\alpha\) drives long-tail behaviour).

## Hybrid Uncertainty Framework
1. **Monte Carlo posterior sampling**
   - Sample \(\theta^{(m)}\) from approximate posterior (Laplace covariance or residual bootstrap).
   - Generate ensemble trajectories \(C^{(m)}(t)\); add optional multiplicative measurement noise.
2. **Conformalized Monte Carlo (CMC)**
   - For calibration points, compute tail probabilities under MC predictive distribution.
   - Obtain conformity scores; set \(q_\alpha\) for target miscoverage.
   - Adjust lower/upper bounds: \(y_k^{\text{lower}} = F_k^{-1}(q_\alpha)\), \(y_k^{\text{upper}} = F_k^{-1}(1-q_\alpha)\).
   - Guarantees finite-sample coverage while retaining MC distributional shape.
3. **Diagnostics**
   - Empirical coverage plots vs targets (90%, 95%, 99.7%).
   - Residual QQ plots / histograms to discuss model mismatch.

## Adaptive Estimation Refresh
- Replace DTW neighbour blending with either:
  - Hierarchical Bayesian prior (neighbour parameters form prior mean/covariance), or
  - Retain DTW but provide theoretical discussion / simulation to show bias reduction.
- Ensure the fractional parameter estimates benefit from neighbour information without violating physics (e.g., enforce \(0<\alpha≤1\)).

## Results & Visualisation
1. **Forecast plots per capacitor**
   - Training scatter, dashed fit, red vertical boundary, fractional forecast curve, dual confidence bands (95%, 90%), ground truth tail, metrics box (RMSE, MAE, coverage).
2. **Tables/figures**
   - Parameter summaries (\(\alpha\) mean ± CI) across capacitors.
   - Model comparison (old vs fractional) with RMSE, WAIC.
   - Coverage performance of CMC vs bootstrap-only.
3. **Text updates**
   - Reframe abstract/introduction: highlight fractional kinetics + conformalized MC.
   - Update methodology sections (fractional derivation, MC+conformal algorithm boxes).
   - Discuss insights (e.g., \(\alpha\approx0.6\) indicates sub-diffusive evaporation).

## References to Integrate
- Fractional diffusion in electrochemistry (e.g., papers on anomalous transport in electrolytes).
- Conformalized Monte Carlo / distribution-free predictive inference (Barber et al., Fisac et al.).
- Hybrid physics-informed UQ literature.

## Implementation Checklist
- [ ] Port fractional Mittage-Leffler solver (e.g., SciPy/mathematical functions) and gradients.
- [ ] Fit fractional model on C1–C8; compute posterior covariance.
- [ ] Implement CMC pipeline; validate on calibration segments.
- [ ] Regenerate figures/tables with new results.
- [ ] Revise manuscript sections per roadmap.

## Deliverables
- Updated code modules (`legacy_core` or new fractional module) with fractional model + CMC.
- New figures (forecast plots, coverage plots, parameter tables).
- Revised manuscript emphasising fractional model and hybrid UQ.

