# Variance-Aware Robust Sensor Placement for Barrier Coverage Systems

## Overview

This repository contains a research task for **Dr. Mingyu Kim's research group** at Georgia Southern University. The work identifies and addresses a methodological gap in the paper "Near-optimal Sensor Placement for Detecting Stochastic Target Trajectories in Barrier Coverage Systems" (Kim et al., IEEE SysCon 2025).

## The Gap

The paper computes **posterior variance Var[λ(ℓ)]** via INLA (Integrated Nested Laplace Approximation) but does not use it in the optimization objective (Equation 3). Variance is used only after the fact, to check robustness. This work puts the variance directly into the sensor placement objective.

## Proposed Extension: Variance-Aware Optimization

### Baseline (Paper's Method)

â = argmax_a exp( −∫_Θ (1/T_c) E[λ(ℓ)] π_C(ℓ,a) dℓ )

### Proposed Robust Method

â_robust(β) = argmax_a exp( −∫_Θ (1/T_c) [E[λ(ℓ)] + β√Var[λ(ℓ)]] π_C(ℓ,a) dℓ )

where **β ∈ [0, 2]** is a robustness parameter balancing nominal optimality and worst-case robustness.

## Key Results

| Metric | Baseline | Robust | Improvement |
|--------|----------|--------|-------------|
| **Nominal Void Probability** | 0.5409 | 0.5650 | **+4.44%** |
| **Worst-Case Void Probability** | 0.7552 | 0.7712 | **+2.13%** |

**Finding:** in this synthetic simulation, the robust method scored higher than the baseline on both metrics.

## Files

### Code
- **`sensorplacement.py`**: the complete Python simulation
  - Implements the greedy sensor placement algorithm
  - Computes void probability for both methods
  - Evaluates nominal and worst-case scenarios
  - Generates the 5 figures below

### Figures
- **`Fig1_Baseline_Placement.png`**: baseline sensor placement on the mean intensity map
- **`Fig2_Robust_Placement.png`**: robust sensor placement on the adjusted intensity
- **`Fig3_Posterior_Uncertainty.png`**: posterior variance map (computed but unused in the original paper)
- **`Fig4_Performance_Comparison.png`**: nominal vs. worst-case performance
- **`Fig5_Trade_off_Curve.png`**: robustness–optimality trade-off as β varies

## Installation & Usage

### Requirements
```bash
pip3 install numpy matplotlib scipy
```

### Run Simulation
```bash
python3 sensorplacement.py
```

This will:
1. Generate synthetic trajectory intensity data (mean + variance)
2. Run the baseline optimization (mean only, the paper's method)
3. Run the robust optimization (mean + variance, the proposed extension)
4. Compute performance metrics for the nominal and worst-case scenarios
5. Generate the 5 figures
6. Print a performance summary to the console

## Methodology

### Simulation Setup
- **Representation space:** 40 × 40 grid (α ∈ [0, π], p ∈ [−10, 10] km)
- **Intensity data:** synthetic Gaussian mean E[λ(ℓ)] and variance Var[λ(ℓ)]
- **Sensor candidates:** 15 × 15 grid (225 locations in inertial space)
- **Algorithm:** greedy sensor placement (5 sensors selected)
- **Detection model:** Gaussian γ(d) = 0.95 · exp(−d²/1.0)

### Evaluation Scenarios
1. **Nominal:** expected intensity E[λ(ℓ)]
2. **Worst case:** conservative bound E[λ(ℓ)] − 2√Var[λ(ℓ)]

## Limitations

- All data are synthetic; the results have not yet been validated on real trajectory data.
- Results come from a single simulated scenario with greedy (not exact) optimization.

## Future Directions

1. **Real data validation:** reproduce with actual AIS data (Hampton Roads)
2. **Better optimization:** implement INLA + Newton refinement
3. **Theory:** convergence guarantees and approximation bounds
4. **Robust baselines:** compare against other robust optimization approaches
5. **Other domains:** underwater sensors, airborne surveillance

## Research Alignment

This work extends Dr. Kim's research on:
- "On the role of uncertainty in Poisson target models" (IEEE Access, 2023)
- "Robust Sensor Placement for Poisson Arrivals with False Alarm Aware Spatiotemporal Sensing" (arXiv, 2025)

## Author

**Abdulahi Taiwo**
PhD Student, Computer Engineering
University of South Carolina
