# Variance-Aware Robust Sensor Placement for Barrier Coverage Systems

## Overview

This repository contains the research task submission for **Dr. Mingyu Kim's research group** at Georgia Southern University. The work identifies and addresses a methodological gap in the paper "Near-optimal Sensor Placement for Detecting Stochastic Target Trajectories in Barrier Coverage Systems" (Kim et al., IEEE SysCon 2025).

## The Gap

The paper computes **posterior variance Var[λ(ℓ)]** via INLA (Integrated Nested Laplace Approximation) but does not incorporate it into the optimization objective (Equation 3). Instead, variance is only used post-hoc for robustness validation. This work proposes directly incorporating variance into the sensor placement objective.

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

**Finding:** The robust method outperforms the baseline on **both metrics**, validating that variance should be incorporated into sensor placement optimization.

## Files

### Code
- **`sensor_placement_SEPARATE_FIGURES.py`** - Complete Python simulation
  - Implements greedy sensor placement algorithm
  - Computes void probability for both methods
  - Evaluates on nominal and worst-case scenarios
  - Generates 5 publication-quality figures

### Report
- **`Research_Task_Final_Report.docx`** - Full 2-4 page research report with:
  - Problem summary and gap identification
  - Mathematical formulation
  - Simulation setup and methodology
  - Results analysis and performance tables
  - Discussion of findings
  - Next steps and future directions

### Figures
- **`Fig1_Baseline_Placement.png`** - Baseline sensor placement on mean intensity map
- **`Fig2_Robust_Placement.png`** - Robust sensor placement on adjusted intensity
- **`Fig3_Posterior_Uncertainty.png`** - Posterior variance map (computed but unused in paper)
- **`Fig4_Performance_Comparison.png`** - Bar chart comparing nominal vs. worst-case performance
- **`Fig5_Trade_off_Curve.png`** - Robustness-optimality trade-off frontier

## Installation & Usage

### Requirements
```bash
pip3 install numpy matplotlib scipy
```

### Run Simulation
```bash
python3 sensor_placement_SEPARATE_FIGURES.py
```

This will:
1. Generate synthetic trajectory intensity data (mean + variance)
2. Run baseline optimization (mean-only, paper's method)
3. Run robust optimization (mean + variance, proposed extension)
4. Compute performance metrics on nominal and worst-case scenarios
5. Generate 5 publication-quality figures
6. Print performance summary to console

### Output
- **Console:** Performance tables, sensor locations, summary statistics
- **Figures:** Fig1-Fig5 saved as PNG files

## Methodology

### Simulation Setup
- **Representation Space:** 40 × 40 grid (α ∈ [0,π], p ∈ [−10, 10] km)
- **Intensity Data:** Synthetic Gaussian mean E[λ(ℓ)] and variance Var[λ(ℓ)]
- **Sensor Candidates:** 15 × 15 grid (225 locations in inertial space)
- **Algorithm:** Greedy sensor placement (5 sensors selected)
- **Detection Model:** Gaussian γ(d) = 0.95 · exp(−d²/1.0)

### Evaluation Scenarios
1. **Nominal:** Expected intensity E[λ(ℓ)]
2. **Worst-Case:** Conservative bound E[λ(ℓ)] − 2√Var[λ(ℓ)]

## Key Contributions

1. **Gap Identification:** Found that posterior variance is computed but unused in optimization
2. **Novel Extension:** Proposed variance-aware objective function
3. **Empirical Validation:** Demonstrated 4.44% nominal + 2.13% robustness improvement
4. **Research Alignment:** Directly addresses Dr. Kim's work on "uncertainty in Poisson target models"

## Future Directions

1. **Real Data Validation** - Reproduce with actual AIS data (Hampton Roads)
2. **Sophisticated Optimization** - Implement INLA + Newton refinement
3. **Theoretical Analysis** - Provide convergence guarantees and approximation bounds
4. **Robust Baselines** - Compare against other robust optimization approaches
5. **Application Domains** - Extend to underwater sensors, airborne surveillance, etc.

## Research Alignment

This work directly extends Dr. Kim's recent research on:
- "On the role of uncertainty in Poisson target models" (IEEE Access, 2023)
- "Robust Sensor Placement for Poisson Arrivals with False Alarm Aware Spatiotemporal Sensing" (arXiv, 2025)

## Author

**Abdulahi Taiwo**  
PhD Student, Computer Engineering  
University of South Carolina

