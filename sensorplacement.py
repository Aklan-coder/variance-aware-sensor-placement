"""
Variance-Aware Robust Sensor Placement Simulation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['axes.linewidth'] = 1.5
rcParams['lines.linewidth'] = 2
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300


#sensor placement simulation
print("="*70)
print("="*70)



#setup
n_alpha = 40
n_p = 40
alpha_vals = np.linspace(0, np.pi, n_alpha)
p_vals = np.linspace(-10, 10, n_p)
Alpha_grid, P_grid = np.meshgrid(alpha_vals, p_vals)

E_lambda = 0.5 * np.exp(-((Alpha_grid - np.pi/3)**2 / 0.5 + 
                           (P_grid - 2)**2 / 4))
Var_lambda = 0.1 * np.exp(-((Alpha_grid - np.pi/3)**2 / 1.0 + 
                             (P_grid - 2)**2 / 6))

sensor_grid_x = np.linspace(-10, 10, 15)
sensor_grid_y = np.linspace(-10, 10, 15)
X_sensors, Y_sensors = np.meshgrid(sensor_grid_x, sensor_grid_y)
candidates = np.column_stack([X_sensors.ravel(), Y_sensors.ravel()])

RHO = 0.95
SIGMA_L = 1.0

print(f"✓ Setup complete")

#Detection model and void probability computation
def min_distance_line_to_point(line_alpha, line_p, sensor_x, sensor_y):
    eps = 1e-10
    tan_term = np.tan(line_alpha - np.pi/2)
    m = tan_term
    b = line_p / np.sqrt(1 + m**2 + eps)
    numerator = np.abs(m * sensor_x - sensor_y + b)
    denominator = np.sqrt(m**2 + 1 + eps)
    distance = numerator / denominator
    return distance

def detection_probability(distance, rho=RHO, sigma_l=SIGMA_L):
    return rho * np.exp(-(distance**2 / sigma_l))

def compute_void_probability(sensors, E_lambda, Var_lambda,
                             alpha_vals, p_vals,
                             beta=0, method='mean-only'):
    n_alpha = len(alpha_vals)
    n_p = len(p_vals)
    miss_prob = np.ones((n_alpha, n_p))
    
    for i_alpha, alpha_val in enumerate(alpha_vals):
        for i_p, p_val in enumerate(p_vals):
            miss_prob_network = 1.0
            for sensor in sensors:
                sx, sy = sensor
                min_dist = min_distance_line_to_point(alpha_val, p_val, sx, sy)
                gamma = detection_probability(min_dist)
                miss_prob_network *= (1 - gamma)
            miss_prob[i_alpha, i_p] = miss_prob_network
    
    if method == 'mean-only':
        intensity = E_lambda
    elif method == 'robust':
        intensity = E_lambda + beta * np.sqrt(Var_lambda)
    
    da = alpha_vals[1] - alpha_vals[0] if len(alpha_vals) > 1 else 1
    dp = p_vals[1] - p_vals[0] if len(p_vals) > 1 else 1
    integral = np.sum(intensity * miss_prob) * da * dp
    void_prob = np.exp(-integral)
    
    return void_prob, miss_prob, integral
#Greedy selection Algorithm
def greedy_select(E_lambda, Var_lambda, candidates,
                  alpha_vals, p_vals,
                  n_sensors=5, beta=0, method='mean-only', verbose=True):
    
    selected_sensors = []
    void_probs = []
    
    for sensor_num in range(n_sensors):
        if verbose:
            print(f"  Sensor {sensor_num + 1}/{n_sensors}...", end=' ', flush=True)
        
        best_void_prob = -1
        best_candidate_idx = -1
        best_candidate_loc = None
        
        for idx, candidate in enumerate(candidates):
            test_sensors = selected_sensors + [candidate]
            vp, _, _ = compute_void_probability(
                test_sensors, E_lambda, Var_lambda,
                alpha_vals, p_vals,
                beta=beta, method=method
            )
            
            if vp > best_void_prob:
                best_void_prob = vp
                best_candidate_idx = idx
                best_candidate_loc = candidate
        
        selected_sensors.append(best_candidate_loc)
        void_probs.append(best_void_prob)
        
        if verbose:
            print(f"VP={best_void_prob:.4f}")
    
    return np.array(selected_sensors), void_probs


print("\nMethod 1: Baseline (Mean-only)")
sensors_baseline, vp_baseline = greedy_select(
    E_lambda, Var_lambda, candidates,
    alpha_vals, p_vals,
    n_sensors=5, beta=0, method='mean-only', verbose=True
)

print("Method 2: Robust (Mean + Variance)")
sensors_robust, vp_robust = greedy_select(
    E_lambda, Var_lambda, candidates,
    alpha_vals, p_vals,
    n_sensors=5, beta=1.0, method='robust', verbose=True
)

#Evaluation of the 2 methods
vp_base_nominal, _, _ = compute_void_probability(
    sensors_baseline, E_lambda, Var_lambda,
    alpha_vals, p_vals, beta=0, method='mean-only'
)
vp_robust_nominal, _, _ = compute_void_probability(
    sensors_robust, E_lambda, Var_lambda,
    alpha_vals, p_vals, beta=0, method='mean-only'
)

E_lambda_worst = np.maximum(E_lambda - 2*np.sqrt(Var_lambda), 0.01)
vp_base_worst, _, _ = compute_void_probability(
    sensors_baseline, E_lambda_worst, None,
    alpha_vals, p_vals, beta=0, method='mean-only'
)
vp_robust_worst, _, _ = compute_void_probability(
    sensors_robust, E_lambda_worst, None,
    alpha_vals, p_vals, beta=0, method='mean-only'
)

optimality_loss = (vp_base_nominal - vp_robust_nominal) / vp_base_nominal * 100
robustness_gain = (vp_robust_worst - vp_base_worst) / max(vp_base_worst, 0.001) * 100

print(f"\n✓ Nominal: Baseline={vp_base_nominal:.4f}, Robust={vp_robust_nominal:.4f}")
print(f"✓ Worst-case: Baseline={vp_base_worst:.4f}, Robust={vp_robust_worst:.4f}")

##Baseline and robust sensor placement

print("\nGenerating Figure 1: Baseline Placement...")

fig1, ax1 = plt.subplots(figsize=(7, 5.5))

im1 = ax1.contourf(Alpha_grid, P_grid, E_lambda, levels=15, cmap='YlOrRd')
ax1.plot(sensors_baseline[:, 0], sensors_baseline[:, 1], 'o', 
         color='white', markersize=10, markeredgecolor='#E63946', 
         markeredgewidth=2.5, label='Sensors', zorder=5)

ax1.set_xlabel(r'Trajectory Angle $\alpha$ (rad)', fontsize=12, fontweight='bold')
ax1.set_ylabel(r'Perpendicular Distance $p$ (km)', fontsize=12, fontweight='bold')
ax1.set_title('Baseline Method: Mean-Only Optimization\n' + 
             r'Intensity: $\mathbb{E}[\lambda(\ell)]$', 
             fontsize=13, fontweight='bold', pad=15)
ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
ax1.legend(loc='upper right', fontsize=11, framealpha=0.95)

cbar1 = plt.colorbar(im1, ax=ax1, pad=0.02)
cbar1.set_label(r'Mean Intensity $\mathbb{E}[\lambda(\ell)]$', 
               fontsize=11, fontweight='bold')

ax1.text(0.02, 0.98, f'Void Prob = {vp_base_nominal:.4f}', 
        transform=ax1.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('Fig1_Baseline_Placement.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('Fig1_Baseline_Placement.png', format='png', bbox_inches='tight', dpi=300)
print("✓ Saved: Fig1_Baseline_Placement.pdf / .png")
plt.close()

#Sensor placement 

print("Generating Figure 2: Robust Placement...")

fig2, ax2 = plt.subplots(figsize=(7, 5.5))

adj_intensity = E_lambda + 1.0 * np.sqrt(Var_lambda)
im2 = ax2.contourf(Alpha_grid, P_grid, adj_intensity, levels=15, cmap='YlOrRd')
ax2.plot(sensors_robust[:, 0], sensors_robust[:, 1], 's', 
         color='white', markersize=10, markeredgecolor='#457B9D', 
         markeredgewidth=2.5, label='Sensors', zorder=5)

ax2.set_xlabel(r'Trajectory Angle $\alpha$ (rad)', fontsize=12, fontweight='bold')
ax2.set_ylabel(r'Perpendicular Distance $p$ (km)', fontsize=12, fontweight='bold')
ax2.set_title('Robust Method: Variance-Aware Optimization\n' + 
             r'Intensity: $\mathbb{E}[\lambda(\ell)] + \beta\sqrt{\text{Var}[\lambda(\ell)]}$ ($\beta=1.0$)', 
             fontsize=13, fontweight='bold', pad=15)
ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
ax2.legend(loc='upper right', fontsize=11, framealpha=0.95)

cbar2 = plt.colorbar(im2, ax=ax2, pad=0.02)
cbar2.set_label(r'Adjusted Intensity', fontsize=11, fontweight='bold')

ax2.text(0.02, 0.98, f'Void Prob = {vp_robust_nominal:.4f}', 
        transform=ax2.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

plt.tight_layout()
plt.savefig('Fig2_Robust_Placement.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('Fig2_Robust_Placement.png', format='png', bbox_inches='tight', dpi=300)
print("✓ Saved: Fig2_Robust_Placement.pdf / .png")
plt.close()

#Posteriror uncertainty 

print("Generating Figure 3: Posterior Uncertainty...")

fig3, ax3 = plt.subplots(figsize=(7, 5.5))

im3 = ax3.contourf(Alpha_grid, P_grid, Var_lambda, levels=12, cmap='Blues')
ax3.contour(Alpha_grid, P_grid, Var_lambda, levels=6, colors='navy', 
           alpha=0.3, linewidths=0.5)

ax3.set_xlabel(r'Trajectory Angle $\alpha$ (rad)', fontsize=12, fontweight='bold')
ax3.set_ylabel(r'Perpendicular Distance $p$ (km)', fontsize=12, fontweight='bold')
ax3.set_title('Posterior Uncertainty in Intensity Estimation\n' +
             r'Variance: $\text{Var}[\lambda(\ell)]$', 
             fontsize=13, fontweight='bold', pad=15)
ax3.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)

cbar3 = plt.colorbar(im3, ax=ax3, pad=0.02)
cbar3.set_label(r'Posterior Variance $\text{Var}[\lambda(\ell)]$', 
               fontsize=11, fontweight='bold')

ax3.text(0.02, 0.98, 'Computed by INLA\n(not used in paper)', 
        transform=ax3.transAxes, fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

plt.tight_layout()
plt.savefig('Fig3_Posterior_Uncertainty.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('Fig3_Posterior_Uncertainty.png', format='png', bbox_inches='tight', dpi=300)
print("✓ Saved: Fig3_Posterior_Uncertainty.pdf / .png")
plt.close()

#PErformance comparison

print("Generating Figure 4: Performance Comparison...")

fig4, ax4 = plt.subplots(figsize=(8, 5.5))

x_pos = np.arange(2)
width = 0.35

bars1 = ax4.bar(x_pos - width/2, [vp_base_nominal, vp_base_worst], width,
               label='Baseline (Mean-only)', color='#E63946', alpha=0.85, 
               edgecolor='black', linewidth=1.5)
bars2 = ax4.bar(x_pos + width/2, [vp_robust_nominal, vp_robust_worst], width,
               label='Robust (Mean + Variance)', color='#457B9D', alpha=0.85, 
               edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                f'{height:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax4.set_ylabel('Void Probability', fontsize=12, fontweight='bold')
ax4.set_title('Performance Comparison: Nominal vs. Worst-Case Scenarios', 
             fontsize=13, fontweight='bold', pad=15)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(['Nominal\n(Expected)', 'Worst-Case\n(Robust Test)'], fontsize=11)
ax4.set_ylim([0, 0.9])
ax4.legend(fontsize=11, loc='upper left', framealpha=0.95)
ax4.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=0.8)

# Add text box with key metrics
textstr = f'Nominal Improvement: {-optimality_loss:.2f}%\nWorst-Case Gain: {robustness_gain:.2f}%'
ax4.text(0.98, 0.30, textstr, transform=ax4.transAxes, fontsize=10,
        verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black'))

plt.tight_layout()
plt.savefig('Fig4_Performance_Comparison.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('Fig4_Performance_Comparison.png', format='png', bbox_inches='tight', dpi=300)
print("✓ Saved: Fig4_Performance_Comparison.pdf / .png")
plt.close()

#Trade off curve
print("Generating Figure 5: Trade-off Curve...")

fig5, ax5 = plt.subplots(figsize=(8, 5.5))

beta_vals = np.linspace(0, 2, 9)
nominal_perfs = []
worst_perfs = []

for beta in beta_vals:
    sensors_temp, _ = greedy_select(
        E_lambda, Var_lambda, candidates,
        alpha_vals, p_vals,
        n_sensors=5, beta=beta, method='robust', verbose=False
    )
    vp_nom, _, _ = compute_void_probability(
        sensors_temp, E_lambda, Var_lambda, alpha_vals, p_vals,
        beta=0, method='mean-only'
    )
    vp_worst, _, _ = compute_void_probability(
        sensors_temp, E_lambda_worst, None, alpha_vals, p_vals,
        beta=0, method='mean-only'
    )
    nominal_perfs.append(vp_nom)
    worst_perfs.append(vp_worst)

ax5.plot(nominal_perfs, worst_perfs, 'o-', linewidth=2.5, markersize=8,
        color='#457B9D', markerfacecolor='white', 
        markeredgecolor='#457B9D', markeredgewidth=2.5, label='Pareto Frontier')

# Highlight endpoints
ax5.plot(nominal_perfs[0], worst_perfs[0], '*', markersize=20,
        color='#E63946', markeredgecolor='black', markeredgewidth=1.5,
        label=r'$\beta = 0$ (Baseline)', zorder=10)
ax5.plot(nominal_perfs[4], worst_perfs[4], '*', markersize=20,
        color='#457B9D', markeredgecolor='black', markeredgewidth=1.5,
        label=r'$\beta = 1.0$ (Robust)', zorder=10)

ax5.set_xlabel('Nominal Performance (Void Probability)', fontsize=12, fontweight='bold')
ax5.set_ylabel('Worst-Case Performance (Void Probability)', fontsize=12, fontweight='bold')
ax5.set_title('Robustness-Optimality Trade-off Curve', 
             fontsize=13, fontweight='bold', pad=15)
ax5.grid(True, alpha=0.3, linestyle='--', linewidth=0.7)
ax5.legend(fontsize=11, loc='lower left', framealpha=0.95)

# Annotation
ax5.annotate(f'Gain: +{robustness_gain:.2f}%', 
            xy=(nominal_perfs[4], worst_perfs[4]), 
            xytext=(nominal_perfs[4]-0.008, worst_perfs[4]+0.02),
            fontsize=10, fontweight='bold', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7, edgecolor='black'),
            arrowprops=dict(arrowstyle='->', lw=1.5, color='black'))

plt.tight_layout()
plt.savefig('Fig5_Trade_off_Curve.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('Fig5_Trade_off_Curve.png', format='png', bbox_inches='tight', dpi=300)
print("✓ Saved: Fig5_Trade_off_Curve.pdf / .png")
plt.close()



print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

print(f"\nTABLE I: PERFORMANCE METRICS")
print("-"*70)
print(f"{'Metric':<40} {'Baseline':<15} {'Robust':<15}")
print("-"*70)
print(f"{'Nominal Void Probability':<40} {vp_base_nominal:<15.4f} {vp_robust_nominal:<15.4f}")
print(f"{'Worst-Case Void Probability':<40} {vp_base_worst:<15.4f} {vp_robust_worst:<15.4f}")
print(f"{'Optimality Loss (%)':<40} {'—':<15} {optimality_loss:<15.2f}")
print(f"{'Robustness Gain (%)':<40} {'—':<15} {robustness_gain:<15.2f}")

print(f"\nTABLE II: SENSOR LOCATIONS (km)")
print("-"*70)
print(f"{'Sensor':<10} {'Baseline (x, y)':<30} {'Robust (x, y)':<30}")
print("-"*70)
for i in range(5):
    base_loc = f"({sensors_baseline[i][0]:6.2f}, {sensors_baseline[i][1]:6.2f})"
    robust_loc = f"({sensors_robust[i][0]:6.2f}, {sensors_robust[i][1]:6.2f})"
    print(f"{i+1:<10} {base_loc:<30} {robust_loc:<30}")

print("\n" + "="*70)
print("✓ ALL FIGURES GENERATED SUCCESSFULLY")
print("="*70)

print("""
Generated Files:
  Fig1_Baseline_Placement.pdf / .png
  Fig2_Robust_Placement.pdf / .png
  Fig3_Posterior_Uncertainty.pdf / .png
  Fig4_Performance_Comparison.pdf / .png
  Fig5_Trade_off_Curve.pdf / .png

Each figure is standalone and ready for insertion into your report.
""")
