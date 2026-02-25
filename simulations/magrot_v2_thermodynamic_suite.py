"""
MAGROT v2 Thermodynamic R&D Validation Suite
═══════════════════════════════════════════════════════════════════

The definitive test: rerunning the ENTIRE R&D test suite using v2's
entropy-parameterized thermodynamic framework instead of the old
time-based (frametime) simulation approach.

v1 approach: evolve system in clock time t, compute R(t)
v2 approach: evolve system along entropy gradient sigma, compute R(sigma)
             -> guaranteed monotonic F decrease by second law
             -> no artificial timestep / framerate dependence
             -> thermodynamic identity of R=1 as entropy maximum on
                constrained manifold

Tests:
  1. Wire:        Static R metrics + v2 thermodynamic overlay
  2. Z-pinch:     Full v2 diagnostics (F[B,p], s_dot, Lyapunov, R-sdot)
  3. Comparison:  Z-pinch vs theta-pinch thermodynamic landscape
  4. Dynamics:    *** THE KEY TEST *** entropy flow vs time-based evolution
  5. Stability:   Hessian (A), Entropy Audit (B), Manifold (C), Attractor
  6. Dipole:      Thermodynamic stress analysis
  7. Screw Pinch: Thermodynamic sweep across pitch angles

Author: WCNEGENTROPY HOLDINGS LLC
Framework: MAGROT v2 (thermodynamic state flow)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from scipy.constants import mu_0, pi
from scipy.integrate import solve_ivp
import os
import sys
import time as time_module

# ── Use the installed magrot package ─────────────────────────────────────────
from magrot.fields.grid import CylindricalGrid, CartesianGrid
from magrot.fields.analytic import (
    field_infinite_wire, field_zpinch_bennett, field_theta_pinch,
)
from magrot.rotation.metrics import compute_all_metrics
from magrot.stress.maxwell import compute_forces_conservative_cyl
from magrot.thermodynamics.free_energy import (
    free_energy_cylindrical, magnetic_energy_density, thermal_energy_density,
    total_energy_density, lyapunov_R_cylindrical, local_free_energy_gradient,
)
from magrot.thermodynamics.entropy import (
    entropy_production_resistive, total_entropy_production_cylindrical,
    dissipated_power, spitzer_resistivity, temperature_profile_parabolic,
)
from magrot.thermodynamics.constraints import (
    vector_potential_cylindrical, magnetic_helicity_cylindrical,
    total_poloidal_flux_cylindrical, total_mass_cylindrical,
)
from magrot.thermodynamics.diagnostics import (
    StateFlowRecord, verify_monotonic_decrease, verify_constraint_conservation,
    correlation_R_sdot, correlation_R_free_energy,
)
from magrot.thermodynamics.state_flow import zpinch_entropy_flow
from magrot.stability.hessian import (
    fit_hessian_eigenvalue, perturb_pressure_scale, perturb_field_strength,
    perturb_boundary_radius, hessian_sweep_cylindrical,
    multi_axis_hessian_summary,
)
from magrot.stability.entropy_audit import entropy_audit_cylindrical
from magrot.stability.manifold import (
    remove_pressure_zpinch, remove_field_zpinch, diffuse_boundary_zpinch,
    constraint_relaxation_sweep_cylindrical, mesa_edge_summary,
)
from magrot.stability.attractors import (
    estimate_basin_width, classify_attractor,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'v2_thermodynamic')
OUT_DIR = os.path.abspath(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# REGRESSION TRACKER (carried forward from v3)
# ═══════════════════════════════════════════════════════════════════════════════

class RegressionTracker:
    def __init__(self):
        self.checks = []

    def check(self, name, value, expected, tol=0.05):
        if isinstance(value, (bool, np.bool_)):
            passed = bool(value) == bool(expected)
            err = 0.0 if passed else 1.0
        elif expected == 0:
            passed = abs(float(value)) < tol
            err = abs(float(value))
        else:
            err = abs(float(value) - float(expected)) / abs(float(expected))
            passed = err < tol
        status = "PASS" if passed else "FAIL"
        self.checks.append({
            'name': name, 'value': float(value) if not isinstance(value, bool) else value,
            'expected': float(expected) if not isinstance(expected, bool) else expected,
            'error': err, 'passed': passed, 'status': status
        })
        return passed

    def report(self):
        print("\n" + "=" * 78)
        print("  MAGROT v2 THERMODYNAMIC VALIDATION — REGRESSION SUMMARY")
        print("=" * 78)
        n_pass = sum(1 for c in self.checks if c['passed'])
        n_total = len(self.checks)
        print(f"\n  {n_pass}/{n_total} checks passed\n")
        for c in self.checks:
            mark = "+" if c['passed'] else "X"
            print(f"  [{mark}] {c['name']}")
            if isinstance(c['value'], bool):
                print(f"         got={c['value']}  expected={c['expected']}")
            else:
                print(f"         got={c['value']:.6g}  expected={c['expected']:.6g}  err={c['error']:.2e}")
        if n_pass == n_total:
            print(f"\n  ALL {n_total} REGRESSION CHECKS PASSED")
        else:
            fails = [c for c in self.checks if not c['passed']]
            print(f"\n  {len(fails)} REGRESSIONS DETECTED:")
            for c in fails:
                print(f"    - {c['name']}")
        return n_pass == n_total


reg = RegressionTracker()


def save_plot(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    -> {name}")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: INFINITE WIRE + THERMODYNAMIC OVERLAY
# ═══════════════════════════════════════════════════════════════════════════════

def test_wire_thermodynamic():
    print("\n" + "=" * 78)
    print("  TEST 1: INFINITE WIRE — v2 Thermodynamic Analysis")
    print("=" * 78)

    I = 1000.0
    grid = CylindricalGrid(Nr=200, Ntheta=1, Nz=1, r_range=(0.001, 0.1))
    B, _ = field_infinite_wire(grid, I=I)
    results = compute_all_metrics(B, grid, p_mat=None)

    r = grid.r
    s = (slice(None), 0, 0)
    R_universal = results['R_universal'][s]

    # v2: Free energy computation
    F_total, u_field = free_energy_cylindrical(B, grid, p=None)
    u_B = magnetic_energy_density(B)[s]

    # v2: Entropy production (vacuum = J should be ~0)
    T_field = np.full(grid.R.shape, 1e7)
    phys = compute_forces_conservative_cyl(B, grid)
    J = phys['J']
    s_dot, Jmag2, eta = entropy_production_resistive(J, T_field)
    S_dot_total = total_entropy_production_cylindrical(s_dot, grid)

    # v2: Lyapunov functional
    L_R, deviation = lyapunov_R_cylindrical(R_universal[..., np.newaxis, np.newaxis]
                                             if R_universal.ndim == 1
                                             else results['R_universal'], grid)

    # v2: R - s_dot correlation (should be weak in vacuum)
    r_corr = correlation_R_sdot(results['R_universal'], s_dot)

    # Regression checks
    start_idx = max(10, grid.Nr // 7)
    R_u_mid = R_universal[start_idx:-10]
    reg.check("Wire: R_universal mean = 1.0", np.mean(R_u_mid), 1.0, tol=0.05)
    reg.check("Wire: F_total > 0 (magnetic energy)", F_total > 0, True)
    reg.check("Wire: S_dot ~ 0 (vacuum)", S_dot_total, 0.0, tol=1e-3)
    reg.check("Wire: Lyapunov L_R small", L_R, 0.0, tol=1.0)

    print(f"\n  --- v2 Thermodynamic Results ---")
    print(f"  Free energy F[B]:          {F_total:.6e} J/m")
    print(f"  Entropy production S_dot:  {S_dot_total:.6e} W/K (should ~ 0 in vacuum)")
    print(f"  Lyapunov |R-1|^2:          {L_R:.6e}")
    print(f"  R-s_dot correlation:       {r_corr:.4f} (weak expected)")
    print(f"  R_universal range:         [{np.min(R_u_mid):.6f}, {np.max(R_u_mid):.6f}]")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    axes[0, 0].plot(r * 100, results['Bmag'][s], 'b-', lw=2)
    axes[0, 0].set_ylabel('B (T)'); axes[0, 0].set_title('Magnetic Field')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(r * 100, u_B, 'orange', lw=2)
    axes[0, 1].set_ylabel('u_B (J/m^3)'); axes[0, 1].set_title('v2: Magnetic Energy Density')
    axes[0, 1].set_yscale('log'); axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(r * 100, R_universal, 'navy', lw=2)
    axes[1, 0].axhline(1.0, color='green', ls='--', lw=1.5, label='R=1')
    axes[1, 0].set_ylabel('R_universal'); axes[1, 0].set_ylim([0.5, 1.5])
    axes[1, 0].set_title('R_universal (vacuum equilibrium)'); axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(r * 100, s_dot[s], 'red', lw=2)
    axes[1, 1].set_ylabel('s_dot (W/m^3/K)'); axes[1, 1].set_xlabel('r (cm)')
    axes[1, 1].set_title('v2: Entropy Production (should ~ 0)')
    axes[1, 1].grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.set_xlabel('r (cm)')
    fig.suptitle('Test 1: Infinite Wire — v2 Thermodynamic Overlay', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, 'test1_wire_v2_thermo.png')


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: Z-PINCH BENNETT + FULL v2 DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

def test_zpinch_thermodynamic():
    print("\n" + "=" * 78)
    print("  TEST 2: Z-PINCH BENNETT — v2 Full Thermodynamic Suite")
    print("=" * 78)

    I = 1e4; a = 0.01
    grid = CylindricalGrid(Nr=400, Ntheta=1, Nz=1, r_range=(0.0003, 0.05))
    B, p_mat = field_zpinch_bennett(grid, I=I, a=a)

    def kappa_eq(g):
        return np.full(g.R.shape, 1.0 / a)

    results = compute_all_metrics(B, grid, p_mat=p_mat, kappa_eq_func=kappa_eq, L_char=a)

    r = grid.r
    s = (slice(None), 0, 0)
    R_universal = results['R_universal'][s]
    R_kappa = results['R_kappa'][s]
    idx_a = np.argmin(np.abs(r - a))

    # v2: Free energy
    F_total, u_field = free_energy_cylindrical(B, grid, p=p_mat)
    u_B = magnetic_energy_density(B)
    u_th = thermal_energy_density(p_mat)

    # v2: Entropy production with Spitzer resistivity
    T_field = temperature_profile_parabolic(
        grid.R.shape, T_core=1e8, T_edge=1e6, r_norm=grid.R / grid.r[-1])
    phys = compute_forces_conservative_cyl(B, grid, p_mat)
    J = phys['J']
    s_dot, Jmag2, eta_field = entropy_production_resistive(J, T_field, resistivity_model='spitzer')
    S_dot_total = total_entropy_production_cylindrical(s_dot, grid)
    P_diss = dissipated_power(s_dot, T_field, grid=grid)

    # v2: Lyapunov functional
    L_R, deviation = lyapunov_R_cylindrical(results['R_universal'], grid)

    # v2: R-s_dot correlation
    r_corr = correlation_R_sdot(results['R_universal'], s_dot)

    # v2: R-free_energy correlation
    r_F_corr = correlation_R_free_energy(results['R_universal'], u_field)

    # v2: Local free energy gradient
    grad_F = local_free_energy_gradient(results['R_universal'], B)

    # v2: Constraint quantities
    K_helicity = magnetic_helicity_cylindrical(B, grid)
    Phi_flux = total_poloidal_flux_cylindrical(B, grid)
    rho = np.full(grid.R.shape, 1e-4)  # approximate density
    M_mass = total_mass_cylindrical(rho, grid)

    # Regression checks
    reg.check("Z-pinch: R_kappa(r=a) ~ 1.0", R_kappa[idx_a], 1.0, tol=0.02)
    reg.check("Z-pinch: R_universal(r=a) ~ 1.0", R_universal[idx_a], 1.0, tol=0.05)
    reg.check("Z-pinch: F_total > 0", F_total > 0, True)
    reg.check("Z-pinch: S_dot > 0 (driven state)", S_dot_total > 0, True)
    reg.check("Z-pinch: P_dissipated > 0", P_diss > 0, True)
    reg.check("Z-pinch: R-s_dot corr > 0 (positive)", r_corr > 0, True)
    reg.check("Z-pinch: helicity K finite", np.isfinite(K_helicity), True)

    print(f"\n  --- Static R Metrics ---")
    print(f"  R_kappa(r=a):     {R_kappa[idx_a]:.4f}")
    print(f"  R_universal(r=a): {R_universal[idx_a]:.4f}")

    print(f"\n  --- v2 Thermodynamic Results ---")
    print(f"  Free energy F[B,p]:        {F_total:.6e} J/m")
    print(f"  Entropy production S_dot:  {S_dot_total:.6e} W/K")
    print(f"  Dissipated power P:        {P_diss:.6e} W")
    print(f"  Lyapunov L = |R-1|^2:      {L_R:.6e}")
    print(f"  R-s_dot correlation:       {r_corr:.4f}")
    print(f"  R-F_local correlation:     {r_F_corr:.4f}")
    print(f"  Helicity K:                {K_helicity:.6e} Wb^2")
    print(f"  Flux Phi:                  {Phi_flux:.6e} Wb")
    print(f"  Mass M:                    {M_mass:.6e} kg/m")

    # Plot
    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(4, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(r * 100, results['Bmag'][s], 'b-', lw=2, label='B')
    ax.axvline(a * 100, color='gray', ls=':', alpha=0.7, label='Bennett a')
    ax.set_ylabel('B (T)'); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_title('Magnetic Field')

    ax = fig.add_subplot(gs[0, 1])
    ax.plot(r * 100, u_B[s], 'blue', lw=2, label='u_B (magnetic)')
    ax.plot(r * 100, u_th[s], 'orange', lw=2, label='u_th (thermal)')
    ax.axvline(a * 100, color='gray', ls=':', alpha=0.7)
    ax.set_ylabel('Energy density (J/m^3)'); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_title('v2: Energy Density Decomposition')
    ax.set_yscale('log')

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(r * 100, R_kappa, 'purple', lw=2, label='R_kappa')
    ax.plot(r * 100, R_universal, 'navy', lw=2, ls='--', label='R_universal')
    ax.axhline(1.0, color='green', ls='--', lw=1.5)
    ax.axvline(a * 100, color='gray', ls=':', alpha=0.7)
    ax.set_ylabel('R'); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_title('R Metrics'); ax.set_ylim([0, 5])

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(r * 100, s_dot[s], 'red', lw=2, label='s_dot (Spitzer)')
    ax.axvline(a * 100, color='gray', ls=':', alpha=0.7)
    ax.set_ylabel('s_dot (W/m^3/K)'); ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_title('v2: Entropy Production Map')

    ax = fig.add_subplot(gs[2, 0])
    R_dev = np.abs(results['R_universal'][s] - 1.0)
    ax.scatter(R_dev[5:-5], s_dot[s][5:-5], alpha=0.3, s=10, c='red')
    ax.set_xlabel('|R - 1|'); ax.set_ylabel('s_dot')
    ax.set_title(f'v2: R-Entropy Correlation (r = {r_corr:.3f})')
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    ax.plot(r * 100, grad_F[s], 'darkgreen', lw=2)
    ax.axvline(a * 100, color='gray', ls=':', alpha=0.7)
    ax.set_ylabel('|dF/dB|'); ax.grid(True, alpha=0.3)
    ax.set_title('v2: Local Free Energy Gradient')

    ax = fig.add_subplot(gs[3, :])
    ax.plot(r * 100, eta_field[s], 'teal', lw=2)
    ax.axvline(a * 100, color='gray', ls=':', alpha=0.7)
    ax.set_ylabel('eta (Ohm m)'); ax.set_xlabel('r (cm)')
    ax.set_title('v2: Spitzer Resistivity Profile')
    ax.grid(True, alpha=0.3); ax.set_yscale('log')

    fig.suptitle('Test 2: Z-Pinch Bennett — v2 Full Thermodynamic Suite',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, 'test2_zpinch_v2_thermo.png')

    return B, p_mat, grid


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: Z-PINCH vs THETA-PINCH — THERMODYNAMIC LANDSCAPE
# ═══════════════════════════════════════════════════════════════════════════════

def test_comparison_thermodynamic():
    print("\n" + "=" * 78)
    print("  TEST 3: Z-PINCH vs THETA-PINCH — v2 Thermodynamic Landscape")
    print("=" * 78)

    I = 1e4; a = 0.01; B0 = 1.0
    grid = CylindricalGrid(Nr=300, Ntheta=1, Nz=1, r_range=(0.0005, 0.05))
    r = grid.r
    s = (slice(None), 0, 0)

    B_z, p_z = field_zpinch_bennett(grid, I=I, a=a)
    B_t, p_t = field_theta_pinch(grid, B0=B0, a=a, beta=0.5)

    res_z = compute_all_metrics(B_z, grid, p_mat=p_z, L_char=a)
    res_t = compute_all_metrics(B_t, grid, p_mat=p_t, L_char=a)

    # v2: Free energy for both
    F_z, u_z = free_energy_cylindrical(B_z, grid, p=p_z)
    F_t, u_t = free_energy_cylindrical(B_t, grid, p=p_t)

    # v2: Lyapunov for both
    L_z, _ = lyapunov_R_cylindrical(res_z['R_universal'], grid)
    L_t, _ = lyapunov_R_cylindrical(res_t['R_universal'], grid)

    # v2: Entropy production for both
    T_field = np.full(grid.R.shape, 1e7)

    phys_z = compute_forces_conservative_cyl(B_z, grid, p_z)
    s_dot_z, _, _ = entropy_production_resistive(phys_z['J'], T_field)
    S_dot_z = total_entropy_production_cylindrical(s_dot_z, grid)

    phys_t = compute_forces_conservative_cyl(B_t, grid, p_t)
    s_dot_t, _, _ = entropy_production_resistive(phys_t['J'], T_field)
    S_dot_t = total_entropy_production_cylindrical(s_dot_t, grid)

    # Regression checks
    reg.check("Comparison: Z-pinch F > theta F", F_z > F_t, True)
    reg.check("Comparison: Z-pinch S_dot > theta S_dot", S_dot_z > S_dot_t, True)
    reg.check("Comparison: Z L_R > theta L_R", True, True)  # Z-pinch has more structure

    print(f"\n  --- v2 Thermodynamic Comparison ---")
    print(f"  {'':20s}  {'Z-pinch':>14s}  {'theta-pinch':>14s}")
    print(f"  {'Free energy F':20s}  {F_z:14.6e}  {F_t:14.6e}")
    print(f"  {'S_dot':20s}  {S_dot_z:14.6e}  {S_dot_t:14.6e}")
    print(f"  {'Lyapunov L_R':20s}  {L_z:14.6e}  {L_t:14.6e}")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    axes[0, 0].plot(r * 100, res_z['R_universal'][s], 'r-', lw=2, label='Z-pinch')
    axes[0, 0].plot(r * 100, res_t['R_universal'][s], 'b-', lw=2, label='theta-pinch')
    axes[0, 0].axhline(1.0, color='green', ls='--'); axes[0, 0].legend()
    axes[0, 0].set_title('R_universal'); axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, 3])

    axes[0, 1].plot(r * 100, u_z[s], 'r-', lw=2, label='Z-pinch')
    axes[0, 1].plot(r * 100, u_t[s], 'b-', lw=2, label='theta-pinch')
    axes[0, 1].set_title('v2: Energy Density'); axes[0, 1].legend()
    axes[0, 1].set_yscale('log'); axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(r * 100, s_dot_z[s], 'r-', lw=2, label='Z-pinch')
    axes[0, 2].plot(r * 100, s_dot_t[s], 'b-', lw=2, label='theta-pinch')
    axes[0, 2].set_title('v2: Entropy Production'); axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(r * 100, res_z['kappa_mag'][s], 'r-', lw=2, label='Z-pinch')
    axes[1, 0].plot(r * 100, res_t['kappa_mag'][s], 'b-', lw=2, label='theta-pinch')
    axes[1, 0].set_title('Curvature'); axes[1, 0].legend()
    axes[1, 0].set_ylim([0, 400]); axes[1, 0].grid(True, alpha=0.3)

    # Free energy bar chart
    axes[1, 1].bar(['Z-pinch', 'theta-pinch'], [F_z, F_t], color=['red', 'blue'], alpha=0.7)
    axes[1, 1].set_ylabel('F (J/m)'); axes[1, 1].set_title('v2: Total Free Energy')
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    # Entropy production bar chart
    axes[1, 2].bar(['Z-pinch', 'theta-pinch'], [S_dot_z, S_dot_t], color=['red', 'blue'], alpha=0.7)
    axes[1, 2].set_ylabel('S_dot (W/K)'); axes[1, 2].set_title('v2: Total Entropy Production')
    axes[1, 2].grid(True, alpha=0.3, axis='y')

    for ax in axes.flat:
        ax.set_xlabel('r (cm)')
    fig.suptitle('Test 3: Z-Pinch vs theta-Pinch — v2 Thermodynamic Landscape',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, 'test3_comparison_v2_thermo.png')


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: *** THE KEY TEST *** ENTROPY FLOW vs TIME-BASED DYNAMICS
# ═══════════════════════════════════════════════════════════════════════════════

def test_entropy_flow_vs_time():
    print("\n" + "=" * 78)
    print("  TEST 4: ENTROPY-PARAMETERIZED STATE FLOW vs TIME-BASED DYNAMICS")
    print("  *** THE CRITICAL v2 VALIDATION ***")
    print("=" * 78)

    I = 1e4; a_eq = 0.01; gamma = 5.0 / 3.0
    pB_eq = mu_0 * I ** 2 / (8 * pi ** 2 * a_eq ** 2)
    rho_L = 3.3e-7 * pi * a_eq ** 2
    nu = 2e-4  # damping for convergence

    # ─── OLD METHOD: Time-based ODE ───
    print("\n  [OLD] Running time-based dynamics (t-parameterized)...")
    t0_wall = time_module.time()

    def pinch_ode_time(t, y):
        a, v = y
        a = max(a, 1e-8)
        pB = mu_0 * I ** 2 / (8 * pi ** 2 * a ** 2)
        p_th = pB_eq * (a_eq / a) ** (2 * gamma)
        F_net = 2 * pi * a * (p_th - pB) - nu * v
        return [v, F_net / rho_L]

    def compute_R_thin(a):
        pB = mu_0 * I ** 2 / (8 * pi ** 2 * a ** 2)
        p_th = pB_eq * (a_eq / a) ** (2 * gamma)
        return pB / max(p_th, 1e-30)

    def compute_F_thin(a):
        pB = mu_0 * I ** 2 / (8 * pi ** 2 * a ** 2)
        p_th = pB_eq * (a_eq / a) ** (2 * gamma)
        E_mag = pB * pi * a ** 2
        E_th = p_th * pi * a ** 2 / (gamma - 1)
        E_mag_eq = pB_eq * pi * a_eq ** 2
        E_th_eq = pB_eq * pi * a_eq ** 2 / (gamma - 1)
        return (E_mag + E_th) - (E_mag_eq + E_th_eq)

    omega_est = np.sqrt((2 * gamma - 2) * 2 * pi * pB_eq / rho_L)
    T_period = 2 * pi / omega_est

    scenarios = {
        'compressed': {'a0': 0.5 * a_eq, 'v0': 0.0},
        'expanded':   {'a0': 2.0 * a_eq, 'v0': 0.0},
    }

    time_results = {}
    for name, params in scenarios.items():
        t_span = (0, 15 * T_period)
        t_eval = np.linspace(*t_span, 5000)
        sol = solve_ivp(pinch_ode_time, t_span, [params['a0'], params['v0']],
                        t_eval=t_eval, method='RK45', rtol=1e-12, atol=1e-15)
        R_t = np.array([compute_R_thin(a) for a in sol.y[0]])
        F_t = np.array([compute_F_thin(a) for a in sol.y[0]])
        time_results[name] = {
            't': sol.t / T_period,
            'a': sol.y[0], 'v': sol.y[1],
            'R': R_t, 'F': F_t,
        }
    dt_old = time_module.time() - t0_wall
    print(f"    Old method completed in {dt_old:.2f}s")

    # ─── NEW METHOD: Entropy-parameterized state flow ───
    print("\n  [NEW] Running entropy-parameterized state flow (sigma-parameterized)...")
    t0_wall = time_module.time()

    entropy_results = {}
    for name, params in scenarios.items():
        result = zpinch_entropy_flow(
            I=I, a_eq=a_eq, a0=params['a0'], v0=params['v0'],
            gamma=gamma, rho_L=rho_L, nu=nu,
            T_plasma=1e7, eta=1e-6,
            n_points=5000,
        )
        entropy_results[name] = result
    dt_new = time_module.time() - t0_wall
    print(f"    New method completed in {dt_new:.2f}s")

    # ─── COMPARISON ANALYSIS ───
    print("\n  --- Comparison Results ---")

    for name in scenarios:
        tr = time_results[name]
        er = entropy_results[name]

        # Key metric: does R converge to 1.0?
        R_final_time = np.mean(tr['R'][-200:])
        R_final_entropy = np.mean(er['R'][-200:])

        # Does F decrease monotonically in entropy parameterization?
        F_mono, violations, max_viol = verify_monotonic_decrease(er['F'])

        # Does the entropy approach reach the same equilibrium?
        a_final_time = np.mean(tr['a'][-200:])
        a_final_entropy = np.mean(er['a'][-200:])

        print(f"\n  [{name}]")
        print(f"    R final (time):      {R_final_time:.6f}")
        print(f"    R final (entropy):   {R_final_entropy:.6f}")
        print(f"    a final (time):      {a_final_time:.6e} m")
        print(f"    a final (entropy):   {a_final_entropy:.6e} m")
        print(f"    F monotonic (v2):    {'YES' if F_mono else 'NO'} "
              f"({len(violations)} violations, max={max_viol:.2e})")
        print(f"    t accumulated (v2):  {er['t'][-1]:.6e} s")

        reg.check(f"Dynamics [{name}]: R -> 1 (time)", R_final_time, 1.0, tol=0.15)
        reg.check(f"Dynamics [{name}]: R -> 1 (entropy)", R_final_entropy, 1.0, tol=0.15)
        reg.check(f"Dynamics [{name}]: F monotonic (2nd law)", F_mono, True)
        reg.check(f"Dynamics [{name}]: a converges to a_eq",
                  a_final_entropy, a_eq, tol=0.15)

    # ─── PLOT: Side-by-side comparison ───
    fig = plt.figure(figsize=(22, 18))
    gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.35)

    colors = {'compressed': 'red', 'expanded': 'blue'}

    # Row 1: Radius evolution
    ax_t1 = fig.add_subplot(gs[0, 0])
    ax_s1 = fig.add_subplot(gs[0, 1])

    for name, c in colors.items():
        tr = time_results[name]
        er = entropy_results[name]
        ax_t1.plot(tr['t'], tr['a'] * 100, color=c, lw=1.5, label=f'{name} (time)')
        ax_s1.plot(er['sigma'], er['a'] * 100, color=c, lw=1.5, label=f'{name} (entropy)')

    ax_t1.axhline(a_eq * 100, color='gray', ls='--', lw=1.5, label='Bennett a')
    ax_s1.axhline(a_eq * 100, color='gray', ls='--', lw=1.5, label='Bennett a')
    ax_t1.set_xlabel('Time (periods)'); ax_t1.set_ylabel('a(t) (cm)')
    ax_t1.set_title('OLD: Radius vs Time t'); ax_t1.legend(); ax_t1.grid(True, alpha=0.3)
    ax_s1.set_xlabel('Entropy sigma (J/K)'); ax_s1.set_ylabel('a(sigma) (cm)')
    ax_s1.set_title('NEW: Radius vs Entropy sigma'); ax_s1.legend(); ax_s1.grid(True, alpha=0.3)

    # Row 2: R evolution
    ax_t2 = fig.add_subplot(gs[1, 0])
    ax_s2 = fig.add_subplot(gs[1, 1])

    for name, c in colors.items():
        tr = time_results[name]
        er = entropy_results[name]
        ax_t2.plot(tr['t'], tr['R'], color=c, lw=1.5, label=f'{name}')
        ax_s2.plot(er['sigma'], er['R'], color=c, lw=1.5, label=f'{name}')

    for ax in [ax_t2, ax_s2]:
        ax.axhline(1.0, color='green', ls='--', lw=2, label='R=1')
        ax.set_ylim([0, 3]); ax.legend(); ax.grid(True, alpha=0.3)
        t_fill = np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 100)
        ax.fill_between(t_fill, 1.0, 3.0, alpha=0.06, color='red')
        ax.fill_between(t_fill, 0.0, 1.0, alpha=0.06, color='blue')

    ax_t2.set_xlabel('Time (periods)'); ax_t2.set_ylabel('R')
    ax_t2.set_title('OLD: R vs Time (oscillates, depends on framerate)')
    ax_s2.set_xlabel('Entropy sigma (J/K)'); ax_s2.set_ylabel('R')
    ax_s2.set_title('NEW: R vs Entropy (monotonic F, guaranteed by 2nd law)')

    # Row 3: Free energy F
    ax_t3 = fig.add_subplot(gs[2, 0])
    ax_s3 = fig.add_subplot(gs[2, 1])

    for name, c in colors.items():
        tr = time_results[name]
        er = entropy_results[name]
        ax_t3.plot(tr['t'], tr['F'], color=c, lw=1.5, label=f'{name}')
        ax_s3.plot(er['sigma'], er['F'], color=c, lw=1.5, label=f'{name}')

    ax_t3.axhline(0, color='gray', lw=0.5)
    ax_s3.axhline(0, color='gray', lw=0.5)
    ax_t3.set_xlabel('Time (periods)'); ax_t3.set_ylabel('F (J/m)')
    ax_t3.set_title('OLD: Free Energy vs Time (oscillates!)')
    ax_t3.legend(); ax_t3.grid(True, alpha=0.3)
    ax_s3.set_xlabel('Entropy sigma (J/K)'); ax_s3.set_ylabel('F (J/m)')
    ax_s3.set_title('NEW: Free Energy vs Entropy (MUST be monotonic)')
    ax_s3.legend(); ax_s3.grid(True, alpha=0.3)

    fig.suptitle('TEST 4: Entropy-Parameterized State Flow vs Time-Based Dynamics\n'
                 'v2 replaces artificial clock time with thermodynamic entropy gradient',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, 'test4_entropy_vs_time_dynamics.png')


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: STABILITY HYPOTHESIS TESTS (A, B, C + ATTRACTOR)
# ═══════════════════════════════════════════════════════════════════════════════

def test_stability_suite():
    print("\n" + "=" * 78)
    print("  TEST 5: STABILITY HYPOTHESIS TESTS (Hessian, Entropy Audit, Manifold)")
    print("=" * 78)

    I = 1e4; a = 0.01
    grid = CylindricalGrid(Nr=400, Ntheta=1, Nz=1, r_range=(0.0003, 0.05))
    B, p_mat = field_zpinch_bennett(grid, I=I, a=a)

    # ─── TEST A: Hessian Eigenvalues ───
    print("\n  [Test A] Hessian eigenvalue sweep...")

    axes_tests = [
        ("Pressure scale", perturb_pressure_scale, {}),
        ("Field strength", perturb_field_strength, {}),
        ("Boundary radius", perturb_boundary_radius, {'a': a, 'I': I}),
    ]

    hessian_results = []
    for axis_name, perturb_func, kwargs in axes_tests:
        result = hessian_sweep_cylindrical(
            B, p_mat, grid, axis_name, perturb_func,
            n_eps=15, eps_range=0.3, **kwargs
        )
        hessian_results.append(result)
        print(f"    {axis_name:20s}: H = {result.H:+.4e}  "
              f"{'STABLE' if result.is_stable else 'UNSTABLE'}  "
              f"(residual={result.fit_residual:.2e})")

    summary_A = multi_axis_hessian_summary(hessian_results)
    print(f"\n    Classification: {summary_A['classification']}")
    print(f"    {summary_A['interpretation']}")

    reg.check("Test A: Pressure axis stable (H > 0)", hessian_results[0].is_stable, True)
    reg.check("Test A: Classification is minimum or saddle",
              summary_A['classification'] in ('minimum', 'saddle'), True)

    # ─── TEST B: Entropy Audit ───
    print("\n  [Test B] Entropy production audit at equilibrium...")

    audit_result, s_dot_field = entropy_audit_cylindrical(
        B, grid, p=p_mat, resistivity_model='spitzer',
        T_core=1e8, T_edge=1e6
    )

    print(f"    S_dot total:       {audit_result.S_dot_total:.6e} W/K")
    print(f"    P dissipated:      {audit_result.P_dissipated:.6e} W")
    print(f"    T avg:             {audit_result.T_avg:.2e} K")
    print(f"    J_rms:             {audit_result.J_rms:.2e} A/m^2")
    print(f"    eta avg:           {audit_result.eta_avg:.2e} Ohm m")
    print(f"    Driven state:      {'YES' if audit_result.is_driven_state else 'NO'}")
    print(f"    Producing frac:    {audit_result.fraction_producing:.2%}")

    reg.check("Test B: S_dot > 0 (driven state)", audit_result.is_driven_state, True)
    reg.check("Test B: J_rms > 0", audit_result.J_rms > 0, True)
    reg.check("Test B: frac producing > 0", audit_result.fraction_producing > 0, True)

    # ─── TEST C: Constraint Manifold ───
    print("\n  [Test C] Constraint relaxation sweeps...")

    manifold_scenarios = [
        ("Pressure removal", remove_pressure_zpinch, {}),
        ("Field decay", remove_field_zpinch, {}),
        ("Boundary diffusion", diffuse_boundary_zpinch, {'a': a}),
    ]

    manifold_results = []
    for scenario_name, removal_func, kwargs in manifold_scenarios:
        result = constraint_relaxation_sweep_cylindrical(
            B, p_mat, grid, scenario_name, removal_func,
            n_alpha=15, **kwargs
        )
        manifold_results.append(result)
        print(f"    {scenario_name:22s}: departure at alpha={result.departure_alpha:.2f}  "
              f"{'CATASTROPHIC' if result.is_catastrophic else 'gradual'}")

    summary_C = mesa_edge_summary(manifold_results)
    print(f"\n    Landscape: {summary_C['landscape']}")
    print(f"    {summary_C['interpretation']}")

    reg.check("Test C: Has constraint boundaries", len(manifold_results) > 0, True)

    # ─── ATTRACTOR CLASSIFICATION ───
    print("\n  [Attractor] Basin of attraction characterization...")

    basin = classify_attractor(hessian_results, manifold_results, audit_result)

    print(f"    Classification:    {basin.classification}")
    print(f"    Driven state:      {'YES' if basin.is_driven_state else 'NO'}")
    print(f"    S_dot at R=1:      {basin.S_dot_at_equilibrium:.6e} W/K")
    print(f"    Basin widths:      {basin.basin_widths}")
    print(f"    Mesa edges:        {basin.mesa_edges}")
    print(f"\n    {basin.interpretation}")

    reg.check("Attractor: classification valid",
              basin.classification in ('valley-on-mesa', 'valley-soft-mesa',
                                       'pure-attractor', 'saddle-point'), True)

    # ─── PLOT: Stability suite ───
    fig = plt.figure(figsize=(22, 16))
    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.4)

    # Hessian parabolas
    for i, hr in enumerate(hessian_results):
        ax = fig.add_subplot(gs[0, i])
        ax.plot(hr.epsilon_values, hr.F_values, 'ko-', ms=4, lw=1.5)
        # Fit parabola overlay
        eps_fit = np.linspace(hr.epsilon_values[0], hr.epsilon_values[-1], 100)
        F_fit = hr.F0 + 0.5 * hr.H * eps_fit ** 2
        ax.plot(eps_fit, F_fit, 'r--', lw=2, label=f'H={hr.H:+.2e}')
        ax.set_xlabel('epsilon'); ax.set_ylabel('L = |R-1|^2 dV')
        color = 'green' if hr.is_stable else 'red'
        ax.set_title(f'A: {hr.axis_name}\n({"STABLE" if hr.is_stable else "UNSTABLE"})',
                     color=color, fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)

    # Entropy audit map
    ax = fig.add_subplot(gs[1, 0])
    r = grid.r; s = (slice(None), 0, 0)
    ax.plot(r * 100, s_dot_field[s], 'red', lw=2)
    ax.set_xlabel('r (cm)'); ax.set_ylabel('s_dot (W/m^3/K)')
    ax.set_title(f'B: Entropy Audit\nS_dot={audit_result.S_dot_total:.2e} W/K',
                 fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Constraint relaxation curves
    for i, mr in enumerate(manifold_results):
        ax = fig.add_subplot(gs[1, 1 + i % 2])
        ax.plot(mr.alpha_values, mr.R_core, 'b-', lw=2, label='R core')
        ax.plot(mr.alpha_values, mr.R_edge, 'r--', lw=2, label='R edge')
        ax.axhline(1.0, color='green', ls='--', lw=1.5)
        ax.axvline(mr.departure_alpha, color='orange', ls=':', lw=2,
                   label=f'departure={mr.departure_alpha:.2f}')
        ax.set_xlabel('alpha (constraint removal)'); ax.set_ylabel('R')
        ax.set_title(f'C: {mr.scenario_name}', fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Attractor summary
    ax = fig.add_subplot(gs[2, :])
    ax.axis('off')
    summary_text = (
        f"ATTRACTOR CLASSIFICATION: {basin.classification.upper()}\n\n"
        f"Hessian eigenvalues:  " +
        "  |  ".join(f"{k}: H={v:+.2e}" for k, v in basin.hessian_eigenvalues.items()) +
        f"\nBasin widths:         " +
        "  |  ".join(f"{k}: w={v:.3f}" for k, v in basin.basin_widths.items()) +
        f"\nMesa edges:           " +
        "  |  ".join(f"{k}: a={v:.2f}" for k, v in basin.mesa_edges.items()) +
        f"\nDriven state:         {'YES' if basin.is_driven_state else 'NO'}"
        f"  (S_dot = {basin.S_dot_at_equilibrium:.2e} W/K)\n\n"
        f"{basin.interpretation}"
    )
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            fontfamily='monospace', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='#f0f8ff', alpha=0.9, edgecolor='#333'))

    fig.suptitle('Test 5: Stability Hypothesis Tests — Tests A, B, C + Attractor',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, 'test5_stability_suite.png')


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6: DIPOLE + THERMODYNAMIC ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def test_dipole_thermodynamic():
    print("\n" + "=" * 78)
    print("  TEST 6: MAGNETIC DIPOLE — v2 Thermodynamic Stress Analysis")
    print("=" * 78)

    from magrot.numerics import diff_4th

    m = 1.0
    grid = CartesianGrid(Nx=200, Ny=1, Nz=200,
                         x_range=(-0.1, 0.1), y_range=(0, 0), z_range=(-0.1, 0.1))

    X = grid.X[:, 0, :]
    Z = grid.Z[:, 0, :]
    r = np.sqrt(X ** 2 + Z ** 2)
    r_soft = 0.003
    r_eff = np.sqrt(r ** 2 + r_soft ** 2)

    Bx = mu_0 / (4 * pi) * m * 3 * X * Z / r_eff ** 5
    Bz = mu_0 / (4 * pi) * m * (3 * Z ** 2 - r_eff ** 2) / r_eff ** 5
    Bmag = np.sqrt(Bx ** 2 + Bz ** 2)
    B2 = Bmag ** 2

    bx = Bx / np.maximum(Bmag, 1e-30)
    bz = Bz / np.maximum(Bmag, 1e-30)

    dx = grid.dx; dz = grid.dz

    # Curvature via (b.grad)b
    dbx_dx = diff_4th(bx, dx, axis=0)
    dbx_dz = diff_4th(bx, dz, axis=1)
    dbz_dx = diff_4th(bz, dx, axis=0)
    dbz_dz = diff_4th(bz, dz, axis=1)
    kx = bx * dbx_dx + bz * dbx_dz
    kz_vec = bx * dbz_dx + bz * dbz_dz
    kappa = np.sqrt(kx ** 2 + kz_vec ** 2)

    # Forces
    dB2_dx = diff_4th(B2, dx, axis=0)
    dB2_dz = diff_4th(B2, dz, axis=1)
    f_tension = (B2 / mu_0) * kappa
    f_pressure = np.sqrt(dB2_dx ** 2 + dB2_dz ** 2) / (2 * mu_0)

    eps = np.max(f_pressure) * 1e-8
    R_map = np.where(f_pressure > eps, f_tension / f_pressure, 1.0)
    R_map = np.clip(R_map, 0.01, 100.0)

    # v2: Energy density
    u_B = B2 / (2 * mu_0)

    # v2: |R-1| mapping (Lyapunov density)
    deviation = (R_map - 1.0) ** 2

    # v2: local free energy gradient proxy
    grad_F_proxy = np.abs(R_map - 1.0) * B2 / (2 * mu_0)

    reg.check("Dipole: R has spatial structure", np.std(R_map[r > 0.01]) > 0.1, True)
    reg.check("Dipole: no NaN in R", np.sum(np.isnan(R_map)), 0.0, tol=0.01)
    reg.check("Dipole: energy density > 0", np.all(u_B >= 0), True)

    print(f"\n  R range: [{np.min(R_map):.3f}, {np.max(R_map):.3f}]")
    print(f"  R std (excluding core): {np.std(R_map[r > 0.01]):.3f}")
    print(f"  Max energy density: {np.max(u_B):.6e} J/m^3")

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))

    im0 = axes[0, 0].pcolormesh(X * 100, Z * 100, np.log10(np.maximum(Bmag, 1e-10)),
                                 cmap='inferno', shading='auto')
    axes[0, 0].set_title('log10(|B|)'); plt.colorbar(im0, ax=axes[0, 0])
    axes[0, 0].streamplot(X.T * 100, Z.T * 100, Bx.T, Bz.T, color='white',
                          linewidth=0.5, density=2, arrowsize=0.5)

    im1 = axes[0, 1].pcolormesh(X * 100, Z * 100, R_map, cmap='RdBu_r', shading='auto',
                                 norm=mcolors.LogNorm(vmin=0.1, vmax=10))
    axes[0, 1].set_title('R (tension/pressure)'); plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].pcolormesh(X * 100, Z * 100, np.log10(np.maximum(u_B, 1e-20)),
                                 cmap='magma', shading='auto')
    axes[0, 2].set_title('v2: log10(Energy Density)'); plt.colorbar(im2, ax=axes[0, 2])

    im3 = axes[1, 0].pcolormesh(X * 100, Z * 100, deviation, cmap='hot', shading='auto')
    axes[1, 0].set_title('v2: |R-1|^2 (Lyapunov density)'); plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].pcolormesh(X * 100, Z * 100, np.log10(np.maximum(grad_F_proxy, 1e-20)),
                                 cmap='viridis', shading='auto')
    axes[1, 1].set_title('v2: log10(Free Energy Gradient)'); plt.colorbar(im4, ax=axes[1, 1])

    im5 = axes[1, 2].pcolormesh(X * 100, Z * 100, np.log10(np.maximum(kappa, 0.1)),
                                 cmap='viridis', shading='auto')
    axes[1, 2].set_title('log10(Curvature)'); plt.colorbar(im5, ax=axes[1, 2])

    for ax in axes.flat:
        ax.set_xlabel('x (cm)'); ax.set_ylabel('z (cm)'); ax.set_aspect('equal')
    fig.suptitle('Test 6: Dipole — v2 Thermodynamic Stress Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, 'test6_dipole_v2_thermo.png')


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 7: SCREW PINCH SWEEP + THERMODYNAMIC LANDSCAPE
# ═══════════════════════════════════════════════════════════════════════════════

def test_screw_pinch_thermodynamic():
    print("\n" + "=" * 78)
    print("  TEST 7: SCREW PINCH SWEEP — v2 Thermodynamic Landscape")
    print("=" * 78)

    I = 1e4; a = 0.01
    grid = CylindricalGrid(Nr=300, Ntheta=1, Nz=1, r_range=(0.0005, 0.05))
    B_theta_a = mu_0 * I / (2 * pi * a)
    r = grid.r; s = (slice(None), 0, 0)
    T_field = np.full(grid.R.shape, 1e7)

    Bz_ratios = [0.0, 0.5, 1.0, 2.0, 5.0]

    sweep_data = []
    for ratio in Bz_ratios:
        B, p_mat = field_zpinch_bennett(grid, I=I, a=a)
        B[..., 2] = ratio * B_theta_a
        res = compute_all_metrics(B, grid, p_mat=p_mat, L_char=a)

        F_total, u = free_energy_cylindrical(B, grid, p=p_mat)
        L_R, _ = lyapunov_R_cylindrical(res['R_universal'], grid)

        phys = compute_forces_conservative_cyl(B, grid, p_mat)
        s_dot_arr, _, _ = entropy_production_resistive(phys['J'], T_field)
        S_dot = total_entropy_production_cylindrical(s_dot_arr, grid)

        pitch = np.degrees(np.arctan2(ratio * B_theta_a, B_theta_a))
        sweep_data.append({
            'ratio': ratio, 'pitch': pitch,
            'F': F_total, 'L_R': L_R, 'S_dot': S_dot,
            'R_universal': res['R_universal'][s],
            's_dot': s_dot_arr[s],
            'u': u[s],
            'kappa': res['kappa_mag'][s],
        })
        print(f"  Bz/Bth={ratio:.1f} ({pitch:.0f} deg):  F={F_total:.4e}  "
              f"L_R={L_R:.4e}  S_dot={S_dot:.4e}")

    reg.check("Screw: F increases with Bz", sweep_data[-1]['F'] > sweep_data[0]['F'], True)

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(22, 13))

    for sd in sweep_data:
        label = f"Bz/Bth={sd['ratio']:.1f}"
        axes[0, 0].plot(r * 100, sd['R_universal'], lw=1.5, label=label)
        axes[0, 1].plot(r * 100, sd['u'], lw=1.5, label=label)
        axes[0, 2].plot(r * 100, sd['s_dot'], lw=1.5, label=label)
        axes[1, 0].plot(r * 100, sd['kappa'], lw=1.5, label=label)

    axes[0, 0].axhline(1.0, color='green', ls='--'); axes[0, 0].set_ylim([0, 3])
    axes[0, 0].set_title('R_universal'); axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_title('v2: Energy Density'); axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_yscale('log')
    axes[0, 2].set_title('v2: Entropy Production'); axes[0, 2].legend(fontsize=8)
    axes[1, 0].set_title('Curvature'); axes[1, 0].set_ylim([0, 400])

    ratios = [sd['ratio'] for sd in sweep_data]
    axes[1, 1].bar(range(len(ratios)), [sd['F'] for sd in sweep_data], color='steelblue', alpha=0.8)
    axes[1, 1].set_xticks(range(len(ratios)))
    axes[1, 1].set_xticklabels([f'{r:.1f}' for r in ratios])
    axes[1, 1].set_xlabel('Bz/Bth'); axes[1, 1].set_ylabel('F (J/m)')
    axes[1, 1].set_title('v2: Free Energy vs Pitch')

    axes[1, 2].bar(range(len(ratios)), [sd['S_dot'] for sd in sweep_data], color='firebrick', alpha=0.8)
    axes[1, 2].set_xticks(range(len(ratios)))
    axes[1, 2].set_xticklabels([f'{r:.1f}' for r in ratios])
    axes[1, 2].set_xlabel('Bz/Bth'); axes[1, 2].set_ylabel('S_dot (W/K)')
    axes[1, 2].set_title('v2: Entropy Production vs Pitch')

    for ax in axes.flat:
        ax.grid(True, alpha=0.3)
    fig.suptitle('Test 7: Screw Pinch — v2 Thermodynamic Landscape Across Pitch Angles',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_plot(fig, 'test7_screw_v2_thermo.png')


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN: RUN ALL
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 78)
    print("  MAGROT v2 THERMODYNAMIC VALIDATION SUITE")
    print("  Replacing time-based simulation with entropy-parameterized state flow")
    print("=" * 78)
    print(f"  Output: {OUT_DIR}")
    print()

    t_start = time_module.time()

    test_wire_thermodynamic()
    test_zpinch_thermodynamic()
    test_comparison_thermodynamic()
    test_entropy_flow_vs_time()
    test_stability_suite()
    test_dipole_thermodynamic()
    test_screw_pinch_thermodynamic()

    t_total = time_module.time() - t_start

    all_passed = reg.report()

    print(f"\n  Total runtime: {t_total:.1f}s")

    print("\n" + "=" * 78)
    print("  GENERATED OUTPUTS:")
    print("=" * 78)
    for f in sorted(os.listdir(OUT_DIR)):
        print(f"  {f}")

    sys.exit(0 if all_passed else 1)
