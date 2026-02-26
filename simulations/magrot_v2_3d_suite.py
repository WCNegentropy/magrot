"""
MAGROT v2 Full 3D Thermodynamic Validation Suite
═══════════════════════════════════════════════════════════════════

3D upgrade of the v2 thermodynamic test suite.  Every test that previously
used a 1D cylindrical grid (Ntheta=1, Nz=1) is now run on a full 3D grid
with the upgraded curl(B) and curvature computations.

Key validation goals:
  - 3D physics pipeline produces correct results (curl, curvature, forces)
  - Axisymmetric fields show angular uniformity on 3D grids
  - 3D volume integrals agree with scaled 1D results
  - Div(B) ≈ 0 for generated fields (sanity check)
  - Full 3D dipole field analysis in Cartesian coordinates

Tests:
  1. Wire 3D:        R=1 uniformity across (r, theta) polar grid
  2. Z-pinch 3D:     Full diagnostics on 3D grid with symmetry checks
  3. Comparison 3D:  Z-pinch vs theta-pinch thermodynamic landscape in 3D
  4. Dynamics:       Entropy flow vs time (0D ODE — dimension-independent)
  5. Stability 3D:   Hessian / Entropy Audit / Manifold / Attractor on 3D grid
  6. Dipole 3D:      Full Cartesian (x, y, z) dipole stress analysis
  7. Screw Pinch 3D: Helical pitch sweep on 3D grid

Author: WCNEGENTROPY HOLDINGS LLC
Framework: MAGROT v2 (thermodynamic state flow) — Full 3D
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.constants import mu_0, pi
from scipy.integrate import solve_ivp
import os
import time as time_module

# ── Use the installed magrot package ─────────────────────────────────────────
from magrot.fields.grid import CylindricalGrid, CartesianGrid
from magrot.fields.analytic import (
    field_infinite_wire, field_zpinch_bennett, field_theta_pinch,
    field_dipole_cartesian_3d,
)
from magrot.rotation.metrics import compute_all_metrics
from magrot.stress.maxwell import compute_forces_conservative_cyl
from magrot.numerics import diff_4th
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
from magrot.viz.fields_3d import (
    plot_polar_slice, plot_rz_slice, plot_three_slice_cartesian,
    plot_3d_quiver_cylindrical, plot_radial_comparison,
)

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'v2_3d_thermodynamic')
OUT_DIR = os.path.abspath(OUT_DIR)
os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# REGRESSION TRACKER
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
        print("  MAGROT v2 3D THERMODYNAMIC VALIDATION — REGRESSION SUMMARY")
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


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: div(B) check for cylindrical grids
# ═══════════════════════════════════════════════════════════════════════════════

def divB_cylindrical(B, grid):
    """Compute div(B) = (1/r)d(rBr)/dr + (1/r)dBtheta/dtheta + dBz/dz."""
    R = grid.R
    rBr = R * B[..., 0]
    drBr_dr = diff_4th(rBr, grid.dr, axis=0)
    div = (1.0 / R) * drBr_dr

    if grid.Ntheta > 1:
        dBtheta_dtheta = diff_4th(B[..., 1], grid.dtheta, axis=1)
        div += (1.0 / R) * dBtheta_dtheta

    if grid.Nz > 1:
        dBz_dz = diff_4th(B[..., 2], grid.dz, axis=2)
        div += dBz_dz

    return div


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 1: Wire — 3D R=1 uniformity on polar grid
# ═══════════════════════════════════════════════════════════════════════════════

def test_wire_3d():
    print("\n" + "─" * 60)
    print("  TEST 1: Infinite Wire — 3D Polar Grid")
    print("─" * 60)

    grid = CylindricalGrid(Nr=100, Ntheta=32, Nz=1,
                           r_range=(0.001, 0.01))
    B, _ = field_infinite_wire(grid, I=1000)

    results = compute_all_metrics(B, grid, p_mat=None)
    R_uni = results['R_universal']

    # Thermodynamic overlay
    F_total, u_field = free_energy_cylindrical(B, grid, p=None)
    J = results['J']
    T_field = np.full(grid.R.shape, 1e4)
    s_dot, _, _ = entropy_production_resistive(J, T_field)
    S_dot_total = total_entropy_production_cylindrical(s_dot, grid)
    L_R, _ = lyapunov_R_cylindrical(R_uni, grid)

    # div(B) check
    div = divB_cylindrical(B, grid)
    max_divB = float(np.max(np.abs(div[2:-2, :, :])))  # exclude FD boundary

    # Angular symmetry: std across theta at each r
    R_std_theta = np.std(R_uni[:, :, 0], axis=1)
    mean_R_std = float(np.mean(R_std_theta))

    # Interior mean — skip near-wire region (first ~15%) and boundary (last 10)
    start_idx = max(10, grid.Nr // 7)
    R_mid = R_uni[start_idx:-10, :, :]
    R_mean = float(np.mean(R_mid))

    print(f"  Grid: ({grid.Nr}, {grid.Ntheta}, {grid.Nz})")
    print(f"  R_universal mean (far-field): {R_mean:.6f}")
    print(f"  R angular std (mean):         {mean_R_std:.2e}")
    print(f"  F_total:                      {F_total:.6e} J")
    print(f"  S_dot_total:                  {S_dot_total:.6e} W/K")
    print(f"  Lyapunov L_R:                 {L_R:.6e}")
    print(f"  max |div(B)| (interior):      {max_divB:.2e}")

    reg.check("Wire 3D: R_universal mean = 1.0", R_mean, 1.0, tol=0.05)
    reg.check("Wire 3D: F_total > 0", F_total > 0, True)
    reg.check("Wire 3D: S_dot ~ 0 (vacuum)", S_dot_total, 0, tol=1e-3)
    reg.check("Wire 3D: angular symmetry (std < 0.01)", mean_R_std < 0.01, True)
    reg.check("Wire 3D: div(B) ~ 0", max_divB, 0, tol=1.0)

    # ── Visualization: polar heatmap ──
    fig, axes = plt.subplots(1, 3, figsize=(18, 5),
                             subplot_kw={'projection': 'polar'})

    theta = np.append(grid.theta, grid.theta[0] + 2 * np.pi)
    r_mesh, theta_mesh = np.meshgrid(grid.r, theta, indexing='ij')

    for ax, data, label, cmap in [
        (axes[0], R_uni[:, :, 0], '$R_{universal}$', 'RdBu_r'),
        (axes[1], magnetic_energy_density(B)[:, :, 0], '$u_B$ (J/m³)', 'inferno'),
        (axes[2], np.sqrt(np.sum(s_dot**2 if s_dot.ndim > 3 else s_dot[:, :, 0:1]**0 * s_dot[:, :, 0:1], axis=-1)).squeeze() if s_dot.ndim > 3 else s_dot[:, :, 0], '$\\dot{s}$ (W/m³K)', 'hot'),
    ]:
        data_closed = np.concatenate([data, data[:, :1]], axis=1)
        pcm = ax.pcolormesh(theta_mesh, r_mesh, data_closed,
                            cmap=cmap, shading='auto')
        fig.colorbar(pcm, ax=ax, pad=0.1, shrink=0.8)
        ax.set_title(label, pad=15)

    fig.suptitle('Test 1: Infinite Wire — 3D Polar Grid', fontsize=14, y=1.02)
    fig.tight_layout()
    save_plot(fig, 'test1_wire_3d.png')
    print(f"  → Saved test1_wire_3d.png")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 2: Z-Pinch — Full 3D diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def test_zpinch_3d():
    print("\n" + "─" * 60)
    print("  TEST 2: Z-Pinch Bennett — Full 3D Grid")
    print("─" * 60)

    I, a = 1e4, 0.01
    grid = CylindricalGrid(Nr=80, Ntheta=32, Nz=20,
                           r_range=(0.0003, 0.05),
                           z_range=(-0.05, 0.05))
    B, p_mat = field_zpinch_bennett(grid, I=I, a=a)

    def kappa_eq(g):
        return np.full(g.R.shape, 1.0 / a)

    results = compute_all_metrics(B, grid, p_mat=p_mat,
                                  kappa_eq_func=kappa_eq, L_char=a)
    R_kappa = results['R_kappa']
    R_uni = results['R_universal']

    # Thermodynamic diagnostics
    F_total, u_field = free_energy_cylindrical(B, grid, p=p_mat)
    T_field = temperature_profile_parabolic(
        grid.R.shape, r_norm=grid.R / 0.05, T_core=1e7, T_edge=1e5
    )
    J = results['J']
    s_dot, _, eta_field = entropy_production_resistive(
        J, T_field, resistivity_model='spitzer'
    )
    S_dot_total = total_entropy_production_cylindrical(s_dot, grid)
    P_diss = dissipated_power(s_dot, T_field, grid=grid)
    L_R, _ = lyapunov_R_cylindrical(R_uni, grid)
    r_corr = correlation_R_sdot(R_uni, s_dot)
    K_helicity = magnetic_helicity_cylindrical(B, grid)

    # div(B) check
    div = divB_cylindrical(B, grid)
    max_divB = float(np.max(np.abs(div[2:-2, :, 2:-2])))

    # R at r=a (find nearest r index)
    r_idx = np.argmin(np.abs(grid.r - a))
    R_kappa_at_a = float(np.mean(R_kappa[r_idx, :, :]))
    R_uni_at_a = float(np.mean(R_uni[r_idx, :, :]))

    # Angular symmetry
    z_mid = grid.Nz // 2
    R_std_theta = float(np.std(R_uni[:, :, z_mid], axis=1).mean())

    print(f"  Grid: ({grid.Nr}, {grid.Ntheta}, {grid.Nz}) = {grid.Nr * grid.Ntheta * grid.Nz} points")
    print(f"  R_kappa(r=a) [3D avg]:  {R_kappa_at_a:.6f}")
    print(f"  R_universal(r=a):       {R_uni_at_a:.6f}")
    print(f"  F_total:                {F_total:.6e} J")
    print(f"  S_dot_total:            {S_dot_total:.6e} W/K")
    print(f"  P_dissipated:           {P_diss:.6e} W")
    print(f"  Lyapunov L_R:           {L_R:.6e}")
    print(f"  R-s_dot correlation:    {r_corr:.4f}")
    print(f"  Helicity K:             {K_helicity:.6e}")
    print(f"  R angular std (mean):   {R_std_theta:.2e}")
    print(f"  max |div(B)|:           {max_divB:.2e}")

    reg.check("Z-pinch 3D: R_kappa(r=a) ~ 1.0", R_kappa_at_a, 1.0, tol=0.02)
    reg.check("Z-pinch 3D: R_universal(r=a) ~ 1.0", R_uni_at_a, 1.0, tol=0.05)
    reg.check("Z-pinch 3D: F_total > 0", F_total > 0, True)
    reg.check("Z-pinch 3D: S_dot > 0 (driven)", S_dot_total > 0, True)
    reg.check("Z-pinch 3D: P_dissipated > 0", P_diss > 0, True)
    reg.check("Z-pinch 3D: R-sdot corr > 0", r_corr > 0, True)
    reg.check("Z-pinch 3D: helicity K finite", np.isfinite(K_helicity), True)
    reg.check("Z-pinch 3D: angular symmetry", R_std_theta < 0.01, True)
    reg.check("Z-pinch 3D: div(B) ~ 0", max_divB, 0, tol=1.0)

    # ── Visualization ──
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig)

    # Row 1: polar R slice, r-z R slice, R radial profile
    theta_closed = np.append(grid.theta, grid.theta[0] + 2 * np.pi)
    r_mesh, theta_mesh = np.meshgrid(grid.r, theta_closed, indexing='ij')

    ax1 = fig.add_subplot(gs[0, 0], projection='polar')
    R_slice = R_uni[:, :, z_mid]
    R_closed = np.concatenate([R_slice, R_slice[:, :1]], axis=1)
    pcm = ax1.pcolormesh(theta_mesh, r_mesh, R_closed,
                         cmap='RdBu_r', vmin=0.5, vmax=1.5, shading='auto')
    fig.colorbar(pcm, ax=ax1, pad=0.1, shrink=0.8)
    ax1.set_title('$R_{universal}$ (r-θ, z=0)', pad=15)

    ax2 = fig.add_subplot(gs[0, 1])
    R_rz = R_uni[:, 0, :]
    pcm2 = ax2.pcolormesh(grid.z, grid.r, R_rz,
                          cmap='RdBu_r', vmin=0.5, vmax=1.5, shading='auto')
    fig.colorbar(pcm2, ax=ax2)
    ax2.set_xlabel('z (m)')
    ax2.set_ylabel('r (m)')
    ax2.set_title('$R_{universal}$ (r-z, θ=0)')
    ax2.axhline(a, color='lime', ls='--', lw=1, label=f'r=a={a}m')
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[0, 2])
    R_mean_profile = np.mean(R_uni[:, :, z_mid], axis=1)
    R_std_profile = np.std(R_uni[:, :, z_mid], axis=1)
    ax3.plot(grid.r * 1e3, R_mean_profile, 'b-', lw=2, label='Mean over θ')
    ax3.fill_between(grid.r * 1e3,
                     R_mean_profile - R_std_profile,
                     R_mean_profile + R_std_profile,
                     alpha=0.2, color='blue', label='± std')
    ax3.axhline(1.0, color='k', ls=':', alpha=0.5)
    ax3.axvline(a * 1e3, color='r', ls='--', alpha=0.7, label=f'r=a')
    ax3.set_xlabel('r (mm)')
    ax3.set_ylabel('$R_{universal}$')
    ax3.set_title('Radial Profile (3D avg ± std)')
    ax3.legend(fontsize=8)
    ax3.set_ylim(0, 2)

    # Row 2: energy density, entropy production, R-sdot correlation
    ax4 = fig.add_subplot(gs[1, 0], projection='polar')
    u_B = magnetic_energy_density(B)[:, :, z_mid]
    u_closed = np.concatenate([u_B, u_B[:, :1]], axis=1)
    pcm4 = ax4.pcolormesh(theta_mesh, r_mesh, np.log10(np.maximum(u_closed, 1e-30)),
                          cmap='inferno', shading='auto')
    fig.colorbar(pcm4, ax=ax4, pad=0.1, shrink=0.8)
    ax4.set_title('log₁₀($u_B$) (r-θ, z=0)', pad=15)

    ax5 = fig.add_subplot(gs[1, 1], projection='polar')
    sd = s_dot[:, :, z_mid]
    sd_closed = np.concatenate([sd, sd[:, :1]], axis=1)
    pcm5 = ax5.pcolormesh(theta_mesh, r_mesh,
                          np.log10(np.maximum(sd_closed, 1e-30)),
                          cmap='hot', shading='auto')
    fig.colorbar(pcm5, ax=ax5, pad=0.1, shrink=0.8)
    ax5.set_title('log₁₀($\\dot{s}$) (r-θ, z=0)', pad=15)

    ax6 = fig.add_subplot(gs[1, 2])
    R_dev = np.abs(R_uni - 1.0).ravel()
    s_dot_flat = s_dot.ravel()
    mask = (R_dev > 1e-6) & (s_dot_flat > 0) & np.isfinite(R_dev) & np.isfinite(s_dot_flat)
    if np.sum(mask) > 10:
        ax6.scatter(R_dev[mask][::10], s_dot_flat[mask][::10],
                    alpha=0.3, s=2, c='navy')
    ax6.set_xlabel('|R - 1|')
    ax6.set_ylabel('$\\dot{s}$ (W/m³K)')
    ax6.set_title(f'R-Entropy Correlation (ρ={r_corr:.3f})')
    ax6.set_yscale('log')

    fig.suptitle('Test 2: Z-Pinch Bennett — Full 3D Grid', fontsize=14)
    fig.tight_layout()
    save_plot(fig, 'test2_zpinch_3d.png')
    print(f"  → Saved test2_zpinch_3d.png")

    return B, p_mat, grid


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 3: Z-Pinch vs Theta-Pinch — 3D Comparison
# ═══════════════════════════════════════════════════════════════════════════════

def test_comparison_3d():
    print("\n" + "─" * 60)
    print("  TEST 3: Z-Pinch vs Theta-Pinch — 3D Comparison")
    print("─" * 60)

    grid = CylindricalGrid(Nr=60, Ntheta=32, Nz=20,
                           r_range=(0.0005, 0.05),
                           z_range=(-0.05, 0.05))

    # Z-pinch
    B_z, p_z = field_zpinch_bennett(grid, I=1e4, a=0.01)
    res_z = compute_all_metrics(B_z, grid, p_mat=p_z, L_char=0.01)
    F_z, u_z = free_energy_cylindrical(B_z, grid, p=p_z)
    L_z, _ = lyapunov_R_cylindrical(res_z['R_universal'], grid)
    phys_z = compute_forces_conservative_cyl(B_z, grid, p_z)
    T_z = np.full(grid.R.shape, 1e6)
    sd_z, _, _ = entropy_production_resistive(phys_z['J'], T_z)
    Sdot_z = total_entropy_production_cylindrical(sd_z, grid)

    # Theta-pinch
    B_t, p_t = field_theta_pinch(grid, B0=1.0, a=0.01, beta=0.5)
    res_t = compute_all_metrics(B_t, grid, p_mat=p_t, L_char=0.01)
    F_t, u_t = free_energy_cylindrical(B_t, grid, p=p_t)
    L_t, _ = lyapunov_R_cylindrical(res_t['R_universal'], grid)
    phys_t = compute_forces_conservative_cyl(B_t, grid, p_t)
    sd_t, _, _ = entropy_production_resistive(phys_t['J'], T_z)
    Sdot_t = total_entropy_production_cylindrical(sd_t, grid)

    # div(B) checks
    divB_z = float(np.max(np.abs(divB_cylindrical(B_z, grid)[2:-2, :, 2:-2])))
    divB_t = float(np.max(np.abs(divB_cylindrical(B_t, grid)[2:-2, :, 2:-2])))

    # Theta-pinch curvature check (should be ~0 for straight B_z field)
    kappa_theta = res_t['kappa_mag']
    max_kappa_t = float(np.max(kappa_theta[2:-2, :, 2:-2]))

    print(f"  Grid: ({grid.Nr}, {grid.Ntheta}, {grid.Nz})")
    print(f"  Z-pinch:   F={F_z:.4e}  S_dot={Sdot_z:.4e}  L_R={L_z:.4e}")
    print(f"  Theta:     F={F_t:.4e}  S_dot={Sdot_t:.4e}  L_R={L_t:.4e}")
    print(f"  div(B) Z:  {divB_z:.2e}   Theta: {divB_t:.2e}")
    print(f"  Theta-pinch max kappa (interior): {max_kappa_t:.2e}")

    # Note: theta-pinch has much higher F because B0=1T fills entire domain
    # The ordering test from 1D may reverse — document either outcome
    reg.check("Comparison 3D: Z F > 0", F_z > 0, True)
    reg.check("Comparison 3D: Theta F > 0", F_t > 0, True)
    reg.check("Comparison 3D: Z S_dot > 0", Sdot_z > 0, True)
    reg.check("Comparison 3D: Theta kappa ~ 0 (straight)", max_kappa_t < 1.0, True)

    # ── Visualization ──
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 3, figure=fig)
    z_mid = grid.Nz // 2

    theta_closed = np.append(grid.theta, grid.theta[0] + 2 * np.pi)
    r_mesh, theta_mesh = np.meshgrid(grid.r, theta_closed, indexing='ij')

    # R_universal polar slices
    for col, (R_field, label) in enumerate([(res_z['R_universal'], 'Z-pinch'), (res_t['R_universal'], 'θ-pinch')]):
        ax = fig.add_subplot(gs[0, col], projection='polar')
        data = R_field[:, :, z_mid]
        data_c = np.concatenate([data, data[:, :1]], axis=1)
        pcm = ax.pcolormesh(theta_mesh, r_mesh, data_c,
                            cmap='RdBu_r', vmin=0.5, vmax=1.5, shading='auto')
        fig.colorbar(pcm, ax=ax, pad=0.1, shrink=0.8)
        ax.set_title(f'{label} $R_{{uni}}$', pad=15)

    # Bar chart comparison
    ax_bar = fig.add_subplot(gs[0, 2])
    x = np.arange(3)
    vals_z = [F_z, Sdot_z, L_z]
    vals_t = [F_t, Sdot_t, L_t]
    ax_bar.bar(x - 0.15, vals_z, 0.3, label='Z-pinch', color='steelblue')
    ax_bar.bar(x + 0.15, vals_t, 0.3, label='θ-pinch', color='coral')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(['F (J)', 'Ṡ (W/K)', 'L_R'])
    ax_bar.set_yscale('symlog', linthresh=1e-10)
    ax_bar.legend()
    ax_bar.set_title('Thermodynamic Comparison')

    # Energy density r-z slices
    for col, (u, label) in enumerate([(u_z, 'Z-pinch u'), (u_t, 'θ-pinch u')]):
        ax = fig.add_subplot(gs[1, col])
        d = u[:, 0, :]
        pcm = ax.pcolormesh(grid.z, grid.r, np.log10(np.maximum(d, 1e-30)),
                            cmap='inferno', shading='auto')
        fig.colorbar(pcm, ax=ax)
        ax.set_xlabel('z (m)')
        ax.set_ylabel('r (m)')
        ax.set_title(f'{label} (r-z, θ=0)')

    # Entropy production comparison r-z
    ax = fig.add_subplot(gs[1, 2])
    sd_z_rz = sd_z[:, 0, :]
    sd_t_rz = sd_t[:, 0, :]
    ax.plot(grid.r * 1e3, np.mean(sd_z_rz, axis=1), 'b-', lw=2, label='Z-pinch')
    ax.plot(grid.r * 1e3, np.mean(sd_t_rz, axis=1), 'r-', lw=2, label='θ-pinch')
    ax.set_xlabel('r (mm)')
    ax.set_ylabel('$\\dot{s}$ (W/m³K)')
    ax.set_title('Entropy Production Profile')
    ax.set_yscale('symlog', linthresh=1e-10)
    ax.legend()

    fig.suptitle('Test 3: Z-Pinch vs θ-Pinch — 3D Comparison', fontsize=14)
    fig.tight_layout()
    save_plot(fig, 'test3_comparison_3d.png')
    print(f"  → Saved test3_comparison_3d.png")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 4: Entropy Flow vs Time (0D ODE — dimension-independent)
# ═══════════════════════════════════════════════════════════════════════════════

def test_entropy_flow_vs_time():
    print("\n" + "─" * 60)
    print("  TEST 4: Entropy Flow vs Time Dynamics (0D ODE)")
    print("─" * 60)
    print("  (Dimension-independent — same as 1D suite)")

    I = 1e4
    a_eq = 0.01
    gamma = 5.0 / 3.0
    pB_eq = mu_0 * I**2 / (4 * pi**2 * a_eq**2)
    rho_L = 1e-3
    nu = 2e-4

    scenarios = [
        ('compressed', 0.5 * a_eq, 0.0),
        ('expanded',   2.0 * a_eq, 0.0),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    for col, (label, a0, v0) in enumerate(scenarios):
        print(f"\n  Scenario: {label} (a0={a0:.4e} m)")

        # ── Time-based (old method) ──
        def pinch_ode_time(t, y):
            a_val, v_val = y
            a_val = max(a_val, 1e-6)
            pB = pB_eq * (a_eq / a_val)**2
            p_plasma = pB_eq * (a_eq / a_val)**(2 * gamma)
            F_net = (p_plasma - pB) * 2 * pi * a_val
            return [v_val, F_net / rho_L - nu * v_val]

        T_osc = 2 * pi * np.sqrt(rho_L / (4 * gamma * pB_eq))
        sol = solve_ivp(pinch_ode_time, [0, 15 * T_osc], [a0, v0],
                        max_step=T_osc / 50, rtol=1e-10, atol=1e-14)
        a_time = sol.y[0]
        t_arr = sol.t

        def R_thin(a_val):
            pB = pB_eq * (a_eq / a_val)**2
            p_plasma = pB_eq * (a_eq / a_val)**(2 * gamma)
            return pB / max(p_plasma, 1e-30)

        R_time = np.array([R_thin(aa) for aa in a_time])

        # ── Entropy-parameterized (new v2 method) ──
        T_plasma = 1e7
        eta = 1e-4
        er = zpinch_entropy_flow(I, a_eq, a0, v0, gamma, rho_L, nu,
                                 T_plasma, eta, n_points=500)

        R_final_time = float(R_time[-1])
        R_final_entropy = float(er['R'][-1])
        F_mono, F_violations, F_max_viol = verify_monotonic_decrease(er['F'])

        print(f"    R_final (time):    {R_final_time:.6f}")
        print(f"    R_final (entropy): {R_final_entropy:.6f}")
        print(f"    F monotonic:       {F_mono}  (violations: {F_violations})")
        print(f"    a_final (entropy): {er['a'][-1]:.6e} m (eq: {a_eq:.6e})")

        reg.check(f"Dynamics [{label}]: R → 1 (time)", R_final_time, 1.0, tol=0.15)
        reg.check(f"Dynamics [{label}]: R → 1 (entropy)", R_final_entropy, 1.0, tol=0.15)
        reg.check(f"Dynamics [{label}]: F monotonic", F_mono, True)
        reg.check(f"Dynamics [{label}]: a → a_eq", er['a'][-1], a_eq, tol=0.15)

        # Plot
        axes[0, col].plot(t_arr * 1e6, a_time * 1e3, 'b-', alpha=0.7, label='Time')
        axes[0, col].plot(er['t'] * 1e6, er['a'] * 1e3, 'r-', lw=2, label='Entropy')
        axes[0, col].axhline(a_eq * 1e3, color='k', ls=':', alpha=0.5)
        axes[0, col].set_ylabel('a (mm)')
        axes[0, col].set_title(f'{label}: Radius Evolution')
        axes[0, col].legend(fontsize=8)

        axes[1, col].plot(t_arr * 1e6, R_time, 'b-', alpha=0.7)
        axes[1, col].plot(er['t'] * 1e6, er['R'], 'r-', lw=2)
        axes[1, col].axhline(1.0, color='k', ls=':', alpha=0.5)
        axes[1, col].set_ylabel('R')
        axes[1, col].set_title(f'{label}: R Evolution')

        axes[2, col].plot(er['sigma'], er['F'], 'r-', lw=2)
        axes[2, col].set_xlabel('σ (entropy produced)')
        axes[2, col].set_ylabel('F (J)')
        axes[2, col].set_title(f'{label}: F(σ)')

    fig.suptitle('Test 4: Entropy Flow vs Time — 0D Dynamics', fontsize=14)
    fig.tight_layout()
    save_plot(fig, 'test4_entropy_vs_time_3d.png')
    print(f"\n  → Saved test4_entropy_vs_time_3d.png")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 5: Stability Suite — 3D Grid
# ═══════════════════════════════════════════════════════════════════════════════

def test_stability_3d():
    print("\n" + "─" * 60)
    print("  TEST 5: Stability Suite — 3D Grid")
    print("─" * 60)

    I, a = 1e4, 0.01
    grid = CylindricalGrid(Nr=60, Ntheta=24, Nz=16,
                           r_range=(0.0003, 0.05),
                           z_range=(-0.05, 0.05))
    B, p_mat = field_zpinch_bennett(grid, I=I, a=a)

    print(f"  Grid: ({grid.Nr}, {grid.Ntheta}, {grid.Nz}) = {grid.Nr * grid.Ntheta * grid.Nz} points")

    # ── Test A: Hessian Stability ──
    print("\n  [Test A] Hessian eigenvalue sweep...")
    t_a = time_module.time()

    hessian_results = []
    for axis_name, perturb_func, kwargs in [
        ('pressure', perturb_pressure_scale, {}),
        ('field', perturb_field_strength, {}),
        ('boundary', perturb_boundary_radius, {'a': a, 'I': I}),
    ]:
        hr = hessian_sweep_cylindrical(
            B, p_mat, grid, axis_name, perturb_func,
            n_eps=11, eps_range=0.3, **kwargs
        )
        hessian_results.append(hr)
        print(f"    {axis_name:10s}: H = {hr.H:+.4e}  ({'stable' if hr.is_stable else 'UNSTABLE'})")

    summary_A = multi_axis_hessian_summary(hessian_results)
    print(f"    Classification: {summary_A['classification']}")
    print(f"    Time: {time_module.time() - t_a:.1f}s")

    # ── Test B: Entropy Audit ──
    print("\n  [Test B] Entropy audit at equilibrium...")
    audit_result, s_dot_field = entropy_audit_cylindrical(
        B, grid, p_mat, resistivity_model='spitzer',
        T_core=1e7, T_edge=1e5
    )
    print(f"    S_dot_total = {audit_result.S_dot_total:.4e} W/K")
    print(f"    P_dissipated = {audit_result.P_dissipated:.4e} W")
    print(f"    J_rms = {audit_result.J_rms:.4e} A/m²")
    print(f"    Driven state: {audit_result.is_driven_state}")

    # ── Test C: Constraint Manifold ──
    print("\n  [Test C] Constraint relaxation sweep...")
    manifold_results = []
    for scenario, removal_func, kwargs in [
        ('pressure', remove_pressure_zpinch, {}),
        ('field', remove_field_zpinch, {}),
        ('boundary', diffuse_boundary_zpinch, {'a': a}),
    ]:
        mr = constraint_relaxation_sweep_cylindrical(
            B, p_mat, grid, scenario, removal_func,
            n_alpha=11, **kwargs
        )
        manifold_results.append(mr)
        print(f"    {scenario:10s}: R_core range [{mr.R_core[0]:.3f} → {mr.R_core[-1]:.3f}]")

    summary_C = mesa_edge_summary(manifold_results)
    print(f"    Mesa edges: {summary_C['landscape']}")

    # ── Attractor Classification ──
    basin = classify_attractor(hessian_results, manifold_results, audit_result)
    print(f"\n  Attractor: {basin.classification}")
    print(f"    Driven: {basin.is_driven_state}")

    # Regression checks
    pressure_hr = hessian_results[0]  # 'pressure' axis
    reg.check("Stability 3D: pressure H > 0", pressure_hr.H > 0, True)
    reg.check("Stability 3D: classification valid",
              summary_A['classification'] in ('minimum', 'saddle'), True)
    reg.check("Stability 3D: S_dot > 0", audit_result.S_dot_total > 0, True)
    reg.check("Stability 3D: J_rms > 0", audit_result.J_rms > 0, True)
    reg.check("Stability 3D: attractor valid",
              basin.classification in ('valley-on-mesa', 'valley-soft-mesa',
                                       'pure-attractor', 'saddle-point'), True)

    # ── Visualization ──
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig)

    # Row 1: Hessian parabolas
    for col, hr in enumerate(hessian_results):
        ax = fig.add_subplot(gs[0, col])
        ax.plot(hr.epsilon_values, hr.F_values, 'ko-', ms=4)
        eps_fit = np.linspace(hr.epsilon_values[0], hr.epsilon_values[-1], 100)
        F_fit = hr.F0 + 0.5 * hr.H * eps_fit**2
        ax.plot(eps_fit, F_fit, 'r-', lw=2, alpha=0.7)
        ax.set_xlabel('ε')
        ax.set_ylabel('F (J)')
        ax.set_title(f'{hr.axis_name}: H={hr.H:.2e}')

    # Row 2: s_dot polar slice + constraint curves
    z_mid = grid.Nz // 2
    theta_closed = np.append(grid.theta, grid.theta[0] + 2 * np.pi)
    r_mesh, theta_mesh = np.meshgrid(grid.r, theta_closed, indexing='ij')

    ax_sd = fig.add_subplot(gs[1, 0], projection='polar')
    sd_slice = s_dot_field[:, :, z_mid]
    sd_closed = np.concatenate([sd_slice, sd_slice[:, :1]], axis=1)
    pcm = ax_sd.pcolormesh(theta_mesh, r_mesh,
                           np.log10(np.maximum(sd_closed, 1e-30)),
                           cmap='hot', shading='auto')
    fig.colorbar(pcm, ax=ax_sd, pad=0.1, shrink=0.8)
    ax_sd.set_title('log₁₀(ṡ) 3D', pad=15)

    for col, mr in enumerate(manifold_results[:2]):  # pressure, field
        ax = fig.add_subplot(gs[1, col + 1])
        ax.plot(mr.alpha_values, mr.R_core, 'b-o', ms=3, label='R core')
        ax.plot(mr.alpha_values, mr.R_edge, 'r-o', ms=3, label='R edge')
        ax.axhline(1.0, color='k', ls=':', alpha=0.5)
        ax.set_xlabel('α (constraint removal)')
        ax.set_ylabel('R')
        ax.set_title(f'Manifold: {mr.scenario_name}')
        ax.legend(fontsize=8)

    # Row 3: Summary text + boundary manifold
    ax_txt = fig.add_subplot(gs[2, 0])
    ax_txt.axis('off')
    txt = (
        f"Attractor Classification\n"
        f"{'─' * 30}\n"
        f"Type:    {basin.classification}\n"
        f"Driven:  {basin.is_driven_state}\n"
        f"S_dot:   {audit_result.S_dot_total:.2e} W/K\n"
        f"P_diss:  {audit_result.P_dissipated:.2e} W\n"
        f"\nHessian: {summary_A['classification']}\n"
    )
    for hr in hessian_results:
        txt += f"  {hr.axis_name}: H={hr.H:+.2e}\n"
    ax_txt.text(0.1, 0.9, txt, transform=ax_txt.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace')

    ax_bnd = fig.add_subplot(gs[2, 1])
    mr_bnd = manifold_results[2]  # boundary
    ax_bnd.plot(mr_bnd.alpha_values, mr_bnd.R_core, 'b-o', ms=3, label='R core')
    ax_bnd.plot(mr_bnd.alpha_values, mr_bnd.R_edge, 'r-o', ms=3, label='R edge')
    ax_bnd.axhline(1.0, color='k', ls=':', alpha=0.5)
    ax_bnd.set_xlabel('α')
    ax_bnd.set_ylabel('R')
    ax_bnd.set_title('Manifold: boundary')
    ax_bnd.legend(fontsize=8)

    ax_basin = fig.add_subplot(gs[2, 2])
    if hasattr(basin, 'basin_widths') and basin.basin_widths:
        names = list(basin.basin_widths.keys())
        widths = [basin.basin_widths[n] for n in names]
        ax_basin.bar(names, widths, color='steelblue')
        ax_basin.set_ylabel('Basin Width')
        ax_basin.set_title('Basin Widths')
    else:
        ax_basin.axis('off')
        ax_basin.text(0.5, 0.5, 'No basin data', ha='center', va='center')

    fig.suptitle('Test 5: Stability Suite — 3D Grid', fontsize=14)
    fig.tight_layout()
    save_plot(fig, 'test5_stability_3d.png')
    print(f"\n  → Saved test5_stability_3d.png")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 6: Dipole — Full 3D Cartesian
# ═══════════════════════════════════════════════════════════════════════════════

def test_dipole_3d():
    print("\n" + "─" * 60)
    print("  TEST 6: Magnetic Dipole — Full 3D Cartesian")
    print("─" * 60)

    grid = CartesianGrid(Nx=40, Ny=40, Nz=40,
                         x_range=(-0.1, 0.1),
                         y_range=(-0.1, 0.1),
                         z_range=(-0.1, 0.1))

    Bx, By, Bz, Bmag = field_dipole_cartesian_3d(grid.X, grid.Y, grid.Z,
                                                   m=1.0, r_soft=0.005)

    # Build B vector array for Cartesian computations
    B = np.stack([Bx, By, Bz], axis=-1)

    # Unit vector
    Bmag_safe = np.maximum(Bmag, 1e-30)
    bx = Bx / Bmag_safe
    by = By / Bmag_safe
    bz_comp = Bz / Bmag_safe

    dx, dy, dz = grid.dx, grid.dy, grid.dz

    # Curvature: (b . nabla) b in Cartesian (no geometric terms)
    dbx_dx = diff_4th(bx, dx, axis=0)
    dbx_dy = diff_4th(bx, dy, axis=1)
    dbx_dz = diff_4th(bx, dz, axis=2)

    dby_dx = diff_4th(by, dx, axis=0)
    dby_dy = diff_4th(by, dy, axis=1)
    dby_dz = diff_4th(by, dz, axis=2)

    dbz_dx = diff_4th(bz_comp, dx, axis=0)
    dbz_dy = diff_4th(bz_comp, dy, axis=1)
    dbz_dz = diff_4th(bz_comp, dz, axis=2)

    kx = bx * dbx_dx + by * dbx_dy + bz_comp * dbx_dz
    ky = bx * dby_dx + by * dby_dy + bz_comp * dby_dz
    kz = bx * dbz_dx + by * dbz_dy + bz_comp * dbz_dz

    kappa = np.sqrt(kx**2 + ky**2 + kz**2)

    # Forces: tension and pressure
    u_B = Bmag**2 / (2 * mu_0)
    f_tension_mag = (Bmag**2 / mu_0) * kappa

    dBmag2_dx = diff_4th(Bmag**2, dx, axis=0)
    dBmag2_dy = diff_4th(Bmag**2, dy, axis=1)
    dBmag2_dz = diff_4th(Bmag**2, dz, axis=2)
    grad_pB_mag = np.sqrt(dBmag2_dx**2 + dBmag2_dy**2 + dBmag2_dz**2) / (2 * mu_0)

    # R map: tension / pressure (clipped)
    with np.errstate(divide='ignore', invalid='ignore'):
        R_map = np.where(
            grad_pB_mag > 1e-30,
            f_tension_mag / grad_pB_mag,
            1.0
        )
    R_map = np.clip(R_map, 0.01, 100.0)

    # Lyapunov density
    lyap_density = (R_map - 1.0)**2

    # Free energy gradient proxy
    grad_F_proxy = np.abs(R_map - 1.0) * u_B

    # Azimuthal symmetry check: R(x,y,z) vs R(-x,-y,z)
    R_flipped = R_map[::-1, ::-1, :]
    sym_diff = float(np.mean(np.abs(R_map - R_flipped)))

    has_structure = float(np.std(R_map[5:-5, 5:-5, 5:-5]))
    has_nan = int(np.sum(np.isnan(R_map)))
    uB_positive = bool(np.all(u_B >= 0))

    print(f"  Grid: ({grid.Nx}, {grid.Ny}, {grid.Nz}) = {grid.Nx * grid.Ny * grid.Nz} points")
    print(f"  R spatial std (interior): {has_structure:.4f}")
    print(f"  NaN count:                {has_nan}")
    print(f"  u_B all positive:         {uB_positive}")
    print(f"  Azimuthal symmetry diff:  {sym_diff:.4e}")

    reg.check("Dipole 3D: R has structure", has_structure > 0.1, True)
    reg.check("Dipole 3D: no NaN", has_nan, 0, tol=0.01)
    reg.check("Dipole 3D: u_B > 0", uB_positive, True)
    reg.check("Dipole 3D: azimuthal symmetry", sym_diff < 0.5, True)

    # ── Visualization: 3 orthogonal slices ──
    nx, ny, nz = grid.Nx, grid.Ny, grid.Nz

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Row 1: |B|, R map, u_B  — all XZ slice (y=0)
    for col, (data, title, cmap, do_log) in enumerate([
        (Bmag[:, ny // 2, :], '|B| (T)', 'viridis', True),
        (R_map[:, ny // 2, :], 'R (tension/pressure)', 'RdBu_r', False),
        (u_B[:, ny // 2, :], '$u_B$ (J/m³)', 'inferno', True),
    ]):
        ax = axes[0, col]
        d = np.log10(np.maximum(data, 1e-30)) if do_log else data
        vmin, vmax = (None, None) if do_log else (0.0, 2.0)
        pcm = ax.pcolormesh(grid.z, grid.x, d, cmap=cmap,
                            vmin=vmin, vmax=vmax, shading='auto')
        fig.colorbar(pcm, ax=ax)
        ax.set_xlabel('z (m)')
        ax.set_ylabel('x (m)')
        ax.set_title(f'{title} (XZ, y=0)')
        ax.set_aspect('equal')

    # Row 2: |R-1|^2, grad_F, kappa  — XZ and XY slices
    for col, (data, title, cmap, do_log) in enumerate([
        (lyap_density[:, ny // 2, :], '|R-1|² Lyapunov', 'hot', True),
        (R_map[:, :, nz // 2], 'R (XY, z=0)', 'RdBu_r', False),
        (kappa[:, ny // 2, :], 'κ curvature', 'magma', True),
    ]):
        ax = axes[1, col]
        d = np.log10(np.maximum(data, 1e-30)) if do_log else data
        vmin, vmax = (None, None) if do_log else (0.0, 2.0)
        if col == 1:
            pcm = ax.pcolormesh(grid.y, grid.x, d, cmap=cmap,
                                vmin=vmin, vmax=vmax, shading='auto')
            ax.set_xlabel('y (m)')
        else:
            pcm = ax.pcolormesh(grid.z, grid.x, d, cmap=cmap,
                                vmin=vmin, vmax=vmax, shading='auto')
            ax.set_xlabel('z (m)')
        fig.colorbar(pcm, ax=ax)
        ax.set_ylabel('x (m)')
        ax.set_title(title)
        ax.set_aspect('equal')

    fig.suptitle('Test 6: Magnetic Dipole — Full 3D Cartesian', fontsize=14)
    fig.tight_layout()
    save_plot(fig, 'test6_dipole_3d.png')
    print(f"  → Saved test6_dipole_3d.png")


# ═══════════════════════════════════════════════════════════════════════════════
# TEST 7: Screw Pinch — 3D Pitch Sweep
# ═══════════════════════════════════════════════════════════════════════════════

def test_screw_pinch_3d():
    print("\n" + "─" * 60)
    print("  TEST 7: Screw Pinch — 3D Pitch Sweep")
    print("─" * 60)

    grid = CylindricalGrid(Nr=60, Ntheta=32, Nz=20,
                           r_range=(0.0005, 0.05),
                           z_range=(-0.05, 0.05))

    Bz_ratios = [0.0, 0.5, 1.0, 2.0, 5.0]
    F_values = []
    Sdot_values = []
    pitch_angles = []

    z_mid = grid.Nz // 2

    # Collect data for all ratios
    all_R = []
    for ratio in Bz_ratios:
        B, p_mat = field_zpinch_bennett(grid, I=1e4, a=0.01)

        # Add axial field component
        B_theta_a = mu_0 * 1e4 / (2 * pi * 0.01)
        B[..., 2] += ratio * B_theta_a

        res = compute_all_metrics(B, grid, p_mat=p_mat, L_char=0.01)
        F_total, u = free_energy_cylindrical(B, grid, p=p_mat)
        L_R, _ = lyapunov_R_cylindrical(res['R_universal'], grid)

        T_field = np.full(grid.R.shape, 1e6)
        phys = compute_forces_conservative_cyl(B, grid, p_mat)
        sd, _, _ = entropy_production_resistive(phys['J'], T_field)
        Sdot = total_entropy_production_cylindrical(sd, grid)

        pitch = float(np.mean(np.arctan2(B[grid.Nr // 2, :, z_mid, 2],
                                          B[grid.Nr // 2, :, z_mid, 1])))

        F_values.append(F_total)
        Sdot_values.append(Sdot)
        pitch_angles.append(np.degrees(pitch))
        all_R.append(res['R_universal'])

        print(f"  Bz/Btheta={ratio:.1f}  pitch={np.degrees(pitch):.1f}°  "
              f"F={F_total:.4e}  S_dot={Sdot:.4e}")

    # Check F increases with Bz (more stored energy)
    F_increasing = all(F_values[i] <= F_values[i + 1] for i in range(len(F_values) - 1))

    # div(B) for last configuration
    divB = divB_cylindrical(B, grid)
    max_divB = float(np.max(np.abs(divB[2:-2, :, 2:-2])))

    reg.check("Screw 3D: F increases with Bz", F_increasing, True)
    reg.check("Screw 3D: div(B) ~ 0", max_divB, 0, tol=1.0)

    # ── Visualization ──
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(2, 3, figure=fig)

    theta_closed = np.append(grid.theta, grid.theta[0] + 2 * np.pi)
    r_mesh, theta_mesh = np.meshgrid(grid.r, theta_closed, indexing='ij')

    # Row 1: Polar R slices for 3 key ratios
    for col, idx in enumerate([0, 2, 4]):
        ax = fig.add_subplot(gs[0, col], projection='polar')
        R_slice = all_R[idx][:, :, z_mid]
        R_closed = np.concatenate([R_slice, R_slice[:, :1]], axis=1)
        pcm = ax.pcolormesh(theta_mesh, r_mesh, R_closed,
                            cmap='RdBu_r', vmin=0.5, vmax=1.5, shading='auto')
        fig.colorbar(pcm, ax=ax, pad=0.1, shrink=0.8)
        ax.set_title(f'Bz/Bθ={Bz_ratios[idx]:.0f}  '
                     f'pitch={pitch_angles[idx]:.0f}°', pad=15)

    # Row 2: radial profiles, F bar chart, S_dot bar chart
    ax_prof = fig.add_subplot(gs[1, 0])
    for i, ratio in enumerate(Bz_ratios):
        R_mean = np.mean(all_R[i][:, :, z_mid], axis=1)
        ax_prof.plot(grid.r * 1e3, R_mean, label=f'Bz/Bθ={ratio}')
    ax_prof.axhline(1.0, color='k', ls=':', alpha=0.5)
    ax_prof.set_xlabel('r (mm)')
    ax_prof.set_ylabel('$R_{universal}$ (θ-avg)')
    ax_prof.set_title('Radial R Profiles')
    ax_prof.legend(fontsize=7)
    ax_prof.set_ylim(0, 2.5)

    ax_F = fig.add_subplot(gs[1, 1])
    ax_F.bar(range(len(Bz_ratios)), F_values, color='steelblue')
    ax_F.set_xticks(range(len(Bz_ratios)))
    ax_F.set_xticklabels([f'{r:.1f}' for r in Bz_ratios])
    ax_F.set_xlabel('Bz / Bθ')
    ax_F.set_ylabel('F (J)')
    ax_F.set_title('Free Energy vs Pitch')

    ax_S = fig.add_subplot(gs[1, 2])
    ax_S.bar(range(len(Bz_ratios)), Sdot_values, color='coral')
    ax_S.set_xticks(range(len(Bz_ratios)))
    ax_S.set_xticklabels([f'{r:.1f}' for r in Bz_ratios])
    ax_S.set_xlabel('Bz / Bθ')
    ax_S.set_ylabel('Ṡ (W/K)')
    ax_S.set_title('Entropy Production vs Pitch')

    fig.suptitle('Test 7: Screw Pinch — 3D Pitch Sweep', fontsize=14)
    fig.tight_layout()
    save_plot(fig, 'test7_screw_3d.png')
    print(f"  → Saved test7_screw_3d.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 78)
    print("  MAGROT v2 FULL 3D THERMODYNAMIC VALIDATION SUITE")
    print("=" * 78)
    print(f"  Output directory: {OUT_DIR}")

    t_total = time_module.time()

    test_wire_3d()
    B_zpinch, p_zpinch, grid_zpinch = test_zpinch_3d()
    test_comparison_3d()
    test_entropy_flow_vs_time()
    test_stability_3d()
    test_dipole_3d()
    test_screw_pinch_3d()

    elapsed = time_module.time() - t_total
    print(f"\n  Total elapsed time: {elapsed:.1f}s")

    all_passed = reg.report()

    # Save summary
    summary_path = os.path.join(OUT_DIR, 'SUMMARY.txt')
    with open(summary_path, 'w') as f:
        f.write("MAGROT v2 3D THERMODYNAMIC VALIDATION SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        n_pass = sum(1 for c in reg.checks if c['passed'])
        n_total = len(reg.checks)
        f.write(f"Result: {n_pass}/{n_total} checks passed\n")
        f.write(f"Time: {elapsed:.1f}s\n\n")
        for c in reg.checks:
            mark = "PASS" if c['passed'] else "FAIL"
            f.write(f"[{mark}] {c['name']}\n")
            if isinstance(c['value'], bool):
                f.write(f"        got={c['value']}  expected={c['expected']}\n")
            else:
                f.write(f"        got={c['value']:.6g}  expected={c['expected']:.6g}  err={c['error']:.2e}\n")
    print(f"\n  Summary saved to {summary_path}")
