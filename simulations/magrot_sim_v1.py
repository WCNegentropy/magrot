"""
MagRot MVP Simulator — Rotational-Vector Framework for Magnetic Field Dynamics
Implements all four ℛ metrics and runs Tier 1 analytic validation tests.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Literal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
from scipy.constants import mu_0, epsilon_0, c

# =============================================================================
# 1. GRID AND DATA STRUCTURES
# =============================================================================

@dataclass
class CylindricalGrid:
    """Cylindrical grid (r, θ, z) for axisymmetric configurations."""
    Nr: int
    Ntheta: int
    Nz: int
    r_range: tuple
    theta_range: tuple = (0, 2 * np.pi)
    z_range: tuple = (-1.0, 1.0)
    
    def __post_init__(self):
        # Avoid r=0 singularity
        self.r = np.linspace(max(self.r_range[0], 1e-6), self.r_range[1], self.Nr)
        self.theta = np.linspace(self.theta_range[0], self.theta_range[1], self.Ntheta, endpoint=False)
        self.z = np.linspace(self.z_range[0], self.z_range[1], self.Nz)
        self.dr = self.r[1] - self.r[0] if self.Nr > 1 else 1.0
        self.R, self.THETA, self.Z = np.meshgrid(self.r, self.theta, self.z, indexing='ij')

@dataclass 
class CartesianGrid:
    """Cartesian grid (x, y, z)."""
    Nx: int
    Ny: int
    Nz: int
    x_range: tuple
    y_range: tuple
    z_range: tuple
    
    def __post_init__(self):
        self.x = np.linspace(self.x_range[0], self.x_range[1], self.Nx)
        self.y = np.linspace(self.y_range[0], self.y_range[1], self.Ny)
        self.z = np.linspace(self.z_range[0], self.z_range[1], self.Nz)
        self.dx = self.x[1] - self.x[0] if self.Nx > 1 else 1.0
        self.dy = self.y[1] - self.y[0] if self.Ny > 1 else 1.0
        self.dz = self.z[1] - self.z[0] if self.Nz > 1 else 1.0
        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing='ij')


# =============================================================================
# 2. ANALYTIC FIELD GENERATORS
# =============================================================================

def field_infinite_wire(grid: CylindricalGrid, I: float = 1.0):
    """
    B field of an infinite straight wire carrying current I along z-axis.
    B_θ = μ₀I/(2πr), B_r = B_z = 0.
    """
    B = np.zeros((*grid.R.shape, 3))  # (r, θ, z) components
    B[..., 1] = mu_0 * I / (2 * np.pi * grid.R)  # B_θ
    return B


def field_zpinch_bennett(grid: CylindricalGrid, I: float = 1e4, a: float = 0.01,
                          n0: float = 1e20, kT: float = 1e-16):
    """
    Bennett Z-pinch equilibrium.
    B_θ(r) = μ₀I/(2πa²) * r  for r < a  (uniform current density)
    B_θ(r) = μ₀I/(2πr)       for r >= a
    
    Material pressure: p(r) = p0 * (1 - (r/a)²) for r < a, 0 for r >= a
    Bennett relation: p0 = μ₀I²/(4π²a⁴) * a²/4 ... simplified.
    We use p(r) from radial force balance.
    """
    B = np.zeros((*grid.R.shape, 3))
    p_mat = np.zeros(grid.R.shape)
    
    inside = grid.R < a
    outside = ~inside
    
    # Magnetic field
    B[..., 1] = np.where(inside,
                          mu_0 * I * grid.R / (2 * np.pi * a**2),
                          mu_0 * I / (2 * np.pi * grid.R))
    
    # Material pressure from radial equilibrium: dp/dr = -J_z * B_θ
    # For uniform current density J_z = I/(πa²):
    # Inside: p(r) = (μ₀I²)/(4π²a⁴) * (a² - r²)/2
    p0 = mu_0 * I**2 / (8 * np.pi**2 * a**2)
    p_mat = np.where(inside, p0 * (1 - (grid.R / a)**2), 0.0)
    
    return B, p_mat


def field_theta_pinch(grid: CylindricalGrid, B0: float = 1.0, a: float = 0.01,
                       beta: float = 0.5):
    """
    θ-pinch: axial field B_z with plasma pressure.
    B_z(r) = B0 for r >= a
    B_z(r) = B0 * sqrt(1 - beta) for r < a  (reduced by plasma diamagnetic effect)
    No azimuthal field. Field lines are straight.
    """
    B = np.zeros((*grid.R.shape, 3))
    p_mat = np.zeros(grid.R.shape)
    
    inside = grid.R < a
    B_inside = B0 * np.sqrt(1 - beta)
    
    B[..., 2] = np.where(inside, B_inside, B0)  # B_z
    
    # Pressure balance: p + B²/(2μ₀) = B0²/(2μ₀)
    p_mat = np.where(inside, (B0**2 - B_inside**2) / (2 * mu_0), 0.0)
    
    return B, p_mat


# =============================================================================
# 3. GEOMETRY: CURVATURE CALCULATION
# =============================================================================

def compute_curvature_cylindrical(B, grid: CylindricalGrid):
    """
    Compute field-line curvature vector κ = (b·∇)b in cylindrical coordinates.
    For axisymmetric fields (∂/∂θ = 0, ∂/∂z = 0 for uniform z):
    
    If B = (B_r, B_θ, B_z), b = B/|B|, then:
    κ = (b·∇)b has specific cylindrical expressions.
    
    For purely azimuthal field (B_r=0, B_z=0):
      κ_r = -b_θ²/r (centripetal curvature inward)
      κ_θ = 0
      κ_z = 0
    """
    Bmag = np.sqrt(np.sum(B**2, axis=-1))
    Bmag = np.maximum(Bmag, 1e-30)  # prevent division by zero
    
    b = B / Bmag[..., np.newaxis]  # unit vector
    b_r = b[..., 0]
    b_theta = b[..., 1]
    b_z = b[..., 2]
    
    # For axisymmetric fields, (b·∇)b in cylindrical coords:
    # κ_r = b_r ∂b_r/∂r + (b_θ/r)(∂b_r/∂θ - b_θ) + b_z ∂b_r/∂z
    # κ_θ = b_r ∂b_θ/∂r + (b_θ/r)(∂b_θ/∂θ + b_r) + b_z ∂b_θ/∂z
    # κ_z = b_r ∂b_z/∂r + (b_θ/r)∂b_z/∂θ + b_z ∂b_z/∂z
    
    # Use 4th-order central differences for radial derivatives
    db_r_dr = np.zeros_like(b_r)
    db_theta_dr = np.zeros_like(b_theta)
    db_z_dr = np.zeros_like(b_z)
    
    dr = grid.dr
    Nr = grid.Nr
    
    # 4th order central difference: interior points
    for arr, darr in [(b_r, db_r_dr), (b_theta, db_theta_dr), (b_z, db_z_dr)]:
        if Nr >= 5:
            darr[2:-2] = (-arr[4:] + 8*arr[3:-1] - 8*arr[1:-3] + arr[:-4]) / (12 * dr)
            # 2nd order at boundaries
            darr[0] = (arr[1] - arr[0]) / dr
            darr[1] = (arr[2] - arr[0]) / (2 * dr)
            darr[-1] = (arr[-1] - arr[-2]) / dr
            darr[-2] = (arr[-1] - arr[-3]) / (2 * dr)
        else:
            # fallback: 2nd order
            darr[1:-1] = (arr[2:] - arr[:-2]) / (2 * dr)
            darr[0] = (arr[1] - arr[0]) / dr
            darr[-1] = (arr[-1] - arr[-2]) / dr
    
    # Axisymmetric: ∂/∂θ = 0, ∂/∂z = 0
    R = grid.R
    
    kappa = np.zeros_like(B)
    kappa[..., 0] = b_r * db_r_dr - b_theta**2 / R  # κ_r
    kappa[..., 1] = b_r * db_theta_dr + b_r * b_theta / R  # κ_θ
    kappa[..., 2] = b_r * db_z_dr  # κ_z
    
    kappa_mag = np.sqrt(np.sum(kappa**2, axis=-1))
    
    return kappa, kappa_mag


# =============================================================================
# 4. STRESS TENSOR AND FORCE DECOMPOSITION
# =============================================================================

def compute_magnetic_forces_cylindrical(B, grid: CylindricalGrid):
    """
    Compute magnetic tension and pressure gradient forces in cylindrical coords.
    
    f_tension = (B²/μ₀) * κ
    f_pressure_grad = -∇(B²/(2μ₀))  [note: outward-pointing when B decreases outward]
    f_mag = f_tension + f_pressure_grad = (1/μ₀)[(B·∇)B - ∇(B²/2)]
    """
    Bmag = np.sqrt(np.sum(B**2, axis=-1))
    Bmag = np.maximum(Bmag, 1e-30)
    
    # Curvature
    kappa_vec, kappa_mag = compute_curvature_cylindrical(B, grid)
    
    # Tension force: (B²/μ₀) * κ
    f_tension = (Bmag**2 / mu_0)[..., np.newaxis] * kappa_vec
    
    # Pressure gradient: -∇(B²/(2μ₀))
    # For axisymmetric: only radial component ∂/∂r
    B2 = Bmag**2
    dB2_dr = np.zeros_like(B2)
    dr = grid.dr
    Nr = grid.Nr
    
    if Nr >= 5:
        dB2_dr[2:-2] = (-B2[4:] + 8*B2[3:-1] - 8*B2[1:-3] + B2[:-4]) / (12 * dr)
        dB2_dr[0] = (B2[1] - B2[0]) / dr
        dB2_dr[1] = (B2[2] - B2[0]) / (2 * dr)
        dB2_dr[-1] = (B2[-1] - B2[-2]) / dr
        dB2_dr[-2] = (B2[-1] - B2[-3]) / (2 * dr)
    else:
        dB2_dr[1:-1] = (B2[2:] - B2[:-2]) / (2 * dr)
        dB2_dr[0] = (B2[1] - B2[0]) / dr
        dB2_dr[-1] = (B2[-1] - B2[-2]) / dr
    
    f_pressure = np.zeros_like(B)
    f_pressure[..., 0] = -dB2_dr / (2 * mu_0)  # radial component
    
    # Total magnetic force
    f_total = f_tension + f_pressure
    
    return f_tension, f_pressure, f_total, kappa_vec, kappa_mag


# =============================================================================
# 5. ROTATIONAL METRICS
# =============================================================================

def compute_all_metrics(B, grid: CylindricalGrid, p_mat=None, 
                         kappa_eq_func=None, L0=None):
    """
    Compute all four ℛ metrics for a cylindrical field configuration.
    
    Returns dict with:
      R_kappa     - Metric A: curvature ratio
      R_collapse  - Metric B: collapse indicator  
      R_misalign  - Metric C: current-field misalignment
      R_universal - Universal: |f_inward| / |f_outward|
    Plus auxiliary data.
    """
    f_tension, f_pressure, f_total, kappa_vec, kappa_mag = \
        compute_magnetic_forces_cylindrical(B, grid)
    
    Bmag = np.sqrt(np.sum(B**2, axis=-1))
    Bmag = np.maximum(Bmag, 1e-30)
    
    results = {
        'kappa_mag': kappa_mag,
        'kappa_vec': kappa_vec,
        'f_tension': f_tension,
        'f_pressure': f_pressure,
        'f_total': f_total,
        'Bmag': Bmag,
    }
    
    # --- Metric A: Curvature Ratio ℛ_κ ---
    if kappa_eq_func is not None:
        kappa_eq = kappa_eq_func(grid)
        kappa_eq = np.maximum(kappa_eq, 1e-30)
        results['R_kappa'] = kappa_mag / kappa_eq
    else:
        results['R_kappa'] = None
    
    # --- Metric B: Collapse Indicator ℛ_C ---
    # C(x) = (f_mag - ∇p_mat) · n̂ / f0
    # n̂ = curvature normal direction (= -r̂ for azimuthal fields)
    # Use radial component as proxy for projection onto curvature normal
    
    f_net_r = f_total[..., 0].copy()  # radial component of magnetic force
    if p_mat is not None:
        # Subtract material pressure gradient (radial)
        dp_dr = np.zeros_like(p_mat)
        dr = grid.dr
        Nr = grid.Nr
        if Nr >= 5:
            dp_dr[2:-2] = (-p_mat[4:] + 8*p_mat[3:-1] - 8*p_mat[1:-3] + p_mat[:-4]) / (12 * dr)
            dp_dr[0] = (p_mat[1] - p_mat[0]) / dr
            dp_dr[1] = (p_mat[2] - p_mat[0]) / (2 * dr)
            dp_dr[-1] = (p_mat[-1] - p_mat[-2]) / dr
            dp_dr[-2] = (p_mat[-1] - p_mat[-3]) / (2 * dr)
        else:
            dp_dr[1:-1] = (p_mat[2:] - p_mat[:-2]) / (2 * dr)
            dp_dr[0] = (p_mat[1] - p_mat[0]) / dr
            dp_dr[-1] = (p_mat[-1] - p_mat[-2]) / dr
        f_net_r -= dp_dr  # outward pressure gradient reduces net inward force
    
    if L0 is None:
        L0_val = grid.r[-1] - grid.r[0]
    else:
        L0_val = L0
    
    f0 = Bmag**2 / (2 * mu_0 * L0_val)
    f0 = np.maximum(f0, 1e-30)
    
    C = f_net_r / f0  # negative = inward force (toward axis)
    # Inward force = collapsing → ℛ < 1 in original convention
    results['R_collapse'] = 1 + C  # positive C = outward = expanding
    results['C_raw'] = C
    
    # --- Metric C: Current-Field Misalignment ℛ_J ---
    # R = B × (∇×B) / |B|²
    # For axisymmetric cylindrical: J = (1/μ₀)∇×B
    # ∇×B for axisymmetric (∂/∂θ=0, ∂/∂z=0):
    #   (∇×B)_r = 0 (no θ or z derivatives)
    #   (∇×B)_θ = -∂B_r/∂z + ∂B_z/∂r ≈ ∂B_z/∂r (axisymmetric, B_r often 0)
    #   (∇×B)_z = (1/r)∂(rB_θ)/∂r = B_θ/r + ∂B_θ/∂r
    
    B_r = B[..., 0]
    B_theta = B[..., 1]
    B_z = B[..., 2]
    R = grid.R
    
    # Compute ∂B_θ/∂r and ∂B_z/∂r with 4th order
    dBtheta_dr = np.zeros_like(B_theta)
    dBz_dr = np.zeros_like(B_z)
    dr = grid.dr
    
    for arr, darr in [(B_theta, dBtheta_dr), (B_z, dBz_dr)]:
        Nr = grid.Nr
        if Nr >= 5:
            darr[2:-2] = (-arr[4:] + 8*arr[3:-1] - 8*arr[1:-3] + arr[:-4]) / (12 * dr)
            darr[0] = (arr[1] - arr[0]) / dr
            darr[1] = (arr[2] - arr[0]) / (2 * dr)
            darr[-1] = (arr[-1] - arr[-2]) / dr
            darr[-2] = (arr[-1] - arr[-3]) / (2 * dr)
        else:
            darr[1:-1] = (arr[2:] - arr[:-2]) / (2 * dr)
            darr[0] = (arr[1] - arr[0]) / dr
            darr[-1] = (arr[-1] - arr[-2]) / dr
    
    curlB = np.zeros_like(B)
    curlB[..., 0] = 0  # (∇×B)_r = 0 for axisymmetric, no z-dependence
    curlB[..., 1] = dBz_dr  # (∇×B)_θ = ∂B_z/∂r (simplified)
    curlB[..., 2] = B_theta / R + dBtheta_dr  # (∇×B)_z = B_θ/r + ∂B_θ/∂r
    
    # R_vec = B × (∇×B) / |B|²
    R_vec = np.cross(B, curlB) / (Bmag**2)[..., np.newaxis]
    R_mag = np.sqrt(np.sum(R_vec**2, axis=-1))
    
    results['R_misalign_vec'] = R_vec
    results['R_misalign_mag'] = R_mag
    # Normalize: for now store raw magnitude; normalization is test-dependent
    results['R_misalign'] = R_mag
    results['curlB'] = curlB
    
    # --- Universal Metric: |f_inward| / |f_outward| ---
    # In cylindrical, "inward" = negative radial direction
    # Tension for azimuthal fields points inward (toward axis)
    # Pressure gradient direction depends on field profile
    
    # Tension radial component (inward for azimuthal fields → negative r̂)
    f_tension_r = f_tension[..., 0]
    f_pressure_r = f_pressure[..., 0]
    
    # Include material pressure as outward force
    if p_mat is not None:
        dp_dr_out = np.zeros_like(p_mat)
        dr = grid.dr
        Nr = grid.Nr
        if Nr >= 5:
            dp_dr_out[2:-2] = (-p_mat[4:] + 8*p_mat[3:-1] - 8*p_mat[1:-3] + p_mat[:-4]) / (12 * dr)
            dp_dr_out[0] = (p_mat[1] - p_mat[0]) / dr
            dp_dr_out[1] = (p_mat[2] - p_mat[0]) / (2 * dr)
            dp_dr_out[-1] = (p_mat[-1] - p_mat[-2]) / dr
            dp_dr_out[-2] = (p_mat[-1] - p_mat[-3]) / (2 * dr)
        else:
            dp_dr_out[1:-1] = (p_mat[2:] - p_mat[:-2]) / (2 * dr)
            dp_dr_out[0] = (p_mat[1] - p_mat[0]) / dr
            dp_dr_out[-1] = (p_mat[-1] - p_mat[-2]) / dr
        f_outward_total = f_pressure_r - dp_dr_out  # negative dp/dr = outward for peaked pressure
    else:
        f_outward_total = f_pressure_r
    
    # Inward = magnitude of tension component pointing inward (negative r for azimuthal fields)
    f_inward_mag = np.abs(np.minimum(f_tension_r, 0))  # inward tension
    # Also add inward pressure gradient if it points inward
    f_inward_mag += np.abs(np.minimum(f_pressure_r, 0))
    
    # Outward = material pressure gradient + any outward magnetic pressure
    f_outward_mag = np.abs(np.maximum(f_pressure_r, 0))
    if p_mat is not None:
        f_outward_mag += np.abs(np.minimum(-dp_dr_out, 0))  # -dp/dr > 0 when p decreases outward
    
    # Avoid division by zero
    eps = 1e-30
    R_universal = f_inward_mag / np.maximum(f_outward_mag, eps)
    
    # Flag degenerate regions (both forces near zero)
    f_ref = np.max(Bmag**2 / (2 * mu_0)) * 1e-10
    degenerate = (f_inward_mag < f_ref) & (f_outward_mag < f_ref)
    R_universal = np.where(degenerate, 1.0, R_universal)
    
    results['R_universal'] = R_universal
    results['f_inward_mag'] = f_inward_mag
    results['f_outward_mag'] = f_outward_mag
    
    return results


# =============================================================================
# 6. VISUALIZATION
# =============================================================================

def plot_radial_profiles(r, profiles, title, filename, ylims=None):
    """Plot multiple quantities vs radius."""
    fig, axes = plt.subplots(len(profiles), 1, figsize=(12, 4 * len(profiles)),
                              sharex=True)
    if len(profiles) == 1:
        axes = [axes]
    
    for ax, (label, data, kwargs) in zip(axes, profiles):
        if isinstance(data, list):
            for d, l, kw in data:
                ax.plot(r, d, label=l, **kw)
            ax.legend(fontsize=10)
        else:
            ax.plot(r, data, **kwargs)
        ax.set_ylabel(label, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)
        if ylims and label in ylims:
            ax.set_ylim(ylims[label])
    
    axes[-1].set_xlabel('Radius r (m)', fontsize=12)
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


# =============================================================================
# 7. TEST CASES
# =============================================================================

def test_infinite_wire():
    """Test 1.1: Infinite straight wire — pure azimuthal field."""
    print("\n" + "="*70)
    print("TEST 1.1: INFINITE STRAIGHT WIRE")
    print("="*70)
    
    I = 1000.0  # 1 kA
    grid = CylindricalGrid(Nr=200, Ntheta=1, Nz=1,
                            r_range=(0.001, 0.1))
    
    B = field_infinite_wire(grid, I=I)
    results = compute_all_metrics(B, grid, p_mat=None)
    
    # Extract radial profiles (θ=0, z=0 slice)
    r = grid.r
    kappa = results['kappa_mag'][:, 0, 0]
    Bmag = results['Bmag'][:, 0, 0]
    f_tension_r = results['f_tension'][:, 0, 0, 0]
    f_pressure_r = results['f_pressure'][:, 0, 0, 0]
    R_collapse = results['R_collapse'][:, 0, 0]
    R_misalign = results['R_misalign'][:, 0, 0]
    R_universal = results['R_universal'][:, 0, 0]
    
    # Analytic expectations
    kappa_analytic = 1.0 / r  # circles of radius r
    B_analytic = mu_0 * I / (2 * np.pi * r)
    
    print(f"\n  Current: I = {I:.0f} A")
    print(f"  Grid: r = [{r[0]:.4f}, {r[-1]:.4f}] m, Nr = {grid.Nr}")
    print(f"\n  --- Curvature Check ---")
    print(f"  κ(r=0.01) computed:  {kappa[np.argmin(np.abs(r-0.01))]:.2f}")
    print(f"  κ(r=0.01) analytic:  {1.0/0.01:.2f}")
    print(f"  κ(r=0.05) computed:  {kappa[np.argmin(np.abs(r-0.05))]:.2f}")
    print(f"  κ(r=0.05) analytic:  {1.0/0.05:.2f}")
    
    kappa_error = np.max(np.abs(kappa - kappa_analytic) / kappa_analytic)
    print(f"  Max relative κ error: {kappa_error:.6f}")
    
    print(f"\n  --- Force Balance ---")
    print(f"  f_tension_r (inward, should be negative):")
    print(f"    at r=0.01: {f_tension_r[np.argmin(np.abs(r-0.01))]:.4e} N/m³")
    print(f"    at r=0.05: {f_tension_r[np.argmin(np.abs(r-0.05))]:.4e} N/m³")
    print(f"  f_pressure_r (pressure gradient):")
    print(f"    at r=0.01: {f_pressure_r[np.argmin(np.abs(r-0.01))]:.4e} N/m³")
    print(f"    at r=0.05: {f_pressure_r[np.argmin(np.abs(r-0.05))]:.4e} N/m³")
    
    # For a wire: f_tension = -B²/(μ₀r) (inward), f_pressure = -B²/(μ₀r) (also inward!)
    # Both forces point inward → R_universal should be very large (all inward, no outward)
    print(f"\n  --- ℛ Metrics ---")
    print(f"  ℛ_collapse at r=0.01: {R_collapse[np.argmin(np.abs(r-0.01))]:.4f}")
    print(f"  ℛ_collapse at r=0.05: {R_collapse[np.argmin(np.abs(r-0.05))]:.4f}")
    print(f"  ℛ_universal at r=0.01: {R_universal[np.argmin(np.abs(r-0.01))]:.4e}")
    print(f"  ℛ_universal at r=0.05: {R_universal[np.argmin(np.abs(r-0.05))]:.4e}")
    print(f"\n  EXPECTED: No outward force in vacuum wire → ℛ_universal → ∞")
    print(f"  RESULT:   ℛ_universal >> 1 everywhere ✓" if np.all(R_universal > 10) else 
          f"  RESULT:   ℛ_universal NOT >> 1 ✗")
    
    # Verify B field accuracy
    B_error = np.max(np.abs(Bmag - B_analytic) / B_analytic)
    print(f"\n  B field max relative error: {B_error:.2e}")
    
    # Plot
    profiles = [
        ('B_θ (T)', [
            (Bmag, 'Computed', {'color': 'blue', 'linewidth': 2}),
            (B_analytic, 'Analytic', {'color': 'red', 'linestyle': '--', 'linewidth': 2}),
        ], {}),
        ('κ (1/m)', [
            (kappa, 'Computed', {'color': 'blue', 'linewidth': 2}),
            (kappa_analytic, 'Analytic (1/r)', {'color': 'red', 'linestyle': '--', 'linewidth': 2}),
        ], {}),
        ('Force density (N/m³)', [
            (f_tension_r, 'f_tension (radial)', {'color': 'red', 'linewidth': 2}),
            (f_pressure_r, 'f_pressure (radial)', {'color': 'blue', 'linewidth': 2}),
            (f_tension_r + f_pressure_r, 'f_total', {'color': 'green', 'linewidth': 2, 'linestyle': '--'}),
        ], {}),
        ('ℛ_collapse', R_collapse, {'color': 'purple', 'linewidth': 2}),
    ]
    plot_radial_profiles(r, profiles, 
                          'Test 1.1: Infinite Wire — Field, Curvature, Forces, ℛ',
                          '/home/claude/test1_wire.png')
    
    return results


def test_zpinch_bennett():
    """Test 1.2: Z-pinch with Bennett equilibrium."""
    print("\n" + "="*70)
    print("TEST 1.2: Z-PINCH (BENNETT EQUILIBRIUM)")
    print("="*70)
    
    I = 1e4      # 10 kA
    a = 0.01     # 1 cm pinch radius
    
    grid = CylindricalGrid(Nr=300, Ntheta=1, Nz=1,
                            r_range=(0.0005, 0.05))
    
    B, p_mat = field_zpinch_bennett(grid, I=I, a=a)
    
    # Equilibrium curvature: κ_eq = 1/a at the boundary
    def kappa_eq_zpinch(g):
        return np.full(g.R.shape, 1.0 / a)
    
    results = compute_all_metrics(B, grid, p_mat=p_mat, 
                                   kappa_eq_func=kappa_eq_zpinch, L0=a)
    
    r = grid.r
    kappa = results['kappa_mag'][:, 0, 0]
    Bmag = results['Bmag'][:, 0, 0]
    f_tension_r = results['f_tension'][:, 0, 0, 0]
    f_pressure_r = results['f_pressure'][:, 0, 0, 0]
    f_total_r = results['f_total'][:, 0, 0, 0]
    R_kappa = results['R_kappa'][:, 0, 0] if results['R_kappa'] is not None else None
    R_collapse = results['R_collapse'][:, 0, 0]
    R_universal = results['R_universal'][:, 0, 0]
    p = p_mat[:, 0, 0]
    
    # Find where ℛ_collapse ≈ 1
    idx_eq = np.argmin(np.abs(R_collapse - 1.0))
    r_eq_collapse = r[idx_eq]
    
    print(f"\n  Current: I = {I:.0f} A")
    print(f"  Bennett radius: a = {a*100:.1f} cm")
    print(f"  Grid: r = [{r[0]*1000:.1f}, {r[-1]*1000:.1f}] mm, Nr = {grid.Nr}")
    
    print(f"\n  --- Magnetic Field ---")
    print(f"  B_θ(r=a) = {Bmag[np.argmin(np.abs(r-a))]:.4f} T")
    print(f"  B_θ analytic at r=a: {mu_0*I/(2*np.pi*a):.4f} T")
    
    print(f"\n  --- Curvature ---")
    idx_a = np.argmin(np.abs(r-a))
    print(f"  κ(r=a) computed:  {kappa[idx_a]:.2f} 1/m")
    print(f"  κ analytic (1/r at r=a): {1.0/a:.2f} 1/m")
    
    if R_kappa is not None:
        print(f"\n  --- Metric A: ℛ_κ (Curvature Ratio) ---")
        print(f"  ℛ_κ(r=a):     {R_kappa[idx_a]:.4f} (should be ~1.0)")
        print(f"  ℛ_κ(r=a/2):   {R_kappa[np.argmin(np.abs(r-a/2))]:.4f} (should be >1)")
        print(f"  ℛ_κ(r=2a):    {R_kappa[np.argmin(np.abs(r-2*a))]:.4f} (should be <1)")
    
    print(f"\n  --- Metric B: ℛ_collapse ---")
    print(f"  ℛ_C(r=a):     {R_collapse[idx_a]:.4f}")
    print(f"  ℛ_C = 1.0 at r = {r_eq_collapse*100:.3f} cm (Bennett a = {a*100:.1f} cm)")
    print(f"  Equilibrium radius agreement: {abs(r_eq_collapse - a)/a * 100:.1f}%")
    
    print(f"\n  --- Metric: ℛ_universal ---")
    idx_eq_u = np.argmin(np.abs(R_universal[10:-10] - 1.0)) + 10
    print(f"  ℛ_univ(r=a):   {R_universal[idx_a]:.4f}")
    print(f"  ℛ_univ = 1.0 at r = {r[idx_eq_u]*100:.3f} cm")
    
    print(f"\n  --- Force Balance at r=a ---")
    print(f"  f_tension_r:   {f_tension_r[idx_a]:.4e} N/m³")
    print(f"  f_pressure_r:  {f_pressure_r[idx_a]:.4e} N/m³")
    print(f"  Material p(r=a): {p[idx_a]:.4e} Pa")
    print(f"  Material p(r=0): {p[0]:.4e} Pa")
    
    # Plot
    fig = plt.figure(figsize=(16, 20))
    gs = GridSpec(5, 2, figure=fig, hspace=0.35)
    
    # B field
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(r*100, Bmag, 'b-', linewidth=2, label='B_θ computed')
    B_in = mu_0 * I * r / (2 * np.pi * a**2)
    B_out = mu_0 * I / (2 * np.pi * r)
    B_analytic = np.where(r < a, B_in, B_out)
    ax1.plot(r*100, B_analytic, 'r--', linewidth=2, label='B_θ analytic')
    ax1.axvline(x=a*100, color='gray', linestyle=':', alpha=0.7, label=f'a = {a*100} cm')
    ax1.set_ylabel('B_θ (T)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Magnetic Field')
    
    # Pressure
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(r*100, p, 'orange', linewidth=2)
    ax2.axvline(x=a*100, color='gray', linestyle=':', alpha=0.7)
    ax2.set_ylabel('p_mat (Pa)')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('Material Pressure')
    
    # Curvature
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(r*100, kappa, 'b-', linewidth=2, label='κ computed')
    ax3.plot(r*100, 1.0/r, 'r--', linewidth=2, label='1/r (analytic for circles)')
    ax3.axvline(x=a*100, color='gray', linestyle=':', alpha=0.7)
    ax3.set_ylabel('κ (1/m)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_title('Field-Line Curvature')
    ax3.set_ylim([0, min(500, kappa.max()*1.1)])
    
    # Forces
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(r*100, f_tension_r, 'r-', linewidth=2, label='f_tension (radial)')
    ax4.plot(r*100, f_pressure_r, 'b-', linewidth=2, label='f_pressure (radial)')
    ax4.plot(r*100, f_total_r, 'g--', linewidth=2, label='f_total (magnetic)')
    ax4.axvline(x=a*100, color='gray', linestyle=':', alpha=0.7)
    ax4.axhline(y=0, color='black', linewidth=0.5)
    ax4.set_ylabel('Force density (N/m³)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Force Decomposition')
    
    # ℛ_κ
    if R_kappa is not None:
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(r*100, R_kappa, 'purple', linewidth=2)
        ax5.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, label='ℛ = 1 (equilibrium)')
        ax5.axvline(x=a*100, color='gray', linestyle=':', alpha=0.7, label=f'Bennett a')
        ax5.set_ylabel('ℛ_κ')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        ax5.set_title('Metric A: Curvature Ratio')
        ax5.set_ylim([0, 5])
    
    # ℛ_collapse
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(r*100, R_collapse, 'darkred', linewidth=2)
    ax6.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, label='ℛ = 1 (equilibrium)')
    ax6.axvline(x=a*100, color='gray', linestyle=':', alpha=0.7, label=f'Bennett a')
    ax6.set_ylabel('ℛ_C')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    ax6.set_title('Metric B: Collapse Indicator')
    ax6.set_ylim([-2, 4])
    
    # ℛ_universal
    ax7 = fig.add_subplot(gs[3, :])
    ax7.semilogy(r*100, R_universal, 'navy', linewidth=2)
    ax7.axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, label='ℛ = 1 (equilibrium)')
    ax7.axvline(x=a*100, color='gray', linestyle=':', alpha=0.7, label=f'Bennett a = {a*100} cm')
    ax7.set_ylabel('ℛ_universal (log scale)')
    ax7.set_xlabel('Radius r (cm)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    ax7.set_title('Universal Force Ratio')
    ax7.set_ylim([0.01, 1000])
    
    # Misalignment
    R_misalign = results['R_misalign'][:, 0, 0]
    ax8 = fig.add_subplot(gs[4, :])
    ax8.plot(r*100, R_misalign, 'teal', linewidth=2)
    ax8.axvline(x=a*100, color='gray', linestyle=':', alpha=0.7, label=f'Bennett a')
    ax8.set_ylabel('|R| (misalignment)')
    ax8.set_xlabel('Radius r (cm)')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    ax8.set_title('Metric C: Current-Field Misalignment')
    
    fig.suptitle('Test 1.2: Z-Pinch Bennett Equilibrium — All ℛ Metrics', 
                  fontsize=16, fontweight='bold', y=1.01)
    plt.savefig('/home/claude/test2_zpinch.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: /home/claude/test2_zpinch.png")
    
    return results


def test_zpinch_vs_theta_pinch():
    """Test 1.3: Z-pinch vs θ-pinch comparison."""
    print("\n" + "="*70)
    print("TEST 1.3: Z-PINCH vs θ-PINCH COMPARISON")
    print("="*70)
    
    I = 1e4
    a = 0.01
    B0 = 1.0
    
    grid = CylindricalGrid(Nr=300, Ntheta=1, Nz=1,
                            r_range=(0.0005, 0.05))
    
    # Z-pinch
    B_z, p_z = field_zpinch_bennett(grid, I=I, a=a)
    res_z = compute_all_metrics(B_z, grid, p_mat=p_z, L0=a)
    
    # θ-pinch
    B_t, p_t = field_theta_pinch(grid, B0=B0, a=a, beta=0.5)
    res_t = compute_all_metrics(B_t, grid, p_mat=p_t, L0=a)
    
    r = grid.r
    
    kappa_z = res_z['kappa_mag'][:, 0, 0]
    kappa_t = res_t['kappa_mag'][:, 0, 0]
    R_c_z = res_z['R_collapse'][:, 0, 0]
    R_c_t = res_t['R_collapse'][:, 0, 0]
    R_u_z = res_z['R_universal'][:, 0, 0]
    R_u_t = res_t['R_universal'][:, 0, 0]
    R_m_z = res_z['R_misalign'][:, 0, 0]
    R_m_t = res_t['R_misalign'][:, 0, 0]
    
    print(f"\n  --- Curvature Comparison ---")
    idx_mid = np.argmin(np.abs(r - a))
    print(f"  κ at r=a:")
    print(f"    Z-pinch:  {kappa_z[idx_mid]:.2f} 1/m")
    print(f"    θ-pinch:  {kappa_t[idx_mid]:.6f} 1/m")
    print(f"    Ratio:    {kappa_z[idx_mid] / max(kappa_t[idx_mid], 1e-30):.1f}x")
    
    print(f"\n  --- ℛ_collapse Comparison at r=a ---")
    print(f"    Z-pinch:  {R_c_z[idx_mid]:.4f}")
    print(f"    θ-pinch:  {R_c_t[idx_mid]:.4f}")
    
    print(f"\n  --- Key Result ---")
    z_has_gradient = np.std(kappa_z[10:-10]) > 0.1
    t_is_flat = np.std(kappa_t[10:-10]) < 0.1
    print(f"  Z-pinch has strong κ gradient: {'✓' if z_has_gradient else '✗'}")
    print(f"  θ-pinch has flat/zero κ:       {'✓' if t_is_flat else '✗'}")
    print(f"  Framework distinguishes them:   {'✓ YES' if z_has_gradient and t_is_flat else '✗ NO'}")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    axes[0, 0].plot(r*100, kappa_z, 'r-', linewidth=2, label='Z-pinch')
    axes[0, 0].plot(r*100, kappa_t, 'b-', linewidth=2, label='θ-pinch')
    axes[0, 0].axvline(x=a*100, color='gray', linestyle=':', alpha=0.7)
    axes[0, 0].set_ylabel('κ (1/m)')
    axes[0, 0].set_title('Field-Line Curvature')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0, min(500, kappa_z.max()*1.1)])
    
    axes[0, 1].plot(r*100, R_c_z, 'r-', linewidth=2, label='Z-pinch')
    axes[0, 1].plot(r*100, R_c_t, 'b-', linewidth=2, label='θ-pinch')
    axes[0, 1].axhline(y=1.0, color='green', linestyle='--', linewidth=1.5)
    axes[0, 1].axvline(x=a*100, color='gray', linestyle=':', alpha=0.7)
    axes[0, 1].set_ylabel('ℛ_C')
    axes[0, 1].set_title('Collapse Indicator')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim([-3, 5])
    
    axes[1, 0].semilogy(r*100, np.maximum(R_u_z, 1e-3), 'r-', linewidth=2, label='Z-pinch')
    axes[1, 0].semilogy(r*100, np.maximum(R_u_t, 1e-3), 'b-', linewidth=2, label='θ-pinch')
    axes[1, 0].axhline(y=1.0, color='green', linestyle='--', linewidth=1.5)
    axes[1, 0].axvline(x=a*100, color='gray', linestyle=':', alpha=0.7)
    axes[1, 0].set_ylabel('ℛ_universal (log)')
    axes[1, 0].set_title('Universal Force Ratio')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(r*100, R_m_z, 'r-', linewidth=2, label='Z-pinch')
    axes[1, 1].plot(r*100, R_m_t, 'b-', linewidth=2, label='θ-pinch')
    axes[1, 1].axvline(x=a*100, color='gray', linestyle=':', alpha=0.7)
    axes[1, 1].set_ylabel('|R| (misalignment)')
    axes[1, 1].set_title('Current-Field Misalignment')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    for ax in axes.flat:
        ax.set_xlabel('Radius r (cm)')
    
    fig.suptitle('Test 1.3: Z-Pinch vs θ-Pinch — All Metrics Compared', 
                  fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/claude/test3_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: /home/claude/test3_comparison.png")
    
    return res_z, res_t


def test_toy_dynamics():
    """
    Toy 1D dynamic test: Z-pinch radius evolution.
    Model: thin shell of radius a(t) carrying current I.
    d²a/dt² = (1/ρ_L) * [p_mat/a - μ₀I²/(4π²a³)]
    where ρ_L is mass per unit length.
    
    Track ℛ(a(t)) during compression/expansion.
    """
    print("\n" + "="*70)
    print("TEST: TOY DYNAMICS — Z-PINCH RADIUS EVOLUTION")
    print("="*70)
    
    from scipy.integrate import solve_ivp
    
    I = 1e4          # Current (A)
    a0 = 0.02        # Initial radius (m) — twice the Bennett radius
    a_bennett = 0.01  # Bennett equilibrium radius
    v0 = 0.0         # Initial radial velocity
    
    # Derived: Bennett pressure at a_bennett
    p0_bennett = mu_0 * I**2 / (8 * np.pi**2 * a_bennett**2)
    
    # Mass per unit length (set to give reasonable dynamics)
    rho_L = 1e-4  # kg/m
    
    def pinch_ode(t, y):
        a, v = y
        a = max(a, 1e-6)  # prevent singularity
        
        # Outward: material pressure (scales as p0 * (a_bennett/a)^2 for adiabatic)
        # Simplified: thermal pressure that matches Bennett at a_bennett
        f_out = p0_bennett * (a_bennett / a)**2
        
        # Inward: magnetic tension ~ μ₀I²/(4π²a)
        f_in = mu_0 * I**2 / (4 * np.pi**2 * a)
        
        # Net force per unit length / mass per unit length
        accel = (f_out - f_in) / (rho_L * a)
        
        return [v, accel]
    
    # Solve
    t_span = (0, 1e-4)  # 0.1 ms
    t_eval = np.linspace(*t_span, 2000)
    sol = solve_ivp(pinch_ode, t_span, [a0, v0], t_eval=t_eval, 
                     method='RK45', rtol=1e-10, atol=1e-14)
    
    a_t = sol.y[0]
    v_t = sol.y[1]
    t = sol.t
    
    # Compute ℛ_universal along trajectory
    R_traj = np.zeros_like(a_t)
    for i, a in enumerate(a_t):
        a = max(a, 1e-6)
        f_in = mu_0 * I**2 / (4 * np.pi**2 * a)  # magnetic tension (inward)
        f_out = p0_bennett * (a_bennett / a)**2     # pressure (outward)
        R_traj[i] = f_in / max(f_out, 1e-30)
    
    # Find where ℛ crosses 1
    crossings = np.where(np.diff(np.sign(R_traj - 1.0)))[0]
    
    print(f"\n  Initial radius: a₀ = {a0*100:.1f} cm")
    print(f"  Bennett equilibrium: a_eq = {a_bennett*100:.1f} cm")
    print(f"  ℛ(t=0): {R_traj[0]:.4f}")
    print(f"  ℛ at first crossing: t = {t[crossings[0]]*1e6:.2f} μs" if len(crossings) > 0 else "  No crossing found")
    print(f"  Minimum radius reached: {np.min(a_t)*100:.4f} cm")
    print(f"  Final radius: {a_t[-1]*100:.4f} cm")
    print(f"  Oscillations visible: {'✓' if len(crossings) >= 3 else '✗'}")
    
    # Plot
    fig, axes = plt.subplots(3, 1, figsize=(14, 14), sharex=True)
    
    axes[0].plot(t*1e6, a_t*100, 'b-', linewidth=2)
    axes[0].axhline(y=a_bennett*100, color='green', linestyle='--', linewidth=1.5,
                     label=f'Bennett a = {a_bennett*100} cm')
    axes[0].set_ylabel('Radius a(t) (cm)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title('Pinch Radius Evolution')
    
    axes[1].plot(t*1e6, v_t, 'r-', linewidth=2)
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].set_ylabel('Radial velocity (m/s)')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title('Radial Velocity')
    
    axes[2].plot(t*1e6, R_traj, 'purple', linewidth=2)
    axes[2].axhline(y=1.0, color='green', linestyle='--', linewidth=1.5, label='ℛ = 1')
    axes[2].fill_between(t*1e6, R_traj, 1.0, where=R_traj > 1.0, 
                          alpha=0.15, color='red', label='Contracting (ℛ > 1)')
    axes[2].fill_between(t*1e6, R_traj, 1.0, where=R_traj < 1.0,
                          alpha=0.15, color='blue', label='Expanding (ℛ < 1)')
    axes[2].set_ylabel('ℛ_universal')
    axes[2].set_xlabel('Time (μs)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title('Rotational Parameter During Dynamics')
    axes[2].set_ylim([0, max(R_traj.max()*1.1, 2)])
    
    fig.suptitle('Toy Dynamics: Z-Pinch Oscillation with ℛ Tracking',
                  fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/claude/test4_dynamics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: /home/claude/test4_dynamics.png")
    
    return sol, R_traj


def test_screw_pinch_sweep():
    """
    Bonus: Screw pinch — sweep pitch angle and show ℛ profiles change.
    Combined B_θ and B_z fields with varying ratio.
    """
    print("\n" + "="*70)
    print("TEST: SCREW PINCH — PITCH ANGLE SWEEP")
    print("="*70)
    
    I = 1e4
    a = 0.01
    grid = CylindricalGrid(Nr=300, Ntheta=1, Nz=1,
                            r_range=(0.0005, 0.05))
    
    # B_z magnitudes as fraction of B_θ(a) = μ₀I/(2πa)
    B_theta_a = mu_0 * I / (2 * np.pi * a)
    Bz_ratios = [0.0, 0.5, 1.0, 2.0, 5.0]
    
    r = grid.r
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for ratio in Bz_ratios:
        B_zpinch, p_mat = field_zpinch_bennett(grid, I=I, a=a)
        
        # Add axial field
        Bz_val = ratio * B_theta_a
        B_screw = B_zpinch.copy()
        B_screw[..., 2] = Bz_val  # uniform B_z
        
        res = compute_all_metrics(B_screw, grid, p_mat=p_mat, L0=a)
        
        kappa = res['kappa_mag'][:, 0, 0]
        R_c = res['R_collapse'][:, 0, 0]
        R_u = res['R_universal'][:, 0, 0]
        R_m = res['R_misalign'][:, 0, 0]
        
        pitch_angle = np.degrees(np.arctan2(Bz_val, B_theta_a))
        label = f'B_z/B_θ = {ratio} (pitch {pitch_angle:.0f}°)'
        
        axes[0, 0].plot(r*100, kappa, linewidth=1.5, label=label)
        axes[0, 1].plot(r*100, R_c, linewidth=1.5, label=label)
        axes[1, 0].semilogy(r*100, np.maximum(R_u, 1e-3), linewidth=1.5, label=label)
        axes[1, 1].plot(r*100, R_m, linewidth=1.5, label=label)
    
    for ax in axes.flat:
        ax.axvline(x=a*100, color='gray', linestyle=':', alpha=0.5)
        ax.set_xlabel('Radius r (cm)')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    
    axes[0, 0].set_ylabel('κ (1/m)')
    axes[0, 0].set_title('Curvature')
    axes[0, 0].set_ylim([0, 400])
    
    axes[0, 1].set_ylabel('ℛ_C')
    axes[0, 1].set_title('Collapse Indicator')
    axes[0, 1].axhline(y=1.0, color='green', linestyle='--', linewidth=1)
    axes[0, 1].set_ylim([-3, 5])
    
    axes[1, 0].set_ylabel('ℛ_universal (log)')
    axes[1, 0].set_title('Universal Force Ratio')
    axes[1, 0].axhline(y=1.0, color='green', linestyle='--', linewidth=1)
    
    axes[1, 1].set_ylabel('|R| (misalignment)')
    axes[1, 1].set_title('Current-Field Misalignment')
    
    fig.suptitle('Screw Pinch: Effect of Axial Field on ℛ Metrics',
                  fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/claude/test5_screw_pinch.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: /home/claude/test5_screw_pinch.png")


def test_dipole_field():
    """
    Bonus: Magnetic dipole field — compute ℛ map on a 2D slice.
    """
    print("\n" + "="*70)
    print("TEST: MAGNETIC DIPOLE — 2D ℛ MAP")
    print("="*70)
    
    m = 1.0  # magnetic moment (A·m²)
    
    grid = CartesianGrid(Nx=200, Ny=1, Nz=200,
                          x_range=(-0.1, 0.1),
                          y_range=(0, 0),
                          z_range=(-0.1, 0.1))
    
    X = grid.X[:, 0, :]  # (Nx, Nz) slice in xz-plane
    Z = grid.Z[:, 0, :]
    
    # Dipole field: B_r = (μ₀/4π)(2m cosθ)/r³, B_θ = (μ₀/4π)(m sinθ)/r³
    # In Cartesian: r = sqrt(x² + z²), cosθ = z/r
    r = np.sqrt(X**2 + Z**2)
    r = np.maximum(r, 1e-4)
    cos_theta = Z / r
    sin_theta = X / r
    
    prefactor = mu_0 * m / (4 * np.pi * r**3)
    
    # B in Cartesian (y=0 plane): Bx = 3m xz/(r⁵) * μ₀/(4π), Bz = m(3z²-r²)/(r⁵) * μ₀/(4π)
    Bx = mu_0 / (4 * np.pi) * m * 3 * X * Z / r**5
    Bz = mu_0 / (4 * np.pi) * m * (3 * Z**2 - r**2) / r**5
    
    Bmag = np.sqrt(Bx**2 + Bz**2)
    
    # Compute curvature from field-line geometry (2D)
    # b = B/|B|
    bx = Bx / np.maximum(Bmag, 1e-30)
    bz = Bz / np.maximum(Bmag, 1e-30)
    
    dx = grid.dx
    dz = grid.dz
    
    # (b·∇)b in 2D Cartesian
    # κ_x = b_x ∂b_x/∂x + b_z ∂b_x/∂z
    # κ_z = b_x ∂b_z/∂x + b_z ∂b_z/∂z
    
    dbx_dx = np.zeros_like(bx)
    dbx_dz = np.zeros_like(bx)
    dbz_dx = np.zeros_like(bz)
    dbz_dz = np.zeros_like(bz)
    
    # 4th order central differences
    for arr, darr, delta, axis in [
        (bx, dbx_dx, dx, 0), (bx, dbx_dz, dz, 1),
        (bz, dbz_dx, dx, 0), (bz, dbz_dz, dz, 1)
    ]:
        N = arr.shape[axis]
        if N >= 5:
            slc = [slice(None)] * 2
            
            # Interior: 4th order
            inner = [slice(None)] * 2
            inner[axis] = slice(2, -2)
            
            p2 = [slice(None)] * 2; p2[axis] = slice(4, None)
            p1 = [slice(None)] * 2; p1[axis] = slice(3, -1)
            m1 = [slice(None)] * 2; m1[axis] = slice(1, -3)
            m2 = [slice(None)] * 2; m2[axis] = slice(None, -4)
            
            darr[tuple(inner)] = (-arr[tuple(p2)] + 8*arr[tuple(p1)] - 
                                    8*arr[tuple(m1)] + arr[tuple(m2)]) / (12 * delta)
    
    kx = bx * dbx_dx + bz * dbx_dz
    kz = bx * dbz_dx + bz * dbz_dz
    kappa = np.sqrt(kx**2 + kz**2)
    
    # Magnetic pressure gradient magnitude
    dB2_dx = np.zeros_like(Bmag)
    dB2_dz = np.zeros_like(Bmag)
    B2 = Bmag**2
    
    if grid.Nx >= 5:
        dB2_dx[2:-2, :] = (-B2[4:, :] + 8*B2[3:-1, :] - 8*B2[1:-3, :] + B2[:-4, :]) / (12*dx)
    if grid.Nz >= 5:
        dB2_dz[:, 2:-2] = (-B2[:, 4:] + 8*B2[:, 3:-1] - 8*B2[:, 1:-3] + B2[:, :-4]) / (12*dz)
    
    f_tension_mag = (Bmag**2 / mu_0) * kappa
    f_pressure_mag = np.sqrt(dB2_dx**2 + dB2_dz**2) / (2 * mu_0)
    
    # Simple ratio diagnostic (not full projection, but informative)
    R_map = f_tension_mag / np.maximum(f_pressure_mag, 1e-30)
    
    # Mask near origin (divergent)
    mask = r < 0.005
    R_map[mask] = np.nan
    kappa[mask] = np.nan
    
    print(f"  Dipole moment: m = {m} A·m²")
    print(f"  Grid: {grid.Nx}×{grid.Nz}, x,z ∈ [{grid.x_range[0]}, {grid.x_range[1]}]")
    print(f"  ℛ range (excluding core): [{np.nanmin(R_map):.3f}, {np.nanmax(R_map):.3f}]")
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # B magnitude
    im0 = axes[0].pcolormesh(X*100, Z*100, np.log10(np.maximum(Bmag, 1e-10)),
                               cmap='inferno', shading='auto')
    axes[0].set_title('log₁₀(|B|)')
    plt.colorbar(im0, ax=axes[0], label='log₁₀(T)')
    
    # Curvature
    im1 = axes[1].pcolormesh(X*100, Z*100, np.log10(np.maximum(kappa, 1e-1)),
                               cmap='viridis', shading='auto')
    axes[1].set_title('log₁₀(κ)')
    plt.colorbar(im1, ax=axes[1], label='log₁₀(1/m)')
    
    # ℛ map
    im2 = axes[2].pcolormesh(X*100, Z*100, R_map,
                               cmap='RdBu_r', shading='auto',
                               norm=mcolors.LogNorm(vmin=0.1, vmax=10))
    axes[2].set_title('ℛ (tension/pressure ratio)')
    plt.colorbar(im2, ax=axes[2], label='ℛ')
    
    for ax in axes:
        ax.set_xlabel('x (cm)')
        ax.set_ylabel('z (cm)')
        ax.set_aspect('equal')
        # Add field lines
        ax.streamplot(X.T*100, Z.T*100, Bx.T, Bz.T, color='white', 
                       linewidth=0.5, density=2, arrowsize=0.5)
    
    fig.suptitle('Magnetic Dipole: Field, Curvature, and ℛ Map',
                  fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/home/claude/test6_dipole.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: /home/claude/test6_dipole.png")


# =============================================================================
# 8. RUN ALL TESTS
# =============================================================================

if __name__ == '__main__':
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║        MagRot Framework — MVP Simulation Suite                 ║")
    print("║        Rotational-Vector Metrics for Magnetic Fields           ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    
    # Tier 1: Analytic
    res_wire = test_infinite_wire()
    res_zpinch = test_zpinch_bennett()
    res_z, res_t = test_zpinch_vs_theta_pinch()
    
    # Toy dynamics
    sol, R_traj = test_toy_dynamics()
    
    # Bonus tests
    test_screw_pinch_sweep()
    test_dipole_field()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)
    print("\nGenerated plots:")
    print("  1. test1_wire.png         — Infinite wire field/curvature/forces/ℛ")
    print("  2. test2_zpinch.png       — Z-pinch Bennett equilibrium (all metrics)")
    print("  3. test3_comparison.png   — Z-pinch vs θ-pinch comparison")
    print("  4. test4_dynamics.png     — Toy dynamics with ℛ tracking")
    print("  5. test5_screw_pinch.png  — Screw pinch pitch angle sweep")
    print("  6. test6_dipole.png       — Dipole 2D ℛ map")
