"""
Entropy production accounting at equilibrium (Test B).

Goal: Determine if R = 1 is a global thermodynamic equilibrium (s_dot = 0)
or a driven steady state (s_dot > 0).

Method:
    1. Compute s_dot(x) = eta(T) |J(x)|^2 / T(x) across equilibrium
    2. Use Spitzer resistivity for realistic T dependence
    3. Integrate total S_dot = integral s_dot dV
    4. Convert to power: P_dissipated = T_avg * S_dot
    5. Compare with known tokamak power balance (ITER: ~50 MW)

Expected result: S_dot > 0 everywhere inside plasma (J != 0, eta > 0),
establishing that R = 1 is NOT global thermodynamic equilibrium.
"""

import numpy as np
from scipy.constants import mu_0, k as k_B
from dataclasses import dataclass


@dataclass
class EntropyAuditResult:
    """Results of the entropy production audit at equilibrium."""
    S_dot_total: float          # Total entropy production rate (W/K)
    P_dissipated: float         # Total dissipated power (W)
    T_avg: float                # Volume-averaged temperature (K)
    s_dot_max: float            # Peak local entropy production rate
    s_dot_min_nonzero: float    # Minimum nonzero s_dot inside plasma
    J_rms: float                # RMS current density inside plasma (A/m^2)
    eta_avg: float              # Volume-averaged resistivity (Ohm m)
    is_driven_state: bool       # True if S_dot > 0 (driven, not equilibrium)
    fraction_producing: float   # Fraction of plasma volume with s_dot > threshold


def entropy_audit_cylindrical(B, grid, p=None, T=None, n_density=None,
                              resistivity_model='spitzer', eta_value=1e-6,
                              T_core=1e8, T_edge=1e6):
    """Run entropy production audit on a cylindrical equilibrium.

    Parameters
    ----------
    B : ndarray, shape (*grid.R.shape, 3)
    grid : CylindricalGrid
    p : ndarray or None
    T : ndarray or None
        Temperature field.  If None, generated from p and n_density,
        or as a parabolic profile.
    n_density : ndarray or float or None
        Number density for T = p / (2 n k_B).
    resistivity_model : str
    eta_value : float
    T_core, T_edge : float
        For parabolic temperature profile if T not given.

    Returns
    -------
    EntropyAuditResult
    s_dot : ndarray
        Local entropy production rate density field.
    """
    from ..stress.maxwell import compute_forces_conservative_cyl
    from ..thermodynamics.entropy import (
        entropy_production_resistive,
        total_entropy_production_cylindrical,
        dissipated_power,
        temperature_profile_parabolic,
    )
    from ..thermodynamics.free_energy import _volume_elements_cylindrical

    # Compute current density
    phys = compute_forces_conservative_cyl(B, grid, p)
    J = phys['J']

    # Temperature
    if T is None:
        if p is not None and n_density is not None:
            from ..thermodynamics.entropy import temperature_from_pressure
            T = temperature_from_pressure(p, n_density)
        else:
            # Parabolic profile based on normalized radius
            r = grid.R
            r_max = grid.r[-1]
            r_norm = r / r_max
            T = temperature_profile_parabolic(
                grid.R.shape, T_core=T_core, T_edge=T_edge, r_norm=r_norm
            )

    # Entropy production
    s_dot, Jmag2, eta_field = entropy_production_resistive(
        J, T, resistivity_model=resistivity_model, eta_value=eta_value
    )

    S_dot_total = total_entropy_production_cylindrical(s_dot, grid)
    P = dissipated_power(s_dot, T, grid=grid)

    # Statistics
    dV = _volume_elements_cylindrical(grid)
    V_total = float(np.sum(dV))
    T_avg = float(np.sum(T * dV) / V_total)
    eta_avg = float(np.sum(eta_field * dV) / V_total)
    J_rms = float(np.sqrt(np.mean(Jmag2)))

    s_flat = s_dot.ravel()
    threshold = np.max(s_flat) * 1e-10
    fraction = float(np.sum(s_flat > threshold)) / max(len(s_flat), 1)

    return EntropyAuditResult(
        S_dot_total=S_dot_total,
        P_dissipated=P,
        T_avg=T_avg,
        s_dot_max=float(np.max(s_dot)),
        s_dot_min_nonzero=float(np.min(s_flat[s_flat > threshold])) if np.any(s_flat > threshold) else 0.0,
        J_rms=J_rms,
        eta_avg=eta_avg,
        is_driven_state=S_dot_total > 0,
        fraction_producing=fraction,
    ), s_dot


def entropy_audit_toroidal(BR, BZ, Bphi, Jmag_or_components, RR, dR, dZ,
                           p=None, T=None, mask=None,
                           resistivity_model='spitzer', eta_value=1e-6,
                           T_core=1e8, T_edge=1e6, psi_norm=None):
    """Run entropy production audit on a toroidal equilibrium.

    Parameters
    ----------
    BR, BZ, Bphi : ndarray (NR, NZ)
    Jmag_or_components : ndarray or tuple
        Either scalar |J| field (NR, NZ), or tuple (JR, Jphi, JZ).
    RR : ndarray (NR, NZ)
    dR, dZ : float
    p : ndarray or None
    T : ndarray or None
    mask : ndarray of bool or None
        Plasma interior mask.
    resistivity_model : str
    eta_value : float
    T_core, T_edge : float
    psi_norm : ndarray or None
        Normalized poloidal flux for temperature profile.

    Returns
    -------
    EntropyAuditResult
    s_dot : ndarray
    """
    from ..thermodynamics.entropy import (
        entropy_production_resistive,
        total_entropy_production_toroidal,
        dissipated_power,
        temperature_profile_parabolic,
    )
    from ..thermodynamics.free_energy import _volume_elements_toroidal

    # Current density magnitude
    if isinstance(Jmag_or_components, tuple):
        JR, Jphi, JZ = Jmag_or_components
        Jmag = np.sqrt(JR ** 2 + Jphi ** 2 + JZ ** 2)
    else:
        Jmag = np.asarray(Jmag_or_components)

    # Temperature
    if T is None:
        if psi_norm is not None:
            T = temperature_profile_parabolic(
                RR.shape, T_core=T_core, T_edge=T_edge, psi_norm=psi_norm
            )
        else:
            # Rough parabolic fallback using distance from grid center
            R0 = (RR.min() + RR.max()) / 2
            Z0 = 0.0
            r_eff = np.sqrt((RR - R0) ** 2 + Z0 ** 2)
            r_norm = r_eff / np.max(r_eff)
            T = temperature_profile_parabolic(
                RR.shape, T_core=T_core, T_edge=T_edge, r_norm=r_norm
            )

    # Entropy production (using scalar Jmag)
    s_dot, Jmag2, eta_field = entropy_production_resistive(
        Jmag, T, resistivity_model=resistivity_model, eta_value=eta_value
    )

    S_dot_total = total_entropy_production_toroidal(s_dot, RR, dR, dZ, mask)
    P = dissipated_power(s_dot, T, RR=RR, dR=dR, dZ=dZ, mask=mask)

    # Statistics
    dV = _volume_elements_toroidal(RR, dR, dZ)
    if mask is not None:
        dV_masked = np.where(mask, dV, 0.0)
    else:
        dV_masked = dV
    V_total = float(np.sum(dV_masked))
    V_total = max(V_total, 1e-30)
    T_avg = float(np.sum(T * dV_masked) / V_total)
    eta_avg = float(np.sum(eta_field * dV_masked) / V_total)

    if mask is not None:
        Jmag2_inside = Jmag2[mask]
        s_flat = s_dot[mask].ravel()
    else:
        Jmag2_inside = Jmag2.ravel()
        s_flat = s_dot.ravel()

    J_rms = float(np.sqrt(np.mean(Jmag2_inside))) if len(Jmag2_inside) > 0 else 0.0
    threshold = np.max(s_flat) * 1e-10 if len(s_flat) > 0 else 0.0
    fraction = float(np.sum(s_flat > threshold)) / max(len(s_flat), 1)

    return EntropyAuditResult(
        S_dot_total=S_dot_total,
        P_dissipated=P,
        T_avg=T_avg,
        s_dot_max=float(np.max(s_dot)),
        s_dot_min_nonzero=float(np.min(s_flat[s_flat > threshold])) if np.any(s_flat > threshold) else 0.0,
        J_rms=J_rms,
        eta_avg=eta_avg,
        is_driven_state=S_dot_total > 0,
        fraction_producing=fraction,
    ), s_dot
