"""
Local entropy production rate density s_dot(x).

For resistive MHD:
    s_dot_resistive = eta |J|^2 / T

For general dissipative MHD:
    s_dot_total = eta|J|^2/T + (viscous terms)/T + (thermal conduction terms)/T

This module provides the resistive term (dominant in most tokamak and pinch
scenarios) with Spitzer resistivity:
    eta(T) = eta_0 (T / T_0)^(-3/2)

The spatial map of s_dot(x) is directly correlated with |R - 1| — regions
where force balance is violated produce entropy faster.
"""

import numpy as np
from scipy.constants import mu_0, e as e_charge, k as k_B


# ── Resistivity models ───────────────────────────────────────────────────────

# Spitzer resistivity reference: eta_0 ~ 5.2e-5 / T^(3/2) [Ohm m] for
# hydrogen plasma with Coulomb logarithm ln(Lambda) ~ 17.
# At T_0 = 1 keV ~ 1.16e7 K: eta_0 ~ 2.8e-8 Ohm m
SPITZER_ETA0 = 2.8e-8       # Ohm m at T_0
SPITZER_T0 = 1.16e7         # K  (1 keV)
SPITZER_LN_LAMBDA = 17.0    # Coulomb logarithm


def spitzer_resistivity(T, eta0=SPITZER_ETA0, T0=SPITZER_T0):
    """Spitzer resistivity eta(T) = eta0 * (T / T0)^(-3/2).

    Parameters
    ----------
    T : ndarray
        Temperature field (K).
    eta0 : float
        Reference resistivity at T0.
    T0 : float
        Reference temperature (K).

    Returns
    -------
    eta : ndarray
        Resistivity field (Ohm m).
    """
    T_safe = np.maximum(np.asarray(T, dtype=float), 1.0)
    return eta0 * (T_safe / T0) ** (-1.5)


def constant_resistivity(T, eta_value=1e-6):
    """Uniform resistivity (useful for testing / non-thermal plasmas)."""
    return np.full_like(np.asarray(T, dtype=float), eta_value)


# ── Temperature models ───────────────────────────────────────────────────────

def temperature_from_pressure(p, n, species_count=2):
    """T = p / (n_species * n * k_B).

    For a hydrogen plasma with n_e = n_i = n, species_count = 2 gives
    p = 2 n k_B T.

    Parameters
    ----------
    p : ndarray
        Plasma pressure (Pa).
    n : ndarray or float
        Number density (m^-3).
    species_count : int
        Number of species contributing to pressure (default 2 for e + i).

    Returns
    -------
    T : ndarray
        Temperature (K).
    """
    p_safe = np.maximum(np.asarray(p, dtype=float), 0.0)
    n_safe = np.maximum(np.asarray(n, dtype=float), 1.0)
    return p_safe / (species_count * n_safe * k_B)


def temperature_profile_parabolic(grid_or_shape, T_core=1e8, T_edge=1e6,
                                  psi_norm=None, r_norm=None):
    """Generate a parabolic temperature profile T = T_edge + (T_core - T_edge) * h(x).

    Accepts either normalized poloidal flux (toroidal) or normalized radius
    (cylindrical).

    Parameters
    ----------
    grid_or_shape : tuple or CylindricalGrid
        Target shape or grid.
    T_core : float
        Core temperature (K).
    T_edge : float
        Edge temperature (K).
    psi_norm : ndarray or None
        Normalized poloidal flux (1 at axis, 0 at edge).
    r_norm : ndarray or None
        Normalized radius (0 at axis, 1 at edge).

    Returns
    -------
    T : ndarray
        Temperature field (K).
    """
    if psi_norm is not None:
        profile = np.clip(psi_norm, 0, 1)
    elif r_norm is not None:
        profile = np.clip(1.0 - r_norm ** 2, 0, 1)
    else:
        raise ValueError("Must provide either psi_norm or r_norm")
    return T_edge + (T_core - T_edge) * profile


# ── Entropy production rate density ──────────────────────────────────────────

def entropy_production_resistive(J, T, eta=None, resistivity_model='spitzer',
                                 eta_value=1e-6):
    """Compute s_dot = eta |J|^2 / T  (resistive entropy production).

    Parameters
    ----------
    J : ndarray
        Current density.  If last axis is 3, treated as vector field and
        |J|^2 = sum(J^2, axis=-1).  Otherwise treated as scalar |J|.
    T : ndarray
        Temperature field (K).  Must broadcast with J.
    eta : ndarray or None
        Pre-computed resistivity.  If None, computed from T using the
        specified resistivity model.
    resistivity_model : str
        'spitzer' or 'constant'.
    eta_value : float
        Resistivity value for 'constant' model (Ohm m).

    Returns
    -------
    s_dot : ndarray
        Local entropy production rate density (W / (m^3 K) = W m^-3 K^-1).
    Jmag2 : ndarray
        |J|^2 field (A^2 / m^4).
    eta_field : ndarray
        Resistivity field used (Ohm m).
    """
    # |J|^2
    if isinstance(J, np.ndarray) and J.ndim >= 2 and J.shape[-1] == 3:
        Jmag2 = np.sum(J ** 2, axis=-1)
    else:
        Jmag2 = np.asarray(J, dtype=float) ** 2

    T_safe = np.maximum(np.asarray(T, dtype=float), 1.0)

    # Resistivity
    if eta is not None:
        eta_field = np.asarray(eta, dtype=float)
    elif resistivity_model == 'spitzer':
        eta_field = spitzer_resistivity(T_safe)
    else:
        eta_field = constant_resistivity(T_safe, eta_value=eta_value)

    s_dot = eta_field * Jmag2 / T_safe
    return s_dot, Jmag2, eta_field


# ── Integrated entropy production rate ────────────────────────────────────────

def total_entropy_production_cylindrical(s_dot, grid):
    """Integrate S_dot_total = integral s_dot dV on a cylindrical grid.

    Parameters
    ----------
    s_dot : ndarray
    grid : CylindricalGrid

    Returns
    -------
    S_dot_total : float
        Total entropy production rate (W/K).
    """
    from .free_energy import _volume_elements_cylindrical
    dV = _volume_elements_cylindrical(grid)
    return float(np.sum(s_dot * dV))


def total_entropy_production_toroidal(s_dot, RR, dR, dZ, mask=None):
    """Integrate S_dot_total = integral s_dot dV on a toroidal grid.

    Parameters
    ----------
    s_dot : ndarray (NR, NZ)
    RR : ndarray (NR, NZ)
    dR, dZ : float
    mask : ndarray of bool or None

    Returns
    -------
    S_dot_total : float
        Total entropy production rate (W/K).
    """
    from .free_energy import _volume_elements_toroidal
    dV = _volume_elements_toroidal(RR, dR, dZ)
    integrand = s_dot * dV
    if mask is not None:
        integrand = np.where(mask, integrand, 0.0)
    return float(np.sum(integrand))


def dissipated_power(s_dot, T, grid=None, RR=None, dR=None, dZ=None,
                     mask=None):
    """Compute dissipated power P = integral eta |J|^2 dV = integral T * s_dot dV.

    Accepts either a CylindricalGrid or toroidal grid parameters.

    Returns
    -------
    P : float
        Total dissipated power (W).
    """
    T_arr = np.asarray(T, dtype=float)
    power_density = T_arr * s_dot

    if grid is not None:
        from .free_energy import _volume_elements_cylindrical
        dV = _volume_elements_cylindrical(grid)
        return float(np.sum(power_density * dV))
    elif RR is not None and dR is not None and dZ is not None:
        from .free_energy import _volume_elements_toroidal
        dV = _volume_elements_toroidal(RR, dR, dZ)
        integrand = power_density * dV
        if mask is not None:
            integrand = np.where(mask, integrand, 0.0)
        return float(np.sum(integrand))
    else:
        raise ValueError("Provide either grid (cylindrical) or RR/dR/dZ (toroidal)")
