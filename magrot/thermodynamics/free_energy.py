"""
Free energy functional F[B, p] for magnetic field configurations.

F[B, p] = integral( B^2/(2 mu_0) + p/(gamma-1) ) dV  -  F_eq

where F_eq is the minimum-energy state satisfying constraints (helicity,
flux, mass).  F >= 0 always; F = 0 at equilibrium.

The R-F connection:
    |R(x) - 1|  is proportional to  |delta F / delta B(x)| / (B^2 / 2 mu_0)

This makes R the dimensionless local gradient of the free energy functional,
measured in units of the local magnetic energy density.

Supports both cylindrical (CylindricalGrid) and toroidal (TokamakEquilibrium-
style RR/ZZ grids) configurations.
"""

import numpy as np
from scipy.constants import mu_0


# ── Cylindrical grid helpers ──────────────────────────────────────────────────

def _volume_elements_cylindrical(grid):
    """Compute dV = r dr dtheta dz for a CylindricalGrid."""
    dr = grid.dr
    dtheta = (grid.theta[1] - grid.theta[0]) if len(grid.theta) > 1 else 2 * np.pi
    dz = (grid.z[1] - grid.z[0]) if len(grid.z) > 1 else 1.0
    return grid.R * dr * dtheta * dz


def _volume_elements_toroidal(RR, dR, dZ):
    """Compute dV = 2 pi R dR dZ for an axisymmetric toroidal grid."""
    return 2 * np.pi * RR * dR * dZ


# ── Energy density ────────────────────────────────────────────────────────────

def magnetic_energy_density(B):
    """Compute B^2 / (2 mu_0) at every grid point.

    Parameters
    ----------
    B : ndarray
        Magnetic field.  Last axis must be the component axis (size 3) for
        vector fields, or a scalar Bmag2 = B^2 array.

    Returns
    -------
    u_B : ndarray
        Magnetic energy density (J/m^3).
    """
    if B.ndim >= 2 and B.shape[-1] == 3:
        Bmag2 = np.sum(B ** 2, axis=-1)
    else:
        Bmag2 = np.asarray(B)
    return Bmag2 / (2 * mu_0)


def thermal_energy_density(p, gamma=5.0 / 3.0):
    """Compute p / (gamma - 1) at every grid point.

    Parameters
    ----------
    p : ndarray
        Plasma pressure (Pa).
    gamma : float
        Adiabatic index.

    Returns
    -------
    u_th : ndarray
        Thermal energy density (J/m^3).
    """
    return np.asarray(p) / (gamma - 1.0)


def total_energy_density(B, p=None, gamma=5.0 / 3.0):
    """Sum of magnetic and thermal energy densities."""
    u = magnetic_energy_density(B)
    if p is not None:
        u = u + thermal_energy_density(p, gamma=gamma)
    return u


# ── Integrated free energy ────────────────────────────────────────────────────

def free_energy_cylindrical(B, grid, p=None, gamma=5.0 / 3.0, F_eq=0.0):
    """Compute F[B, p] on a CylindricalGrid.

    Parameters
    ----------
    B : ndarray, shape (*grid.R.shape, 3)
        Magnetic field vector.
    grid : CylindricalGrid
    p : ndarray or None
        Plasma pressure.
    gamma : float
        Adiabatic index.
    F_eq : float
        Equilibrium free energy to subtract (default 0).

    Returns
    -------
    F : float
        Total free energy (J/m along z, or J if z extent is physical).
    u : ndarray
        Energy density field (J/m^3).
    """
    u = total_energy_density(B, p, gamma)
    dV = _volume_elements_cylindrical(grid)
    F = float(np.sum(u * dV)) - F_eq
    return F, u


def free_energy_toroidal(BR, BZ, Bphi, RR, dR, dZ, p=None, mask=None,
                         gamma=5.0 / 3.0, F_eq=0.0):
    """Compute F[B, p] on a toroidal (R, Z) grid.

    Parameters
    ----------
    BR, BZ, Bphi : ndarray (NR, NZ)
        Magnetic field components.
    RR : ndarray (NR, NZ)
        Major radius at each grid point.
    dR, dZ : float
        Grid spacings.
    p : ndarray or None
        Plasma pressure.
    mask : ndarray of bool or None
        If given, integrate only where mask is True (e.g. inside plasma).
    gamma : float
    F_eq : float

    Returns
    -------
    F : float
        Total free energy (J).
    u : ndarray
        Energy density field (J/m^3).
    """
    Bmag2 = BR ** 2 + BZ ** 2 + Bphi ** 2
    u = Bmag2 / (2 * mu_0)
    if p is not None:
        u = u + np.asarray(p) / (gamma - 1.0)

    dV = _volume_elements_toroidal(RR, dR, dZ)

    integrand = u * dV
    if mask is not None:
        integrand = np.where(mask, integrand, 0.0)

    F = float(np.sum(integrand)) - F_eq
    return F, u


# ── R-deviation Lyapunov functional ──────────────────────────────────────────

def lyapunov_R_cylindrical(R_field, grid):
    """Compute the Lyapunov functional L = integral |R - 1|^2 dV.

    This is the global measure of deviation from equilibrium and must
    decrease monotonically along the entropy flow.

    Parameters
    ----------
    R_field : ndarray
        Any R metric field (R_universal, R_kappa, etc.).
    grid : CylindricalGrid

    Returns
    -------
    L : float
        Integrated squared deviation from equilibrium.
    deviation : ndarray
        |R - 1|^2 field.
    """
    deviation = (R_field - 1.0) ** 2
    dV = _volume_elements_cylindrical(grid)
    L = float(np.sum(deviation * dV))
    return L, deviation


def lyapunov_R_toroidal(R_field, RR, dR, dZ, mask=None):
    """Compute the Lyapunov functional L = integral |R - 1|^2 dV on a
    toroidal grid.

    Parameters
    ----------
    R_field : ndarray (NR, NZ)
    RR : ndarray (NR, NZ)
    dR, dZ : float
    mask : ndarray of bool or None

    Returns
    -------
    L : float
    deviation : ndarray
    """
    deviation = (R_field - 1.0) ** 2
    dV = _volume_elements_toroidal(RR, dR, dZ)
    integrand = deviation * dV
    if mask is not None:
        integrand = np.where(mask, integrand, 0.0)
    L = float(np.sum(integrand))
    return L, deviation


# ── Local free-energy gradient (|R - 1| connection) ──────────────────────────

def local_free_energy_gradient(R_field, B):
    """Estimate local |delta F / delta B| from the R metric.

    The plan states:
        |R(x) - 1|  proportional to  |delta F / delta B(x)| / (B^2/(2 mu_0))

    So:
        |delta F / delta B|  approx  |R - 1| * B^2 / (2 mu_0)

    Parameters
    ----------
    R_field : ndarray
    B : ndarray (vector field with last axis = 3, or scalar Bmag2)

    Returns
    -------
    grad_F : ndarray
        Estimated magnitude of the functional derivative of F.
    """
    if B.ndim >= 2 and B.shape[-1] == 3:
        Bmag2 = np.sum(B ** 2, axis=-1)
    else:
        Bmag2 = np.asarray(B)
    return np.abs(R_field - 1.0) * Bmag2 / (2 * mu_0)
