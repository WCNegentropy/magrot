"""
Conservation constraints for thermodynamic state flow.

The equilibrium free energy F_eq is the minimum of F subject to:
    K   = integral A . B dV    (magnetic helicity, conserved in ideal MHD)
    Phi = integral B . dA      (magnetic flux through boundary)
    M   = integral rho dV      (total mass)

These quantities define the constrained manifold on which R = 1 lives.
Constraint enforcement during state flow evolution uses Lagrange multipliers.
"""

import numpy as np
from scipy.constants import mu_0

from ..numerics import diff_4th


# ── Magnetic vector potential ────────────────────────────────────────────────

def vector_potential_cylindrical(B, grid):
    """Compute the magnetic vector potential A for axisymmetric cylindrical
    fields by direct integration of B = curl A.

    For axisymmetric fields with d/dtheta = d/dz = 0:
        B_theta = -dA_z/dr           =>  A_z = -integral B_theta dr
        B_z     = (1/r) d(r A_theta)/dr

    We compute A_theta from B_z by integrating:
        r A_theta = integral r B_z dr
        A_theta   = (1/r) integral r B_z dr

    and A_z from B_theta by integrating:
        A_z = -integral B_theta dr

    Parameters
    ----------
    B : ndarray, shape (*grid.R.shape, 3)
        Magnetic field (B_r, B_theta, B_z).
    grid : CylindricalGrid

    Returns
    -------
    A : ndarray, shape (*grid.R.shape, 3)
        Magnetic vector potential (A_r, A_theta, A_z).
    """
    A = np.zeros_like(B)
    r = grid.r
    dr = grid.dr

    # We integrate along the r-axis (axis 0).
    # For a 3D grid shape (Nr, Ntheta, Nz), iterate over (theta, z) implicitly
    # using cumulative trapezoidal integration along axis 0.

    B_theta = B[..., 1]
    B_z = B[..., 2]

    # A_z = -integral_0^r B_theta(r') dr'  (cumulative trapezoid along axis 0)
    A_z = -np.cumsum(B_theta * dr, axis=0)
    # Adjust: subtract half of first and last bin (trapezoidal correction)
    A_z = A_z - 0.5 * B_theta * dr
    # Shift so A_z(r_min) ~ 0 (gauge choice)
    A[..., 2] = A_z

    # A_theta from B_z: r A_theta = integral_0^r r' B_z(r') dr'
    r_bcast = grid.R
    integrand = r_bcast * B_z
    rAtheta = np.cumsum(integrand * dr, axis=0) - 0.5 * integrand * dr
    A[..., 1] = rAtheta / np.maximum(r_bcast, 1e-30)

    return A


def vector_potential_toroidal_poloidal(BR, BZ, RR, dR, dZ):
    """Compute the toroidal component of the vector potential A_phi from
    the poloidal field (BR, BZ) on an (R, Z) grid.

    For axisymmetric fields:  B_pol = curl(A_phi e_phi)
        BR = -(1/R) d(R A_phi)/dZ  ...  approximately
        BZ = (1/R) d(R A_phi)/dR

    We use the stream function: psi = R * A_phi, so B_R = -dpsi/dZ / R,
    B_Z = dpsi/dR / R.  Therefore psi = integral B_Z R dR (along R at fixed Z),
    or equivalently psi = -integral B_R R dZ (along Z at fixed R).

    We use the R-integration (more stable for typical grids).

    Parameters
    ----------
    BR, BZ : ndarray (NR, NZ)
    RR : ndarray (NR, NZ)
    dR, dZ : float

    Returns
    -------
    A_phi : ndarray (NR, NZ)
        Toroidal vector potential component.
    psi : ndarray (NR, NZ)
        Poloidal flux function psi = R * A_phi.
    """
    # psi = integral_{R_min}^{R} BZ(R', Z) R' dR'
    integrand = BZ * RR
    psi = np.cumsum(integrand * dR, axis=0) - 0.5 * integrand * dR
    A_phi = psi / np.maximum(RR, 1e-30)
    return A_phi, psi


# ── Magnetic helicity ────────────────────────────────────────────────────────

def magnetic_helicity_cylindrical(B, grid, A=None):
    """Compute magnetic helicity K = integral A . B dV on a cylindrical grid.

    Parameters
    ----------
    B : ndarray, shape (*grid.R.shape, 3)
    grid : CylindricalGrid
    A : ndarray or None
        Pre-computed vector potential.  If None, computed from B.

    Returns
    -------
    K : float
        Total magnetic helicity (Wb^2).
    """
    from .free_energy import _volume_elements_cylindrical

    if A is None:
        A = vector_potential_cylindrical(B, grid)

    A_dot_B = np.sum(A * B, axis=-1)
    dV = _volume_elements_cylindrical(grid)
    K = float(np.sum(A_dot_B * dV))
    return K


def magnetic_helicity_toroidal(BR, BZ, Bphi, RR, dR, dZ, mask=None,
                               A_phi=None):
    """Compute magnetic helicity K = integral A . B dV on a toroidal grid.

    For axisymmetric fields: A . B = A_phi * B_phi + (poloidal A) . (poloidal B).
    The dominant contribution in a tokamak is from A_phi * B_phi (the linkage
    of toroidal and poloidal flux).

    For a more complete estimate we also include the poloidal stream function
    contribution: psi * J_phi / (mu_0 R).  For simplicity and physical
    transparency, we compute the gauge-invariant relative helicity as
    K = 2 * integral A_phi * B_phi * 2 pi R dR dZ.

    Parameters
    ----------
    BR, BZ, Bphi : ndarray (NR, NZ)
    RR : ndarray (NR, NZ)
    dR, dZ : float
    mask : ndarray of bool or None
    A_phi : ndarray (NR, NZ) or None
        Pre-computed toroidal vector potential.

    Returns
    -------
    K : float
        Magnetic helicity (Wb^2).
    """
    from .free_energy import _volume_elements_toroidal

    if A_phi is None:
        A_phi, _ = vector_potential_toroidal_poloidal(BR, BZ, RR, dR, dZ)

    # Dominant toroidal-poloidal linkage term
    integrand_density = A_phi * Bphi

    dV = _volume_elements_toroidal(RR, dR, dZ)
    integrand = integrand_density * dV
    if mask is not None:
        integrand = np.where(mask, integrand, 0.0)

    K = float(np.sum(integrand))
    return K


# ── Total magnetic flux ──────────────────────────────────────────────────────

def total_toroidal_flux(Bphi, RR, dR, dZ, mask=None):
    """Compute toroidal magnetic flux Phi_tor = integral B_phi dA (poloidal
    cross-section area).

    For axisymmetric geometry: Phi_tor = integral B_phi dR dZ (over poloidal
    cross section).  This is the flux per radian; multiply by 2 pi for total.

    Parameters
    ----------
    Bphi : ndarray (NR, NZ)
    RR : ndarray (NR, NZ)
    dR, dZ : float
    mask : ndarray of bool or None

    Returns
    -------
    Phi_tor : float
        Toroidal magnetic flux (Wb).
    """
    integrand = Bphi * dR * dZ
    if mask is not None:
        integrand = np.where(mask, integrand, 0.0)
    return float(np.sum(integrand))


def total_poloidal_flux_cylindrical(B, grid):
    """Compute axial (poloidal-equivalent) flux Phi = integral B_z dA on a
    cylindrical grid.

    Parameters
    ----------
    B : ndarray, shape (*grid.R.shape, 3)
    grid : CylindricalGrid

    Returns
    -------
    Phi : float
    """
    B_z = B[..., 2]
    dr = grid.dr
    dtheta = (grid.theta[1] - grid.theta[0]) if len(grid.theta) > 1 else 2 * np.pi
    dA = grid.R * dr * dtheta
    # Sum over r and theta, pick central z slice
    nz_mid = B_z.shape[2] // 2
    return float(np.sum(B_z[..., nz_mid] * dA[..., nz_mid]))


# ── Total mass ───────────────────────────────────────────────────────────────

def total_mass_cylindrical(rho, grid):
    """Compute total mass M = integral rho dV on a cylindrical grid.

    Parameters
    ----------
    rho : ndarray
        Mass density (kg/m^3).
    grid : CylindricalGrid

    Returns
    -------
    M : float
        Total mass (kg/m along z, or kg if z extent is physical).
    """
    from .free_energy import _volume_elements_cylindrical
    dV = _volume_elements_cylindrical(grid)
    return float(np.sum(np.asarray(rho) * dV))


def total_mass_toroidal(rho, RR, dR, dZ, mask=None):
    """Compute total mass M = integral rho dV on a toroidal grid.

    Parameters
    ----------
    rho : ndarray (NR, NZ)
    RR : ndarray (NR, NZ)
    dR, dZ : float
    mask : ndarray of bool or None

    Returns
    -------
    M : float
        Total mass (kg).
    """
    from .free_energy import _volume_elements_toroidal
    dV = _volume_elements_toroidal(RR, dR, dZ)
    integrand = np.asarray(rho) * dV
    if mask is not None:
        integrand = np.where(mask, integrand, 0.0)
    return float(np.sum(integrand))
