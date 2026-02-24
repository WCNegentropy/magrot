"""
Analytic magnetic field generators for validation test cases.

Each generator returns (B, p_mat) where:
    B      -- magnetic field array, shape (*grid.shape, 3)
    p_mat  -- material pressure array or None, shape (*grid.shape)
"""

import numpy as np
from scipy.constants import mu_0

from .grid import CylindricalGrid


def field_infinite_wire(grid: CylindricalGrid, I: float = 1.0):
    """Infinite straight wire: B_theta = mu_0 I / (2 pi r).

    Parameters
    ----------
    grid : CylindricalGrid
    I : float
        Current in amperes.

    Returns
    -------
    B : ndarray, shape (*grid.R.shape, 3)
        (B_r, B_theta, B_z) components.
    p_mat : None
        No material pressure in vacuum.
    """
    B = np.zeros((*grid.R.shape, 3))
    B[..., 1] = mu_0 * I / (2 * np.pi * grid.R)
    return B, None


def field_zpinch_bennett(grid: CylindricalGrid, I: float = 1e4, a: float = 0.01):
    """Bennett Z-pinch equilibrium with corrected self-consistent pressure.

    Uniform current density J_z = I/(pi a^2) for r < a.
    Radial equilibrium: dp/dr = -J_z B_theta.

    Parameters
    ----------
    grid : CylindricalGrid
    I : float
        Total current (A).
    a : float
        Equilibrium pinch radius (m).

    Returns
    -------
    B : ndarray
    p_mat : ndarray
    """
    B = np.zeros((*grid.R.shape, 3))
    p_mat = np.zeros(grid.R.shape)

    inside = grid.R < a

    B[..., 1] = np.where(
        inside,
        mu_0 * I * grid.R / (2 * np.pi * a**2),
        mu_0 * I / (2 * np.pi * grid.R),
    )

    # Corrected pressure from integrating dp/dr = -J_z * B_theta
    p0 = mu_0 * I**2 / (4 * np.pi**2 * a**2)
    p_mat = np.where(inside, p0 * (1 - (grid.R / a) ** 2), 0.0)

    return B, p_mat


def field_theta_pinch(
    grid: CylindricalGrid, B0: float = 1.0, a: float = 0.01, beta: float = 0.5
):
    """Theta-pinch: axial B_z with diamagnetic plasma pressure.

    Field lines are straight (zero curvature). Confinement is purely
    by magnetic pressure gradient.

    Parameters
    ----------
    grid : CylindricalGrid
    B0 : float
        External axial field (T).
    a : float
        Plasma radius (m).
    beta : float
        Ratio of plasma to magnetic pressure inside.

    Returns
    -------
    B : ndarray
    p_mat : ndarray
    """
    B = np.zeros((*grid.R.shape, 3))
    p_mat = np.zeros(grid.R.shape)

    inside = grid.R < a
    B_inside = B0 * np.sqrt(1 - beta)

    B[..., 2] = np.where(inside, B_inside, B0)
    p_mat = np.where(inside, (B0**2 - B_inside**2) / (2 * mu_0), 0.0)

    return B, p_mat


def field_dipole_cartesian(X, Z, m=1.0, r_soft=0.003):
    """Magnetic dipole field in the xz-plane (y=0) with softened origin.

    Uses a softening radius to eliminate the origin singularity, producing
    smooth, NaN-free fields everywhere.

    Parameters
    ----------
    X, Z : ndarray
        Cartesian coordinate arrays in the xz-plane.
    m : float
        Magnetic dipole moment (A m^2).
    r_soft : float
        Softening radius (m) for origin regularization.

    Returns
    -------
    Bx, Bz : ndarray
        Field components.
    Bmag : ndarray
        Field magnitude.
    """
    r_eff = np.sqrt(X**2 + Z**2 + r_soft**2)
    Bx = mu_0 / (4 * np.pi) * m * 3 * X * Z / r_eff**5
    Bz = mu_0 / (4 * np.pi) * m * (3 * Z**2 - r_eff**2) / r_eff**5
    Bmag = np.sqrt(Bx**2 + Bz**2)
    return Bx, Bz, Bmag
