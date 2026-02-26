"""
Maxwell stress tensor and conservative force computation.

Uses J x B (Lorentz force) as the primary force computation rather than
tension - pressure subtraction, which suffers from catastrophic cancellation
when both are large and nearly equal (v3 Fix #2).
"""

import numpy as np
from scipy.constants import mu_0

from ..numerics import diff_4th
from ..fields.grid import CylindricalGrid


def compute_forces_conservative_cyl(B, grid: CylindricalGrid, p_mat=None):
    """Compute magnetic forces using conservative J x B form.

    Supports full 3D cylindrical coordinates.  When the grid has
    Ntheta > 1 or Nz > 1 the complete curl(B) is computed; otherwise
    the axisymmetric (d/dtheta = d/dz = 0) shortcut is used.

    Parameters
    ----------
    B : ndarray, shape (*grid.R.shape, 3)
    grid : CylindricalGrid
    p_mat : ndarray or None
        Material pressure.

    Returns
    -------
    dict with keys:
        f_lorentz, f_tension, f_pressure, kappa_vec, kappa_mag,
        Bmag, Bmag2, curlB, J, dp_dr
    """
    from ..geometry.curvature import compute_curvature_cylindrical

    Bmag2 = np.sum(B**2, axis=-1)
    Bmag = np.sqrt(np.maximum(Bmag2, 1e-60))

    B_r = B[..., 0]
    B_theta = B[..., 1]
    B_z = B[..., 2]
    R = grid.R
    dr = grid.dr

    # ── curl(B) in cylindrical coordinates ──
    # Radial derivatives (always needed)
    dBtheta_dr = diff_4th(B_theta, dr, axis=0)
    dBz_dr = diff_4th(B_z, dr, axis=0)

    curlB = np.zeros_like(B)

    # Start with axisymmetric terms
    curlB[..., 1] = -dBz_dr
    curlB[..., 2] = B_theta / R + dBtheta_dr

    # Add theta-derivative terms when Ntheta > 1
    if grid.Ntheta > 1:
        dBz_dtheta = diff_4th(B_z, grid.dtheta, axis=1)
        dBr_dtheta = diff_4th(B_r, grid.dtheta, axis=1)
        curlB[..., 0] += (1.0 / R) * dBz_dtheta
        curlB[..., 2] -= (1.0 / R) * dBr_dtheta

    # Add z-derivative terms when Nz > 1
    if grid.Nz > 1:
        dBtheta_dz = diff_4th(B_theta, grid.dz, axis=2)
        dBr_dz = diff_4th(B_r, grid.dz, axis=2)
        curlB[..., 0] -= dBtheta_dz
        curlB[..., 1] += dBr_dz

    J = curlB / mu_0
    f_lorentz = np.cross(J, B)

    # Diagnostic tension / pressure (not used for metrics, but useful)
    kappa_vec, kappa_mag = compute_curvature_cylindrical(B, grid)
    f_tension = (Bmag2 / mu_0)[..., np.newaxis] * kappa_vec

    # Magnetic pressure gradient
    dBmag2_dr = diff_4th(Bmag2, dr, axis=0)
    f_pressure = np.zeros_like(B)
    f_pressure[..., 0] = -dBmag2_dr / (2 * mu_0)

    if grid.Ntheta > 1:
        dBmag2_dtheta = diff_4th(Bmag2, grid.dtheta, axis=1)
        f_pressure[..., 1] = -(1.0 / R) * dBmag2_dtheta / (2 * mu_0)

    if grid.Nz > 1:
        dBmag2_dz = diff_4th(Bmag2, grid.dz, axis=2)
        f_pressure[..., 2] = -dBmag2_dz / (2 * mu_0)

    # Material pressure gradient
    dp_dr = None
    if p_mat is not None:
        dp_dr = diff_4th(p_mat, dr, axis=0)

    return {
        'f_lorentz': f_lorentz,
        'f_tension': f_tension,
        'f_pressure': f_pressure,
        'kappa_vec': kappa_vec,
        'kappa_mag': kappa_mag,
        'Bmag': Bmag,
        'Bmag2': Bmag2,
        'curlB': curlB,
        'J': J,
        'dp_dr': dp_dr,
    }
