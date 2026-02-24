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

    In cylindrical axisymmetric coordinates with d/dtheta = d/dz = 0,
    the curl-based computation involves only first derivatives and avoids
    the subtraction of nearly-equal large quantities.

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

    B_theta = B[..., 1]
    B_z = B[..., 2]
    R = grid.R
    dr = grid.dr

    # curl B in cylindrical axisymmetric
    dBtheta_dr = diff_4th(B_theta, dr, axis=0)
    dBz_dr = diff_4th(B_z, dr, axis=0)

    curlB = np.zeros_like(B)
    curlB[..., 0] = 0
    curlB[..., 1] = -dBz_dr
    curlB[..., 2] = B_theta / R + dBtheta_dr

    J = curlB / mu_0
    f_lorentz = np.cross(J, B)

    # Diagnostic tension / pressure (not used for metrics, but useful)
    kappa_vec, kappa_mag = compute_curvature_cylindrical(B, grid)
    f_tension = (Bmag2 / mu_0)[..., np.newaxis] * kappa_vec

    dBmag2_dr = diff_4th(Bmag2, dr, axis=0)
    f_pressure = np.zeros_like(B)
    f_pressure[..., 0] = -dBmag2_dr / (2 * mu_0)

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
