"""
Field-line curvature computation.

Computes kappa = (b . nabla) b where b = B/|B| is the unit tangent
along the field line.  Uses 4th-order central finite differences with
2nd-order boundary fallback.
"""

import numpy as np

from ..numerics import diff_4th
from ..fields.grid import CylindricalGrid


def compute_curvature_cylindrical(B, grid: CylindricalGrid):
    """Curvature vector and magnitude for axisymmetric cylindrical fields.

    For purely azimuthal field (B_r=0, B_z=0):
        kappa_r = -b_theta^2 / r  (centripetal, inward)

    Parameters
    ----------
    B : ndarray, shape (*grid.R.shape, 3)
        Magnetic field in (r, theta, z) components.
    grid : CylindricalGrid

    Returns
    -------
    kappa_vec : ndarray, shape (*grid.R.shape, 3)
        Curvature vector.
    kappa_mag : ndarray, shape (*grid.R.shape)
        Scalar curvature |kappa|.
    """
    Bmag = np.sqrt(np.sum(B**2, axis=-1))
    Bmag = np.maximum(Bmag, 1e-30)

    b = B / Bmag[..., np.newaxis]
    b_r = b[..., 0]
    b_theta = b[..., 1]
    b_z = b[..., 2]

    dr = grid.dr

    db_r_dr = diff_4th(b_r, dr, axis=0)
    db_theta_dr = diff_4th(b_theta, dr, axis=0)
    db_z_dr = diff_4th(b_z, dr, axis=0)

    R = grid.R

    kappa_vec = np.zeros_like(B)
    kappa_vec[..., 0] = b_r * db_r_dr - b_theta**2 / R
    kappa_vec[..., 1] = b_r * db_theta_dr + b_r * b_theta / R
    kappa_vec[..., 2] = b_r * db_z_dr

    kappa_mag = np.sqrt(np.sum(kappa_vec**2, axis=-1))

    return kappa_vec, kappa_mag


def compute_curvature_cartesian_2d(bx, bz, dx, dz):
    """Curvature in 2D Cartesian (xz-plane).

    kappa = (b . nabla) b with b = (bx, bz).

    Parameters
    ----------
    bx, bz : ndarray (Nx, Nz)
        Unit tangent components.
    dx, dz : float
        Grid spacings.

    Returns
    -------
    kappa : ndarray (Nx, Nz)
        Scalar curvature magnitude.
    """
    dbx_dx = diff_4th(bx, dx, axis=0)
    dbx_dz = diff_4th(bx, dz, axis=1)
    dbz_dx = diff_4th(bz, dx, axis=0)
    dbz_dz = diff_4th(bz, dz, axis=1)

    kx = bx * dbx_dx + bz * dbx_dz
    kz = bx * dbz_dx + bz * dbz_dz
    return np.sqrt(kx**2 + kz**2)
