"""
JAX field-line curvature: kappa = (b . nabla) b in cylindrical coords.

Full 3D with geometric correction terms.  Theta axis uses periodic
differentiation; r and z use non-periodic boundary stencils.
"""

import jax.numpy as jnp

from .numerics_jax import diff_4th_jax


def compute_curvature_cylindrical_jax(B, grid_info):
    """Curvature vector and magnitude for cylindrical fields.

    Parameters
    ----------
    B : jnp.ndarray, shape (Nr, Ntheta, Nz, 3)
        Magnetic field in (r, theta, z) components.
    grid_info : dict
        Static grid arrays from make_grid_info().

    Returns
    -------
    kappa_vec : jnp.ndarray, shape (Nr, Ntheta, Nz, 3)
    kappa_mag : jnp.ndarray, shape (Nr, Ntheta, Nz)
    """
    Bmag = jnp.sqrt(jnp.sum(B ** 2, axis=-1))
    Bmag = jnp.maximum(Bmag, 1e-30)

    b = B / Bmag[..., jnp.newaxis]
    b_r = b[..., 0]
    b_theta = b[..., 1]
    b_z = b[..., 2]

    R = grid_info['R']
    dr = grid_info['dr']
    Ntheta = grid_info['Ntheta']
    Nz = grid_info['Nz']

    # Radial advection: b_r * d/dr (always, non-periodic)
    db_r_dr = diff_4th_jax(b_r, dr, axis=0, periodic=False)
    db_theta_dr = diff_4th_jax(b_theta, dr, axis=0, periodic=False)
    db_z_dr = diff_4th_jax(b_z, dr, axis=0, periodic=False)

    kappa_r = b_r * db_r_dr
    kappa_theta = b_r * db_theta_dr
    kappa_z = b_r * db_z_dr

    # Theta advection: (b_theta / r) * d/dtheta (periodic)
    if Ntheta > 1:
        dtheta = grid_info['dtheta']
        db_r_dtheta = diff_4th_jax(b_r, dtheta, axis=1, periodic=True)
        db_theta_dtheta = diff_4th_jax(b_theta, dtheta, axis=1, periodic=True)
        db_z_dtheta = diff_4th_jax(b_z, dtheta, axis=1, periodic=True)

        kappa_r = kappa_r + (b_theta / R) * db_r_dtheta
        kappa_theta = kappa_theta + (b_theta / R) * db_theta_dtheta
        kappa_z = kappa_z + (b_theta / R) * db_z_dtheta

    # Z advection: b_z * d/dz (non-periodic)
    if Nz > 1:
        dz = grid_info['dz']
        db_r_dz = diff_4th_jax(b_r, dz, axis=2, periodic=False)
        db_theta_dz = diff_4th_jax(b_theta, dz, axis=2, periodic=False)
        db_z_dz = diff_4th_jax(b_z, dz, axis=2, periodic=False)

        kappa_r = kappa_r + b_z * db_r_dz
        kappa_theta = kappa_theta + b_z * db_theta_dz
        kappa_z = kappa_z + b_z * db_z_dz

    # Geometric terms (curvilinear basis)
    kappa_r = kappa_r - b_theta ** 2 / R
    kappa_theta = kappa_theta + b_r * b_theta / R

    kappa_vec = jnp.stack([kappa_r, kappa_theta, kappa_z], axis=-1)
    kappa_mag = jnp.sqrt(jnp.sum(kappa_vec ** 2, axis=-1))

    return kappa_vec, kappa_mag
