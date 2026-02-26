"""
JAX Maxwell stress and conservative J x B force computation.

Full 3D cylindrical curl(B) with periodic theta differentiation.
"""

import jax.numpy as jnp
from scipy.constants import mu_0

from .numerics_jax import diff_4th_jax
from .curvature_jax import compute_curvature_cylindrical_jax


def compute_forces_conservative_cyl_jax(B, grid_info):
    """Compute curl(B), J, J x B, curvature, and pressure gradient.

    Parameters
    ----------
    B : jnp.ndarray, shape (Nr, Ntheta, Nz, 3)
    grid_info : dict

    Returns
    -------
    dict with keys: f_lorentz, f_tension, f_pressure, kappa_vec,
        kappa_mag, Bmag, Bmag2, curlB, J
    """
    Bmag2 = jnp.sum(B ** 2, axis=-1)
    Bmag = jnp.sqrt(jnp.maximum(Bmag2, 1e-60))

    B_r = B[..., 0]
    B_theta = B[..., 1]
    B_z = B[..., 2]
    R = grid_info['R']
    dr = grid_info['dr']
    Ntheta = grid_info['Ntheta']
    Nz = grid_info['Nz']

    # curl(B) in cylindrical coordinates
    dBtheta_dr = diff_4th_jax(B_theta, dr, axis=0, periodic=False)
    dBz_dr = diff_4th_jax(B_z, dr, axis=0, periodic=False)

    # Start with axisymmetric terms
    curl_r = jnp.zeros_like(B_r)
    curl_theta = -dBz_dr
    curl_z = B_theta / R + dBtheta_dr

    # Theta-derivative terms (periodic)
    if Ntheta > 1:
        dtheta = grid_info['dtheta']
        dBz_dtheta = diff_4th_jax(B_z, dtheta, axis=1, periodic=True)
        dBr_dtheta = diff_4th_jax(B_r, dtheta, axis=1, periodic=True)
        curl_r = curl_r + (1.0 / R) * dBz_dtheta
        curl_z = curl_z - (1.0 / R) * dBr_dtheta

    # Z-derivative terms (non-periodic)
    if Nz > 1:
        dz = grid_info['dz']
        dBtheta_dz = diff_4th_jax(B_theta, dz, axis=2, periodic=False)
        dBr_dz = diff_4th_jax(B_r, dz, axis=2, periodic=False)
        curl_r = curl_r - dBtheta_dz
        curl_theta = curl_theta + dBr_dz

    curlB = jnp.stack([curl_r, curl_theta, curl_z], axis=-1)
    J = curlB / mu_0
    f_lorentz = jnp.cross(J, B)

    # Curvature (for tension diagnostic)
    kappa_vec, kappa_mag = compute_curvature_cylindrical_jax(B, grid_info)
    f_tension = (Bmag2 / mu_0)[..., jnp.newaxis] * kappa_vec

    # Magnetic pressure gradient
    dBmag2_dr = diff_4th_jax(Bmag2, dr, axis=0, periodic=False)
    fp_r = -dBmag2_dr / (2.0 * mu_0)
    fp_theta = jnp.zeros_like(B_r)
    fp_z = jnp.zeros_like(B_r)

    if Ntheta > 1:
        dBmag2_dtheta = diff_4th_jax(Bmag2, grid_info['dtheta'], axis=1, periodic=True)
        fp_theta = -(1.0 / R) * dBmag2_dtheta / (2.0 * mu_0)

    if Nz > 1:
        dBmag2_dz = diff_4th_jax(Bmag2, grid_info['dz'], axis=2, periodic=False)
        fp_z = -dBmag2_dz / (2.0 * mu_0)

    f_pressure = jnp.stack([fp_r, fp_theta, fp_z], axis=-1)

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
    }
