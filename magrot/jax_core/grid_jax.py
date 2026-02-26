"""
Static grid arrays for JAX — no dataclass, no mutation.

The optimizer never differentiates through the grid, so these are
plain JAX arrays passed as static arguments to JIT-compiled functions.
"""

import jax.numpy as jnp
import numpy as np


def make_grid_info(Nr, Ntheta, Nz, r_range, theta_range=(0, 2 * np.pi),
                   z_range=(-1.0, 1.0)):
    """Build a dict of static grid arrays suitable for JAX.

    Parameters
    ----------
    Nr, Ntheta, Nz : int
        Grid resolution along each axis.
    r_range : tuple (r_min, r_max)
    theta_range : tuple (theta_min, theta_max)
    z_range : tuple (z_min, z_max)

    Returns
    -------
    info : dict
        Keys: r, theta, z, dr, dtheta, dz, R, THETA, Z, Nr, Ntheta, Nz.
        All arrays are jnp.ndarray (float64 where supported, float32 on GPU).
    """
    r = np.linspace(max(r_range[0], 1e-6), r_range[1], Nr)
    theta = np.linspace(theta_range[0], theta_range[1], Ntheta, endpoint=False)
    z = np.linspace(z_range[0], z_range[1], Nz)

    dr = float(r[1] - r[0]) if Nr > 1 else 1.0
    dtheta = float(theta[1] - theta[0]) if Ntheta > 1 else 2 * np.pi
    dz = float(z[1] - z[0]) if Nz > 1 else 1.0

    R, THETA, Z = np.meshgrid(r, theta, z, indexing='ij')

    return {
        'r': jnp.array(r),
        'theta': jnp.array(theta),
        'z': jnp.array(z),
        'dr': dr,
        'dtheta': dtheta,
        'dz': dz,
        'R': jnp.array(R),
        'THETA': jnp.array(THETA),
        'Z': jnp.array(Z),
        'Nr': Nr,
        'Ntheta': Ntheta,
        'Nz': Nz,
    }


def grid_info_from_numpy(grid):
    """Convert an existing CylindricalGrid to a JAX grid_info dict.

    Useful for parity testing: pass the same grid to both NumPy and JAX
    pipelines.
    """
    return {
        'r': jnp.array(grid.r),
        'theta': jnp.array(grid.theta),
        'z': jnp.array(grid.z),
        'dr': float(grid.dr),
        'dtheta': float(grid.dtheta),
        'dz': float(grid.dz),
        'R': jnp.array(grid.R),
        'THETA': jnp.array(grid.THETA),
        'Z': jnp.array(grid.Z),
        'Nr': grid.Nr,
        'Ntheta': grid.Ntheta,
        'Nz': grid.Nz,
    }
