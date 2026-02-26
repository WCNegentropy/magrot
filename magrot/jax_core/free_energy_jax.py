"""
JAX free energy functional F[B, p] and Lyapunov L_R = integral |R-1|^2 dV.

These are the actual loss functions for the Forge optimizer.
Both are differentiable via jax.grad and JIT-compiled.
"""

import jax.numpy as jnp
from scipy.constants import mu_0

import numpy as np


# ── Volume elements ──────────────────────────────────────────────────────

def _volume_elements_jax(grid_info):
    """dV = r * dr * dtheta * dz for a cylindrical grid."""
    dr = grid_info['dr']
    # Ntheta/Nz are plain Python ints in grid_info — safe for if-checks.
    # dtheta/dz are Python floats — also safe.
    Ntheta = grid_info['Ntheta']
    Nz = grid_info['Nz']
    dtheta = grid_info['dtheta'] if Ntheta > 1 else 2.0 * np.pi
    dz = grid_info['dz'] if Nz > 1 else 1.0
    return grid_info['R'] * dr * dtheta * dz


# ── Energy densities ─────────────────────────────────────────────────────

def magnetic_energy_density_jax(B):
    """B^2 / (2 mu_0)."""
    Bmag2 = jnp.sum(B ** 2, axis=-1)
    return Bmag2 / (2.0 * mu_0)


def total_energy_density_jax(B, p=None, gamma=5.0 / 3.0):
    """Magnetic + thermal energy density."""
    u = magnetic_energy_density_jax(B)
    if p is not None:
        u = u + jnp.asarray(p) / (gamma - 1.0)
    return u


# ── Integrated free energy ───────────────────────────────────────────────

def free_energy_cylindrical_jax(B, grid_info, p=None, gamma=5.0 / 3.0,
                                 F_eq=0.0):
    """Total free energy F[B, p] on a cylindrical grid.

    Parameters
    ----------
    B : jnp.ndarray, shape (Nr, Ntheta, Nz, 3)
    grid_info : dict
    p, gamma, F_eq : as in NumPy version

    Returns
    -------
    F : scalar (jnp float)
    u : jnp.ndarray  (energy density field)
    """
    u = total_energy_density_jax(B, p, gamma)
    dV = _volume_elements_jax(grid_info)
    F = jnp.sum(u * dV) - F_eq
    return F, u


# ── Lyapunov functional L_R ─────────────────────────────────────────────

def lyapunov_R_jax(R_field, grid_info):
    """L = integral |R - 1|^2 dV  —  the primary Forge loss.

    Parameters
    ----------
    R_field : jnp.ndarray
        R_universal (or any R metric) field.
    grid_info : dict

    Returns
    -------
    L : scalar (jnp float)
    deviation : jnp.ndarray  ( |R-1|^2 field )
    """
    deviation = (R_field - 1.0) ** 2
    dV = _volume_elements_jax(grid_info)
    L = jnp.sum(deviation * dV)
    return L, deviation


# ── Local free-energy gradient (R-F connection) ─────────────────────────

def local_free_energy_gradient_jax(R_field, B):
    """|delta F / delta B| ~ |R - 1| * B^2 / (2 mu_0)."""
    Bmag2 = jnp.sum(B ** 2, axis=-1)
    return jnp.abs(R_field - 1.0) * Bmag2 / (2.0 * mu_0)
