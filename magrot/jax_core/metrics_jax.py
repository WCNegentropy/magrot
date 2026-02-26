"""
JAX R_universal computation — the loss-critical metric.

All np.errstate context managers replaced with jnp.where guards.
All boolean masks are element-wise — fully JIT-compatible.

For the Forge optimizer, only R_universal matters (it's what the
Lyapunov loss integrates over).  The other three R metrics (kappa,
collapse, misalign) are diagnostic only and not ported to JAX.
"""

import jax.numpy as jnp
from scipy.constants import mu_0

from .maxwell_jax import compute_forces_conservative_cyl_jax
from .decompose_jax import decompose_inward_outward_jax


def compute_R_universal_jax(B, grid_info, dp_dr=None):
    """Compute R_universal = |f_inward| / |f_outward| with epsilon-floor.

    Parameters
    ----------
    B : jnp.ndarray, shape (Nr, Ntheta, Nz, 3)
    grid_info : dict
    dp_dr : jnp.ndarray or None

    Returns
    -------
    R_universal : jnp.ndarray, shape (Nr, Ntheta, Nz)
    phys : dict
        Physics intermediates (Bmag, curlB, J, f_lorentz, etc.)
    """
    phys = compute_forces_conservative_cyl_jax(B, grid_info)

    Bmag2 = phys['Bmag2']
    f_lorentz = phys['f_lorentz']
    f_mag_r = f_lorentz[..., 0]

    total_inward, total_outward = decompose_inward_outward_jax(f_mag_r, dp_dr)

    # Epsilon-floor regularisation (Fix #1)
    local_pB = Bmag2 / (2.0 * mu_0)
    significance = 1e-2
    abs_floor = jnp.max(local_pB) * 1e-12
    eps_floor = jnp.maximum(local_pB * significance, abs_floor)

    R_MAX = 100.0
    R_MIN = 0.01

    total_force = total_inward + total_outward
    is_quiet = total_force < eps_floor
    active = (~is_quiet) & (total_outward > 0.0) & (total_inward > 0.0)
    inward_dom = (~is_quiet) & (total_outward <= abs_floor) & (total_inward > eps_floor)
    outward_dom = (~is_quiet) & (total_inward <= abs_floor) & (total_outward > eps_floor)

    # Branchless R_universal via nested jnp.where
    R_universal = jnp.where(
        active,
        total_inward / jnp.maximum(total_outward, abs_floor),
        1.0  # default for quiet regions
    )
    R_universal = jnp.where(inward_dom, R_MAX, R_universal)
    R_universal = jnp.where(outward_dom, R_MIN, R_universal)
    R_universal = jnp.where(is_quiet, 1.0, R_universal)
    R_universal = jnp.clip(R_universal, R_MIN, R_MAX)

    phys['R_universal'] = R_universal
    phys['total_inward'] = total_inward
    phys['total_outward'] = total_outward

    return R_universal, phys
