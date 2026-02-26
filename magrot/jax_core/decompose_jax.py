"""
JAX inward / outward force decomposition.

Trivial port — np.maximum → jnp.maximum, np.zeros_like → jnp.zeros_like.
"""

import jax.numpy as jnp


def decompose_inward_outward_jax(f_lorentz_r, dp_dr=None):
    """Separate radial forces into inward and outward components.

    Parameters
    ----------
    f_lorentz_r : jnp.ndarray
        Radial component of J x B.
    dp_dr : jnp.ndarray or None
        Radial material pressure gradient.

    Returns
    -------
    total_inward, total_outward : jnp.ndarray
    """
    f_mag_inward = jnp.maximum(-f_lorentz_r, 0.0)
    f_mag_outward = jnp.maximum(f_lorentz_r, 0.0)

    if dp_dr is not None:
        f_mat_outward = jnp.maximum(-dp_dr, 0.0)
        f_mat_inward = jnp.maximum(dp_dr, 0.0)
    else:
        f_mat_outward = jnp.zeros_like(f_lorentz_r)
        f_mat_inward = jnp.zeros_like(f_lorentz_r)

    return f_mag_inward + f_mat_inward, f_mag_outward + f_mat_outward
