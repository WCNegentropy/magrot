"""
JAX-accelerated numerical utilities for MagRot Forge.

Provides the 4th-order central finite-difference stencil using pure
functional JAX operations (no in-place mutations).  Includes native
support for periodic boundaries (theta axis in 3D topologies).

When periodic=True, jnp.roll handles wrapping and every point gets
4th-order accuracy.  When periodic=False, boundary points use 2nd-order
one-sided stencils applied via jnp.where (branchless, JIT-safe).
"""

import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnums=(2, 3))
def diff_4th_jax(arr, dx, axis=0, periodic=False):
    """4th-order central finite difference along *axis*.

    Interior (4th-order):
        df/dx ~ (-f[i+2] + 8f[i+1] - 8f[i-1] + f[i-2]) / (12 dx)

    Parameters
    ----------
    arr : jnp.ndarray
        Input array.
    dx : float
        Grid spacing along *axis*.
    axis : int
        Axis along which to differentiate.
    periodic : bool
        If True, wrap at boundaries (theta axis).
        If False, 2nd-order one-sided stencils at edges.

    Returns
    -------
    result : jnp.ndarray
        Derivative array, same shape as *arr*.
    """
    N = arr.shape[axis]

    # 4th-order central stencil — jnp.roll wraps at edges
    f_p2 = jnp.roll(arr, -2, axis=axis)
    f_p1 = jnp.roll(arr, -1, axis=axis)
    f_m1 = jnp.roll(arr, 1, axis=axis)
    f_m2 = jnp.roll(arr, 2, axis=axis)

    central = (-f_p2 + 8.0 * f_p1 - 8.0 * f_m1 + f_m2) / (12.0 * dx)

    if periodic:
        return central

    # Non-periodic: fix 4 boundary points with 2nd-order stencils.
    # Build index array along the target axis for jnp.where masking.
    idx = jax.lax.broadcasted_iota(jnp.int32, arr.shape, dimension=axis)

    # Helper: slice_in_dim keeps the axis dimension (size 1) so it
    # broadcasts correctly against the full-shape idx array.
    def _slice(i):
        return jax.lax.slice_in_dim(arr, i, i + 1, axis=axis)

    # 2nd-order forward/backward stencils
    bnd_0 = (-3.0 * _slice(0) + 4.0 * _slice(1) - _slice(2)) / (2.0 * dx)
    bnd_1 = (_slice(2) - _slice(0)) / (2.0 * dx)
    bnd_Nm2 = (_slice(N - 1) - _slice(N - 3)) / (2.0 * dx)
    bnd_Nm1 = (3.0 * _slice(N - 1) - 4.0 * _slice(N - 2) + _slice(N - 3)) / (2.0 * dx)

    result = jnp.where(idx == 0, bnd_0, central)
    result = jnp.where(idx == 1, bnd_1, result)
    result = jnp.where(idx == N - 2, bnd_Nm2, result)
    result = jnp.where(idx == N - 1, bnd_Nm1, result)

    return result
