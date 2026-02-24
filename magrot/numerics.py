"""
Shared numerical utilities.

Provides the 4th-order central finite-difference stencil used throughout
the framework.  Boundary points fall back to 2nd-order one-sided stencils.
"""

import numpy as np


def diff_4th(arr, dx, axis=0):
    """4th-order central finite difference along *axis*.

    Interior:
        df/dx ~ (-f[i+2] + 8f[i+1] - 8f[i-1] + f[i-2]) / (12 dx)
    Boundaries (2nd-order one-sided):
        df/dx ~ (-3f[0] + 4f[1] - f[2]) / (2 dx)   (left)
        df/dx ~  (3f[-1] - 4f[-2] + f[-3]) / (2 dx) (right)

    Parameters
    ----------
    arr : ndarray
        Input array.
    dx : float
        Grid spacing along *axis*.
    axis : int
        Axis along which to differentiate.

    Returns
    -------
    result : ndarray
        Same shape as *arr*.
    """
    result = np.zeros_like(arr)
    N = arr.shape[axis]

    def sl(ax, s):
        slices = [slice(None)] * arr.ndim
        slices[ax] = s
        return tuple(slices)

    if N >= 5:
        # 4th-order interior
        result[sl(axis, slice(2, -2))] = (
            -arr[sl(axis, slice(4, None))]
            + 8 * arr[sl(axis, slice(3, -1))]
            - 8 * arr[sl(axis, slice(1, -3))]
            + arr[sl(axis, slice(None, -4))]
        ) / (12 * dx)
        # 2nd-order boundaries
        result[sl(axis, 0)] = (
            -3 * arr[sl(axis, 0)] + 4 * arr[sl(axis, 1)] - arr[sl(axis, 2)]
        ) / (2 * dx)
        result[sl(axis, 1)] = (arr[sl(axis, 2)] - arr[sl(axis, 0)]) / (2 * dx)
        result[sl(axis, -1)] = (
            3 * arr[sl(axis, -1)] - 4 * arr[sl(axis, -2)] + arr[sl(axis, -3)]
        ) / (2 * dx)
        result[sl(axis, -2)] = (arr[sl(axis, -1)] - arr[sl(axis, -3)]) / (2 * dx)
    elif N >= 3:
        result[sl(axis, slice(1, -1))] = (
            arr[sl(axis, slice(2, None))] - arr[sl(axis, slice(None, -2))]
        ) / (2 * dx)
        result[sl(axis, 0)] = (arr[sl(axis, 1)] - arr[sl(axis, 0)]) / dx
        result[sl(axis, -1)] = (arr[sl(axis, -1)] - arr[sl(axis, -2)]) / dx

    return result
