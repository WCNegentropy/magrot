"""Tension / pressure decomposition utilities."""

import numpy as np
from scipy.constants import mu_0


def decompose_inward_outward(f_lorentz_r, dp_dr=None):
    """Separate radial forces into inward and outward components.

    Parameters
    ----------
    f_lorentz_r : ndarray
        Radial component of J x B.
    dp_dr : ndarray or None
        Radial material pressure gradient.

    Returns
    -------
    total_inward, total_outward : ndarray
    """
    f_mag_inward = np.maximum(-f_lorentz_r, 0)
    f_mag_outward = np.maximum(f_lorentz_r, 0)

    if dp_dr is not None:
        f_mat_outward = np.maximum(-dp_dr, 0)
        f_mat_inward = np.maximum(dp_dr, 0)
    else:
        f_mat_outward = np.zeros_like(f_lorentz_r)
        f_mat_inward = np.zeros_like(f_lorentz_r)

    return f_mag_inward + f_mat_inward, f_mag_outward + f_mat_outward
