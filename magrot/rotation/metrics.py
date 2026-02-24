"""
All four rotational parameter (R) definitions.

Metric A -- R_kappa:    kappa / kappa_eq  (curvature ratio)
Metric B -- R_collapse: 1 + C             (collapse indicator)
Metric C -- R_misalign: |B x curl(B)| / |B|^2  (current-field misalignment)
Universal -- R_universal: |f_inward| / |f_outward|

Engineering fixes from v3 are baked in:
    Fix #1  epsilon-floor regularization for R_universal
    Fix #2  conservative J x B for force-based metrics
    Fix #3  dynamic L0 = min(1/kappa, L_char) for Metric B
"""

import numpy as np
from scipy.constants import mu_0

from ..stress.maxwell import compute_forces_conservative_cyl
from ..stress.decompose import decompose_inward_outward
from ..fields.grid import CylindricalGrid


def compute_all_metrics(B, grid: CylindricalGrid, p_mat=None,
                        kappa_eq_func=None, L_char=None):
    """Compute all four R metrics for a cylindrical field configuration.

    Parameters
    ----------
    B : ndarray
        Magnetic field array.
    grid : CylindricalGrid
    p_mat : ndarray or None
        Material (plasma) pressure.
    kappa_eq_func : callable or None
        Function(grid) -> kappa_eq array for Metric A.
    L_char : float or None
        Characteristic length for Metric B normalisation.

    Returns
    -------
    dict
        Contains R_kappa, R_collapse, R_misalign, R_universal and
        auxiliary fields (Bmag, kappa_mag, forces, etc.).
    """
    phys = compute_forces_conservative_cyl(B, grid, p_mat)

    Bmag = phys['Bmag']
    Bmag2 = phys['Bmag2']
    kappa_mag = phys['kappa_mag']
    f_lorentz = phys['f_lorentz']
    f_mag_r = f_lorentz[..., 0]

    f_ref = np.max(Bmag2) / (2 * mu_0 * (grid.r[-1] - grid.r[0]))

    results = {**phys}

    # ── Metric A: Curvature Ratio ──
    if kappa_eq_func is not None:
        kappa_eq = np.maximum(kappa_eq_func(grid), 1e-30)
        results['R_kappa'] = kappa_mag / kappa_eq
    else:
        results['R_kappa'] = None

    # ── Metric B: Collapse Indicator (dynamic L0, Fix #3) ──
    f_net_r = f_mag_r.copy()
    if phys['dp_dr'] is not None:
        f_net_r -= phys['dp_dr']

    L_char_val = L_char if L_char is not None else (grid.r[-1] - grid.r[0])

    with np.errstate(divide='ignore', invalid='ignore'):
        L0_local = np.where(
            kappa_mag > 1e-10,
            np.minimum(1.0 / kappa_mag, L_char_val),
            L_char_val,
        )

    f0 = Bmag2 / (2 * mu_0 * L0_local)
    f0 = np.maximum(f0, f_ref * 1e-12)

    C = -f_net_r / f0
    results['R_collapse'] = 1 + C
    results['C_raw'] = C
    results['L0_local'] = L0_local

    # ── Metric C: Current-Field Misalignment ──
    R_vec = np.cross(B, phys['curlB']) / np.maximum(Bmag2, 1e-60)[..., np.newaxis]
    results['R_misalign'] = np.sqrt(np.sum(R_vec**2, axis=-1))
    results['R_misalign_vec'] = R_vec

    # ── Universal: |f_inward| / |f_outward| (Fix #1: epsilon-floor) ──
    total_inward, total_outward = decompose_inward_outward(
        f_mag_r, phys['dp_dr']
    )

    local_pB = Bmag2 / (2 * mu_0)
    significance = 1e-2
    abs_floor = np.max(local_pB) * 1e-12
    eps_floor = np.maximum(local_pB * significance, abs_floor)

    R_MAX, R_MIN = 100.0, 0.01
    total_force = total_inward + total_outward
    is_quiet = total_force < eps_floor

    R_universal = np.ones_like(f_mag_r)
    active = ~is_quiet & (total_outward > 0) & (total_inward > 0)
    inward_dom = ~is_quiet & (total_outward <= abs_floor) & (total_inward > eps_floor)
    outward_dom = ~is_quiet & (total_inward <= abs_floor) & (total_outward > eps_floor)

    with np.errstate(divide='ignore', invalid='ignore'):
        R_universal = np.where(
            active, total_inward / np.maximum(total_outward, abs_floor), R_universal
        )
    R_universal = np.where(inward_dom, R_MAX, R_universal)
    R_universal = np.where(outward_dom, R_MIN, R_universal)
    R_universal = np.where(is_quiet, 1.0, R_universal)
    R_universal = np.clip(R_universal, R_MIN, R_MAX)

    results['R_universal'] = R_universal
    results['total_inward'] = total_inward
    results['total_outward'] = total_outward

    return results
