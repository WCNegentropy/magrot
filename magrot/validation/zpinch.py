"""Z-pinch analytic benchmarks."""

import numpy as np
from scipy.constants import mu_0

from ..fields.grid import CylindricalGrid
from ..fields.analytic import field_zpinch_bennett
from ..rotation.metrics import compute_all_metrics


def validate_zpinch(I=1e4, a=0.01, Nr=400, verbose=True):
    """Run Z-pinch Bennett equilibrium validation.

    Checks:
        - R_kappa(r=a) ~ 1.0
        - R_kappa(r=a/2) > 1  (contracting inside)
        - R_kappa(r=2a) < 1   (expanding outside)
        - R_universal(r=a) ~ 1.0
        - R_universal outside ~ 1.0 (vacuum equilibrium)

    Returns
    -------
    dict of check results (name -> passed bool).
    """
    grid = CylindricalGrid(Nr=Nr, Ntheta=1, Nz=1, r_range=(0.0003, 0.05))
    B, p_mat = field_zpinch_bennett(grid, I=I, a=a)

    def kappa_eq(g):
        return np.full(g.R.shape, 1.0 / a)

    res = compute_all_metrics(B, grid, p_mat=p_mat,
                              kappa_eq_func=kappa_eq, L_char=a)

    r = grid.r
    s = (slice(None), 0, 0)
    R_kappa = res['R_kappa'][s]
    R_universal = res['R_universal'][s]

    idx_a = np.argmin(np.abs(r - a))
    idx_half = np.argmin(np.abs(r - a / 2))
    idx_2a = np.argmin(np.abs(r - 2 * a))

    checks = {
        'R_kappa_at_a': abs(R_kappa[idx_a] - 1.0) < 0.02,
        'R_kappa_inside_gt1': R_kappa[idx_half] > 1.0,
        'R_kappa_outside_lt1': R_kappa[idx_2a] < 1.0,
        'R_universal_at_a': abs(R_universal[idx_a] - 1.0) < 0.05,
        'R_universal_outside': abs(np.mean(R_universal[r > 1.5 * a]) - 1.0) < 0.05,
    }

    if verbose:
        for name, passed in checks.items():
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    return checks
