"""Theta-pinch analytic benchmarks."""

import numpy as np

from ..fields.grid import CylindricalGrid
from ..fields.analytic import field_theta_pinch
from ..rotation.metrics import compute_all_metrics


def validate_theta_pinch(B0=1.0, a=0.01, beta=0.5, Nr=300, verbose=True):
    """Validate theta-pinch: curvature ~ 0, R_collapse ~ 1."""
    grid = CylindricalGrid(Nr=Nr, Ntheta=1, Nz=1, r_range=(0.0005, 0.05))
    B, p_mat = field_theta_pinch(grid, B0=B0, a=a, beta=beta)
    res = compute_all_metrics(B, grid, p_mat=p_mat, L_char=a)

    r = grid.r
    s = (slice(None), 0, 0)
    kappa = res['kappa_mag'][s]
    R_c = res['R_collapse'][s]

    idx_a = np.argmin(np.abs(r - a))

    checks = {
        'kappa_near_zero': kappa[idx_a] < 1.0,
        'R_collapse_near_1': abs(np.mean(R_c[10:-10]) - 1.0) < 0.1,
    }

    if verbose:
        for name, passed in checks.items():
            print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    return checks
