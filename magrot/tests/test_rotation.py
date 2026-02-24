"""Tests for rotational parameter metrics."""

import numpy as np
import pytest

from magrot.fields.grid import CylindricalGrid
from magrot.fields.analytic import field_infinite_wire, field_zpinch_bennett
from magrot.rotation.metrics import compute_all_metrics


class TestWireMetrics:
    def test_universal_near_one_in_vacuum(self):
        """Vacuum wire should give R_universal ~ 1 (self-equilibrated)."""
        grid = CylindricalGrid(Nr=200, Ntheta=1, Nz=1, r_range=(0.001, 0.1))
        B, _ = field_infinite_wire(grid, I=1000.0)
        res = compute_all_metrics(B, grid)
        R_u = res['R_universal'][:, 0, 0]
        # Skip boundary region
        start = max(10, grid.Nr // 7)
        np.testing.assert_allclose(R_u[start:-10], 1.0, atol=0.05)


class TestZPinchMetrics:
    def test_kappa_at_boundary(self):
        """R_kappa should be ~1.0 at the Bennett equilibrium radius."""
        a = 0.01
        grid = CylindricalGrid(Nr=400, Ntheta=1, Nz=1, r_range=(0.0003, 0.05))
        B, p = field_zpinch_bennett(grid, I=1e4, a=a)

        def kappa_eq(g):
            return np.full(g.R.shape, 1.0 / a)

        res = compute_all_metrics(B, grid, p_mat=p, kappa_eq_func=kappa_eq, L_char=a)
        R_kappa = res['R_kappa'][:, 0, 0]
        idx_a = np.argmin(np.abs(grid.r - a))
        assert abs(R_kappa[idx_a] - 1.0) < 0.02
