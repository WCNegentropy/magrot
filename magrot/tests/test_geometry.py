"""Tests for curvature computation."""

import numpy as np
import pytest

from magrot.fields.grid import CylindricalGrid
from magrot.fields.analytic import field_infinite_wire
from magrot.geometry.curvature import compute_curvature_cylindrical


class TestCurvatureCylindrical:
    def test_wire_curvature_matches_1_over_r(self):
        """For circular field lines, kappa should equal 1/r."""
        grid = CylindricalGrid(Nr=200, Ntheta=1, Nz=1, r_range=(0.001, 0.1))
        B, _ = field_infinite_wire(grid, I=1000.0)
        _, kappa_mag = compute_curvature_cylindrical(B, grid)

        r = grid.r
        kappa = kappa_mag[:, 0, 0]
        kappa_analytic = 1.0 / r

        # Skip boundary points where FD is lower order
        np.testing.assert_allclose(
            kappa[5:-5], kappa_analytic[5:-5], rtol=0.01
        )
