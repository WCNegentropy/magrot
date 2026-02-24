"""Tests for Maxwell stress / force computations."""

import numpy as np
import pytest
from scipy.constants import mu_0

from magrot.fields.grid import CylindricalGrid
from magrot.fields.analytic import field_infinite_wire
from magrot.stress.maxwell import compute_forces_conservative_cyl


class TestConservativeForces:
    def test_wire_lorentz_force_near_zero(self):
        """In vacuum (J=0), J x B should be ~0."""
        grid = CylindricalGrid(Nr=200, Ntheta=1, Nz=1, r_range=(0.001, 0.1))
        B, _ = field_infinite_wire(grid, I=1000.0)
        phys = compute_forces_conservative_cyl(B, grid)

        f_lor = phys['f_lorentz'][:, 0, 0, 0]
        f_tension = phys['f_tension'][:, 0, 0, 0]

        # Lorentz force should be much smaller than tension
        ratio = np.max(np.abs(f_lor[5:-5])) / np.max(np.abs(f_tension[5:-5]))
        assert ratio < 0.05
