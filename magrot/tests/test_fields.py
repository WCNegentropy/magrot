"""Tests for analytic field generators."""

import numpy as np
import pytest
from scipy.constants import mu_0

from magrot.fields.grid import CylindricalGrid, CartesianGrid
from magrot.fields.analytic import (
    field_infinite_wire,
    field_zpinch_bennett,
    field_theta_pinch,
    field_dipole_cartesian,
)


@pytest.fixture
def cyl_grid():
    return CylindricalGrid(Nr=200, Ntheta=1, Nz=1, r_range=(0.001, 0.1))


class TestInfiniteWire:
    def test_field_magnitude(self, cyl_grid):
        I = 1000.0
        B, p = field_infinite_wire(cyl_grid, I=I)
        r = cyl_grid.r
        Bmag = np.sqrt(np.sum(B**2, axis=-1))[:, 0, 0]
        B_analytic = mu_0 * I / (2 * np.pi * r)
        np.testing.assert_allclose(Bmag, B_analytic, rtol=1e-10)

    def test_no_pressure(self, cyl_grid):
        _, p = field_infinite_wire(cyl_grid, I=100.0)
        assert p is None


class TestZPinch:
    def test_pressure_peaks_at_center(self):
        grid = CylindricalGrid(Nr=300, Ntheta=1, Nz=1, r_range=(0.0003, 0.05))
        _, p = field_zpinch_bennett(grid, I=1e4, a=0.01)
        # Pressure should be highest near r=0, zero at r >= a
        assert p[0, 0, 0] > p[-1, 0, 0]

    def test_pressure_at_center(self):
        a = 0.01
        I = 1e4
        grid = CylindricalGrid(Nr=300, Ntheta=1, Nz=1, r_range=(0.0003, 0.05))
        _, p = field_zpinch_bennett(grid, I=I, a=a)
        p0_expected = mu_0 * I**2 / (4 * np.pi**2 * a**2)
        assert abs(p[0, 0, 0] - p0_expected) / p0_expected < 0.05


class TestThetaPinch:
    def test_no_azimuthal_field(self):
        grid = CylindricalGrid(Nr=100, Ntheta=1, Nz=1, r_range=(0.001, 0.05))
        B, _ = field_theta_pinch(grid, B0=1.0, a=0.01, beta=0.5)
        assert np.allclose(B[..., 1], 0.0)  # B_theta = 0

    def test_pressure_balance(self):
        grid = CylindricalGrid(Nr=100, Ntheta=1, Nz=1, r_range=(0.001, 0.05))
        B0 = 1.0
        beta = 0.5
        B, p = field_theta_pinch(grid, B0=B0, a=0.01, beta=beta)
        # Inside: p + B^2/(2mu0) should equal B0^2/(2mu0)
        inside = grid.R < 0.01
        total = p[inside] + np.sum(B[inside]**2, axis=-1) / (2 * mu_0)
        expected = B0**2 / (2 * mu_0)
        np.testing.assert_allclose(total, expected, rtol=1e-10)


class TestDipole:
    def test_no_nans(self):
        x = np.linspace(-0.1, 0.1, 50)
        X, Z = np.meshgrid(x, x, indexing='ij')
        Bx, Bz, Bmag = field_dipole_cartesian(X, Z, m=1.0, r_soft=0.003)
        assert not np.any(np.isnan(Bmag))
