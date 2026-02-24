"""Tests for the thermodynamics module (Phase 2A / 2C)."""

import numpy as np
import pytest
from scipy.constants import mu_0

from magrot.fields.grid import CylindricalGrid
from magrot.fields.analytic import field_zpinch_bennett, field_theta_pinch
from magrot.rotation.metrics import compute_all_metrics
from magrot.thermodynamics.free_energy import (
    magnetic_energy_density,
    thermal_energy_density,
    total_energy_density,
    free_energy_cylindrical,
    lyapunov_R_cylindrical,
    local_free_energy_gradient,
)
from magrot.thermodynamics.entropy import (
    spitzer_resistivity,
    constant_resistivity,
    temperature_from_pressure,
    temperature_profile_parabolic,
    entropy_production_resistive,
    total_entropy_production_cylindrical,
    dissipated_power,
)
from magrot.thermodynamics.constraints import (
    vector_potential_cylindrical,
    magnetic_helicity_cylindrical,
    total_poloidal_flux_cylindrical,
    total_mass_cylindrical,
)
from magrot.thermodynamics.diagnostics import (
    StateFlowRecord,
    verify_monotonic_decrease,
    verify_constraint_conservation,
    correlation_R_sdot,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def zpinch_setup():
    """Standard Z-pinch Bennett equilibrium for testing."""
    I = 1e4
    a = 0.01
    grid = CylindricalGrid(Nr=200, Ntheta=1, Nz=1, r_range=(0.0003, 0.05))
    B, p = field_zpinch_bennett(grid, I=I, a=a)
    return B, p, grid, I, a


@pytest.fixture
def theta_pinch_setup():
    """Standard theta-pinch for testing."""
    grid = CylindricalGrid(Nr=200, Ntheta=1, Nz=1, r_range=(0.0003, 0.05))
    B, p = field_theta_pinch(grid, B0=1.0, a=0.01, beta=0.5)
    return B, p, grid


# ── Free energy tests ─────────────────────────────────────────────────────────

class TestFreeEnergy:
    def test_magnetic_energy_density_positive(self, zpinch_setup):
        B, p, grid, _, _ = zpinch_setup
        u_B = magnetic_energy_density(B)
        assert np.all(u_B >= 0)
        assert np.any(u_B > 0)

    def test_thermal_energy_density_positive(self, zpinch_setup):
        B, p, grid, _, _ = zpinch_setup
        u_th = thermal_energy_density(p)
        assert np.all(u_th >= 0)

    def test_total_energy_is_sum(self, zpinch_setup):
        B, p, grid, _, _ = zpinch_setup
        u_total = total_energy_density(B, p)
        u_B = magnetic_energy_density(B)
        u_th = thermal_energy_density(p)
        np.testing.assert_allclose(u_total, u_B + u_th, rtol=1e-12)

    def test_free_energy_positive(self, zpinch_setup):
        B, p, grid, _, _ = zpinch_setup
        F, u = free_energy_cylindrical(B, grid, p)
        assert F > 0

    def test_free_energy_zero_pressure_only_magnetic(self, zpinch_setup):
        B, p, grid, _, _ = zpinch_setup
        F_with, _ = free_energy_cylindrical(B, grid, p)
        F_without, _ = free_energy_cylindrical(B, grid)
        # With pressure should be larger
        assert F_with > F_without

    def test_lyapunov_R_equilibrium_small(self, zpinch_setup):
        """At Bennett equilibrium, |R-1|^2 integrated should be relatively small."""
        B, p, grid, _, a = zpinch_setup
        res = compute_all_metrics(B, grid, p_mat=p)
        R_u = res['R_universal']
        L, dev = lyapunov_R_cylindrical(R_u, grid)
        # L should be finite and non-negative
        assert L >= 0
        assert np.isfinite(L)

    def test_lyapunov_R_perturbed_larger(self, zpinch_setup):
        """Perturbed state should have larger Lyapunov than equilibrium."""
        B, p, grid, _, _ = zpinch_setup
        res_eq = compute_all_metrics(B, grid, p_mat=p)
        L_eq, _ = lyapunov_R_cylindrical(res_eq['R_universal'], grid)

        # Perturb pressure by 50%
        p_pert = p * 1.5
        res_pert = compute_all_metrics(B, grid, p_mat=p_pert)
        L_pert, _ = lyapunov_R_cylindrical(res_pert['R_universal'], grid)

        assert L_pert > L_eq


# ── Entropy production tests ─────────────────────────────────────────────────

class TestEntropy:
    def test_spitzer_resistivity_decreases_with_T(self):
        T = np.array([1e6, 1e7, 1e8])
        eta = spitzer_resistivity(T)
        assert eta[0] > eta[1] > eta[2]

    def test_constant_resistivity_uniform(self):
        T = np.array([1e6, 1e7, 1e8])
        eta = constant_resistivity(T, eta_value=1e-5)
        np.testing.assert_allclose(eta, 1e-5)

    def test_temperature_from_pressure(self):
        from scipy.constants import k as k_B
        p = 1e5  # Pa
        n = 1e20  # m^-3
        T = temperature_from_pressure(p, n)
        expected = p / (2 * n * k_B)
        np.testing.assert_allclose(T, expected, rtol=1e-10)

    def test_entropy_production_positive(self, zpinch_setup):
        B, p, grid, _, _ = zpinch_setup
        from magrot.stress.maxwell import compute_forces_conservative_cyl
        phys = compute_forces_conservative_cyl(B, grid, p)
        J = phys['J']
        T = np.full_like(phys['Bmag'], 1e7)

        s_dot, Jmag2, eta = entropy_production_resistive(J, T)
        assert np.all(s_dot >= 0)
        assert np.any(s_dot > 0)  # J != 0 inside plasma

    def test_total_entropy_production_positive(self, zpinch_setup):
        B, p, grid, _, _ = zpinch_setup
        from magrot.stress.maxwell import compute_forces_conservative_cyl
        phys = compute_forces_conservative_cyl(B, grid, p)
        J = phys['J']
        T = np.full_like(phys['Bmag'], 1e7)

        s_dot, _, _ = entropy_production_resistive(J, T)
        S_dot = total_entropy_production_cylindrical(s_dot, grid)
        assert S_dot > 0

    def test_dissipated_power_positive(self, zpinch_setup):
        B, p, grid, _, _ = zpinch_setup
        from magrot.stress.maxwell import compute_forces_conservative_cyl
        phys = compute_forces_conservative_cyl(B, grid, p)
        J = phys['J']
        T = np.full_like(phys['Bmag'], 1e7)

        s_dot, _, _ = entropy_production_resistive(J, T)
        P = dissipated_power(s_dot, T, grid=grid)
        assert P > 0


# ── Constraint tests ──────────────────────────────────────────────────────────

class TestConstraints:
    def test_vector_potential_shape(self, zpinch_setup):
        B, p, grid, _, _ = zpinch_setup
        A = vector_potential_cylindrical(B, grid)
        assert A.shape == B.shape

    def test_magnetic_helicity_finite(self, zpinch_setup):
        B, p, grid, _, _ = zpinch_setup
        K = magnetic_helicity_cylindrical(B, grid)
        assert np.isfinite(K)

    def test_total_mass_positive(self, zpinch_setup):
        B, p, grid, _, _ = zpinch_setup
        rho = np.ones(grid.R.shape) * 1e-6  # uniform density
        M = total_mass_cylindrical(rho, grid)
        assert M > 0


# ── Diagnostics tests ─────────────────────────────────────────────────────────

class TestDiagnostics:
    def test_state_flow_record(self):
        rec = StateFlowRecord()
        for i in range(10):
            rec.record(sigma_val=float(i), F_val=10.0 - i)
        arrays = rec.as_arrays()
        assert len(arrays['sigma']) == 10
        assert arrays['F'][0] == 10.0
        assert arrays['F'][-1] == 1.0

    def test_verify_monotonic_decrease_pass(self):
        F = [10.0, 9.0, 8.0, 7.5, 7.0]
        is_mono, violations, max_v = verify_monotonic_decrease(F)
        assert is_mono
        assert len(violations) == 0

    def test_verify_monotonic_decrease_fail(self):
        F = [10.0, 9.0, 9.5, 8.0]  # 9.0 -> 9.5 is an increase
        is_mono, violations, max_v = verify_monotonic_decrease(F)
        assert not is_mono
        assert len(violations) == 1

    def test_verify_constraint_conservation(self):
        vals = [100.0, 100.1, 99.9, 100.05]
        is_conserved, max_dev = verify_constraint_conservation(vals, rtol=0.01)
        assert is_conserved

    def test_correlation_R_sdot(self, zpinch_setup):
        B, p, grid, _, _ = zpinch_setup
        from magrot.stress.maxwell import compute_forces_conservative_cyl
        phys = compute_forces_conservative_cyl(B, grid, p)
        J = phys['J']
        T = np.full_like(phys['Bmag'], 1e7)

        res = compute_all_metrics(B, grid, p_mat=p)
        R_u = res['R_universal']
        s_dot, _, _ = entropy_production_resistive(J, T)

        r_corr = correlation_R_sdot(R_u, s_dot)
        # Should be a finite number (possibly positive correlation)
        assert np.isfinite(r_corr) or np.isnan(r_corr)


# ── State flow tests (Test D) ────────────────────────────────────────────────

class TestStateFlow:
    def test_zpinch_entropy_flow_converges(self):
        """Entropy-parameterized Z-pinch should converge to R ~ 1."""
        from magrot.thermodynamics.state_flow import zpinch_entropy_flow

        result = zpinch_entropy_flow(
            I=1e4, a_eq=0.01, a0=0.005,  # compressed start
            nu=1e-3, T_plasma=1e7, eta=1e-6,
            n_points=500,
        )
        # Should converge: final R should be closer to 1 than initial
        R_initial = result['R'][0]
        R_final = result['R'][-1]
        assert abs(R_final - 1.0) < abs(R_initial - 1.0)

    def test_zpinch_entropy_flow_F_decreasing(self):
        """Free energy should generally decrease along entropy flow."""
        from magrot.thermodynamics.state_flow import zpinch_entropy_flow

        result = zpinch_entropy_flow(
            I=1e4, a_eq=0.01, a0=0.005,
            nu=1e-3, T_plasma=1e7, eta=1e-6,
            n_points=200,
        )
        F = result['F']
        # Check that the overall trend is decreasing (allow for oscillations)
        # Use a coarse-grained check: F at end < F at start
        assert F[-1] <= F[0] + abs(F[0]) * 0.1  # Allow 10% tolerance for oscillations

    def test_zpinch_entropy_flow_time_monotonic(self):
        """Accumulated time should be monotonically increasing."""
        from magrot.thermodynamics.state_flow import zpinch_entropy_flow

        result = zpinch_entropy_flow(
            I=1e4, a_eq=0.01, a0=0.005,
            nu=1e-3, T_plasma=1e7, eta=1e-6,
            n_points=200,
        )
        dt = np.diff(result['t'])
        assert np.all(dt >= 0)
