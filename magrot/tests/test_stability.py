"""Tests for the stability module (Phase 2B)."""

import numpy as np
import pytest
from scipy.constants import mu_0

from magrot.fields.grid import CylindricalGrid
from magrot.fields.analytic import field_zpinch_bennett
from magrot.stability.hessian import (
    fit_hessian_eigenvalue,
    perturb_pressure_scale,
    perturb_field_strength,
    perturb_boundary_radius,
    perturb_current_profile,
    hessian_sweep_cylindrical,
    multi_axis_hessian_summary,
    HessianResult,
)
from magrot.stability.entropy_audit import entropy_audit_cylindrical
from magrot.stability.manifold import (
    remove_pressure_zpinch,
    remove_field_zpinch,
    constraint_relaxation_sweep_cylindrical,
    mesa_edge_summary,
)
from magrot.stability.attractors import (
    estimate_basin_width,
    classify_attractor,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def zpinch():
    I = 1e4
    a = 0.01
    grid = CylindricalGrid(Nr=200, Ntheta=1, Nz=1, r_range=(0.0003, 0.05))
    B, p = field_zpinch_bennett(grid, I=I, a=a)
    return B, p, grid, I, a


# ── Hessian tests (Test A) ───────────────────────────────────────────────────

class TestHessian:
    def test_fit_hessian_parabola(self):
        """Known parabola should recover correct curvature."""
        eps = np.linspace(-1, 1, 21)
        H_true = 5.0
        F = 3.0 + 0.5 * H_true * eps ** 2
        H, F0, residual = fit_hessian_eigenvalue(eps, F)
        np.testing.assert_allclose(H, H_true, rtol=1e-6)
        np.testing.assert_allclose(F0, 3.0, rtol=1e-6)
        assert residual < 1e-10

    def test_fit_hessian_negative(self):
        """Inverted parabola should give negative H."""
        eps = np.linspace(-1, 1, 21)
        F = 10.0 - 3.0 * eps ** 2
        H, F0, residual = fit_hessian_eigenvalue(eps, F)
        assert H < 0

    def test_perturb_pressure_scale(self, zpinch):
        B, p, grid, _, _ = zpinch
        B_pert, p_pert = perturb_pressure_scale(B, p, grid, 0.5)
        np.testing.assert_allclose(p_pert, p * 1.5, rtol=1e-12)
        np.testing.assert_allclose(B_pert, B, rtol=1e-12)

    def test_perturb_field_strength(self, zpinch):
        B, p, grid, _, _ = zpinch
        B_pert, p_pert = perturb_field_strength(B, p, grid, 0.1)
        np.testing.assert_allclose(B_pert, B * 1.1, rtol=1e-12)

    def test_perturb_boundary_radius(self, zpinch):
        B, p, grid, I, a = zpinch
        B_pert, p_pert = perturb_boundary_radius(B, p, grid, 0.0, a=a, I=I)
        # At epsilon=0, should be identical to original
        np.testing.assert_allclose(B_pert, B, rtol=1e-12)

    def test_hessian_pressure_axis_positive(self, zpinch):
        """Pressure perturbation around equilibrium should give H > 0."""
        B, p, grid, I, a = zpinch
        result = hessian_sweep_cylindrical(
            B, p, grid,
            axis_name='pressure',
            perturbation_func=perturb_pressure_scale,
            n_eps=11, eps_range=0.3,
        )
        assert result.H > 0, f"H = {result.H}, expected > 0"
        assert result.is_stable

    def test_multi_axis_summary(self):
        """Summary logic for stable configuration."""
        r1 = HessianResult('pressure', np.zeros(5), np.zeros(5), 10.0, 1.0, True, 0.0)
        r2 = HessianResult('field', np.zeros(5), np.zeros(5), 5.0, 1.0, True, 0.0)
        summary = multi_axis_hessian_summary([r1, r2])
        assert summary['all_stable']
        assert summary['classification'] == 'minimum'

    def test_multi_axis_summary_saddle(self):
        """Summary logic for saddle point."""
        r1 = HessianResult('pressure', np.zeros(5), np.zeros(5), 10.0, 1.0, True, 0.0)
        r2 = HessianResult('field', np.zeros(5), np.zeros(5), -5.0, 1.0, False, 0.0)
        summary = multi_axis_hessian_summary([r1, r2])
        assert not summary['all_stable']
        assert summary['classification'] == 'saddle'
        assert 'field' in summary['unstable_axes']


# ── Entropy audit tests (Test B) ─────────────────────────────────────────────

class TestEntropyAudit:
    def test_zpinch_entropy_audit_driven(self, zpinch):
        """Z-pinch with current should be a driven state (S_dot > 0)."""
        B, p, grid, _, _ = zpinch
        result, s_dot = entropy_audit_cylindrical(
            B, grid, p, T_core=1e8, T_edge=1e6,
        )
        assert result.is_driven_state
        assert result.S_dot_total > 0
        assert result.P_dissipated > 0
        assert result.fraction_producing > 0

    def test_zpinch_entropy_audit_Jrms_positive(self, zpinch):
        B, p, grid, _, _ = zpinch
        result, _ = entropy_audit_cylindrical(B, grid, p)
        assert result.J_rms > 0


# ── Manifold tests (Test C) ──────────────────────────────────────────────────

class TestManifold:
    def test_remove_pressure_alpha0_is_equilibrium(self, zpinch):
        B, p, grid, _, _ = zpinch
        B_pert, p_pert = remove_pressure_zpinch(B, p, grid, 0.0)
        np.testing.assert_allclose(p_pert, p, rtol=1e-12)

    def test_remove_pressure_alpha1_is_zero(self, zpinch):
        B, p, grid, _, _ = zpinch
        B_pert, p_pert = remove_pressure_zpinch(B, p, grid, 1.0)
        np.testing.assert_allclose(p_pert, 0.0)

    def test_constraint_sweep_runs(self, zpinch):
        B, p, grid, _, _ = zpinch
        result = constraint_relaxation_sweep_cylindrical(
            B, p, grid,
            scenario_name='pressure_removal',
            removal_func=remove_pressure_zpinch,
            n_alpha=7,
        )
        assert len(result.R_core) == 7
        assert len(result.L_values) == 7
        # L should generally increase as constraints are removed
        assert result.L_values[-1] >= result.L_values[0] * 0.5

    def test_mesa_edge_summary(self):
        from magrot.stability.manifold import ConstraintRelaxationResult
        r1 = ConstraintRelaxationResult(
            'pressure', np.linspace(0, 1, 5),
            np.ones(5), np.ones(5), np.ones(5),
            np.array([1.0, 2.0, 5.0, 20.0, 100.0]),
            departure_alpha=0.5, is_catastrophic=True,
        )
        r2 = ConstraintRelaxationResult(
            'field', np.linspace(0, 1, 5),
            np.ones(5), np.ones(5), np.ones(5),
            np.array([1.0, 1.1, 1.2, 1.3, 1.4]),
            departure_alpha=0.9, is_catastrophic=False,
        )
        summary = mesa_edge_summary([r1, r2])
        assert summary['landscape'] == 'structured'


# ── Attractor tests ──────────────────────────────────────────────────────────

class TestAttractors:
    def test_basin_width_estimation(self):
        eps = np.linspace(-1, 1, 21)
        F = 1.0 + 5.0 * eps ** 2
        width = estimate_basin_width(eps, F, threshold_factor=2.0)
        # F(0) = 1, threshold = 2.0
        # F = 2 when 5*eps^2 = 1, eps = 1/sqrt(5) ~ 0.447
        assert 0.3 < width < 0.6

    def test_classify_attractor_valley_on_mesa(self):
        from magrot.stability.hessian import HessianResult
        from magrot.stability.manifold import ConstraintRelaxationResult
        from magrot.stability.entropy_audit import EntropyAuditResult

        hessian_results = [
            HessianResult('pressure', np.zeros(5), np.arange(5, dtype=float),
                          10.0, 1.0, True, 0.0),
            HessianResult('field', np.zeros(5), np.arange(5, dtype=float),
                          5.0, 1.0, True, 0.0),
        ]
        manifold_results = [
            ConstraintRelaxationResult(
                'pressure', np.linspace(0, 1, 5),
                np.ones(5), np.ones(5), np.ones(5),
                np.array([1.0, 2.0, 5.0, 20.0, 100.0]),
                departure_alpha=0.5, is_catastrophic=True,
            ),
        ]
        entropy_result = EntropyAuditResult(
            S_dot_total=1e3, P_dissipated=1e10, T_avg=1e7,
            s_dot_max=1e5, s_dot_min_nonzero=1e-2, J_rms=1e6,
            eta_avg=1e-7, is_driven_state=True, fraction_producing=0.95,
        )

        basin = classify_attractor(hessian_results, manifold_results, entropy_result)
        assert basin.classification == 'valley-on-mesa'
        assert basin.is_driven_state
