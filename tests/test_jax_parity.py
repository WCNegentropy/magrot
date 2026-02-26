"""
JAX ↔ NumPy parity tests for MagRot Forge.

Every test creates identical inputs, runs both the NumPy reference and
JAX implementations, and asserts agreement to 1e-6.  If these fail,
the optimizer would chase numerical artifacts instead of physics.

Run:  python -m pytest tests/test_jax_parity.py -v
"""

import numpy as np
import jax
import jax.numpy as jnp
import pytest

# Enable 64-bit precision in JAX (critical for parity with NumPy float64)
jax.config.update("jax_enable_x64", True)

# NumPy reference
from magrot.numerics import diff_4th
from magrot.fields.grid import CylindricalGrid
from magrot.geometry.curvature import compute_curvature_cylindrical
from magrot.stress.maxwell import compute_forces_conservative_cyl
from magrot.stress.decompose import decompose_inward_outward
from magrot.rotation.metrics import compute_all_metrics
from magrot.thermodynamics.free_energy import (
    free_energy_cylindrical,
    lyapunov_R_cylindrical,
    magnetic_energy_density,
)

# JAX implementations
from magrot.jax_core.numerics_jax import diff_4th_jax
from magrot.jax_core.grid_jax import grid_info_from_numpy
from magrot.jax_core.curvature_jax import compute_curvature_cylindrical_jax
from magrot.jax_core.maxwell_jax import compute_forces_conservative_cyl_jax
from magrot.jax_core.decompose_jax import decompose_inward_outward_jax
from magrot.jax_core.metrics_jax import compute_R_universal_jax
from magrot.jax_core.free_energy_jax import (
    free_energy_cylindrical_jax,
    lyapunov_R_jax,
    magnetic_energy_density_jax,
)


# ── Shared test fixtures ─────────────────────────────────────────────────

ATOL = 1e-6
RTOL = 1e-5


@pytest.fixture
def grid_1d():
    """1D radial grid (axisymmetric, Ntheta=1, Nz=1)."""
    return CylindricalGrid(Nr=60, Ntheta=1, Nz=1, r_range=(0.1, 2.0))


@pytest.fixture
def grid_3d():
    """3D grid with angular and axial resolution."""
    return CylindricalGrid(Nr=30, Ntheta=16, Nz=10,
                           r_range=(0.2, 1.5),
                           z_range=(-0.5, 0.5))


def _make_zpinch_B(grid):
    """Z-pinch: B_theta = mu_0 I / (2 pi r), B_r = B_z = 0."""
    from scipy.constants import mu_0
    I = 1e4
    R = grid.R
    B = np.zeros((*R.shape, 3))
    B[..., 1] = mu_0 * I / (2 * np.pi * R)
    return B


def _make_dipole_B(grid):
    """Simple dipole: Bz = B0 / (1 + r^2), Br = 0, Btheta = 0."""
    R = grid.R
    B = np.zeros((*R.shape, 3))
    B[..., 2] = 1.0 / (1.0 + R ** 2)
    return B


# ═══════════════════════════════════════════════════════════════════════
# Test 1: diff_4th — the foundation
# ═══════════════════════════════════════════════════════════════════════

class TestDiff4th:
    """Parity tests for the 4th-order finite difference stencil."""

    def test_1d_array(self):
        """1D array, non-periodic."""
        x = np.linspace(0, 2 * np.pi, 100)
        arr = np.sin(x)
        dx = x[1] - x[0]

        result_np = diff_4th(arr, dx, axis=0)
        result_jax = np.asarray(diff_4th_jax(jnp.array(arr), dx, axis=0,
                                              periodic=False))

        np.testing.assert_allclose(result_jax, result_np, atol=ATOL, rtol=RTOL)

    def test_3d_radial(self):
        """3D array, radial axis (axis=0), non-periodic."""
        rng = np.random.default_rng(42)
        arr = rng.standard_normal((30, 16, 10))
        dx = 0.05

        result_np = diff_4th(arr, dx, axis=0)
        result_jax = np.asarray(diff_4th_jax(jnp.array(arr), dx, axis=0,
                                              periodic=False))

        np.testing.assert_allclose(result_jax, result_np, atol=ATOL, rtol=RTOL)

    def test_3d_z_axis(self):
        """3D array, z axis (axis=2), non-periodic."""
        rng = np.random.default_rng(123)
        arr = rng.standard_normal((20, 8, 15))
        dx = 0.1

        result_np = diff_4th(arr, dx, axis=2)
        result_jax = np.asarray(diff_4th_jax(jnp.array(arr), dx, axis=2,
                                              periodic=False))

        np.testing.assert_allclose(result_jax, result_np, atol=ATOL, rtol=RTOL)

    def test_periodic_accuracy(self):
        """Periodic mode should give 4th-order accuracy for sin(theta)."""
        # Use enough points that h^4 truncation error is small
        N = 256
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
        arr = np.sin(3 * theta)  # d/dtheta = 3 cos(3 theta)
        dtheta = theta[1] - theta[0]

        result = np.asarray(diff_4th_jax(jnp.array(arr), dtheta, axis=0,
                                          periodic=True))
        expected = 3.0 * np.cos(3 * theta)

        # 4th-order: error ~ h^4 * f^(5) / 30.  With N=256, h~0.0245,
        # f^(5) = 3^5 = 243, so max error ~ 0.0245^4 * 243 / 30 ~ 2.9e-6.
        np.testing.assert_allclose(result, expected, atol=5e-6)

    def test_periodic_3d_theta_axis(self):
        """3D array, periodic along theta (axis=1)."""
        Nr, Ntheta, Nz = 20, 128, 8
        dtheta = 2 * np.pi / Ntheta

        # Verify derivative of known function: d/dtheta sin(theta) = cos(theta)
        theta = np.linspace(0, 2 * np.pi, Ntheta, endpoint=False)
        test_arr = np.sin(theta)[np.newaxis, :, np.newaxis] * np.ones((Nr, 1, Nz))
        result = np.asarray(diff_4th_jax(jnp.array(test_arr), dtheta, axis=1,
                                          periodic=True))
        expected = np.cos(theta)[np.newaxis, :, np.newaxis] * np.ones((Nr, 1, Nz))
        # 4th-order with 128 points: h ~ 0.049, h^4 ~ 5.8e-6
        np.testing.assert_allclose(result, expected, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════
# Test 2: Curvature
# ═══════════════════════════════════════════════════════════════════════

class TestCurvature:

    def test_zpinch_1d(self, grid_1d):
        B = _make_zpinch_B(grid_1d)
        gi = grid_info_from_numpy(grid_1d)

        kv_np, km_np = compute_curvature_cylindrical(B, grid_1d)
        kv_jax, km_jax = compute_curvature_cylindrical_jax(jnp.array(B), gi)

        np.testing.assert_allclose(np.asarray(km_jax), km_np, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(np.asarray(kv_jax), kv_np, atol=ATOL, rtol=RTOL)

    def test_dipole_3d(self, grid_3d):
        B = _make_dipole_B(grid_3d)
        gi = grid_info_from_numpy(grid_3d)

        kv_np, km_np = compute_curvature_cylindrical(B, grid_3d)
        kv_jax, km_jax = compute_curvature_cylindrical_jax(jnp.array(B), gi)

        np.testing.assert_allclose(np.asarray(km_jax), km_np, atol=ATOL, rtol=RTOL)
        np.testing.assert_allclose(np.asarray(kv_jax), kv_np, atol=ATOL, rtol=RTOL)


# ═══════════════════════════════════════════════════════════════════════
# Test 3: Maxwell / Forces
# ═══════════════════════════════════════════════════════════════════════

class TestMaxwell:

    def test_zpinch_1d(self, grid_1d):
        B = _make_zpinch_B(grid_1d)
        gi = grid_info_from_numpy(grid_1d)

        phys_np = compute_forces_conservative_cyl(B, grid_1d)
        phys_jax = compute_forces_conservative_cyl_jax(jnp.array(B), gi)

        for key in ['curlB', 'J', 'f_lorentz', 'Bmag', 'Bmag2']:
            np.testing.assert_allclose(
                np.asarray(phys_jax[key]), phys_np[key],
                atol=ATOL, rtol=RTOL,
                err_msg=f"Mismatch in {key}"
            )

    def test_dipole_3d(self, grid_3d):
        B = _make_dipole_B(grid_3d)
        gi = grid_info_from_numpy(grid_3d)

        phys_np = compute_forces_conservative_cyl(B, grid_3d)
        phys_jax = compute_forces_conservative_cyl_jax(jnp.array(B), gi)

        for key in ['curlB', 'J', 'f_lorentz', 'Bmag', 'Bmag2']:
            np.testing.assert_allclose(
                np.asarray(phys_jax[key]), phys_np[key],
                atol=ATOL, rtol=RTOL,
                err_msg=f"Mismatch in {key} (3D)"
            )


# ═══════════════════════════════════════════════════════════════════════
# Test 4: Decompose
# ═══════════════════════════════════════════════════════════════════════

class TestDecompose:

    def test_basic(self):
        rng = np.random.default_rng(77)
        f_r = rng.standard_normal(100)

        inw_np, outw_np = decompose_inward_outward(f_r)
        inw_jax, outw_jax = decompose_inward_outward_jax(jnp.array(f_r))

        np.testing.assert_allclose(np.asarray(inw_jax), inw_np, atol=1e-12)
        np.testing.assert_allclose(np.asarray(outw_jax), outw_np, atol=1e-12)

    def test_with_pressure(self):
        rng = np.random.default_rng(88)
        f_r = rng.standard_normal(50)
        dp = rng.standard_normal(50)

        inw_np, outw_np = decompose_inward_outward(f_r, dp)
        inw_jax, outw_jax = decompose_inward_outward_jax(jnp.array(f_r),
                                                           jnp.array(dp))

        np.testing.assert_allclose(np.asarray(inw_jax), inw_np, atol=1e-12)
        np.testing.assert_allclose(np.asarray(outw_jax), outw_np, atol=1e-12)


# ═══════════════════════════════════════════════════════════════════════
# Test 5: R_universal
# ═══════════════════════════════════════════════════════════════════════

class TestRUniversal:

    def test_zpinch_1d(self, grid_1d):
        B = _make_zpinch_B(grid_1d)
        gi = grid_info_from_numpy(grid_1d)

        metrics_np = compute_all_metrics(B, grid_1d)
        R_np = metrics_np['R_universal']

        R_jax, _ = compute_R_universal_jax(jnp.array(B), gi)

        np.testing.assert_allclose(np.asarray(R_jax), R_np,
                                   atol=ATOL, rtol=RTOL)

    def test_dipole_3d(self, grid_3d):
        B = _make_dipole_B(grid_3d)
        gi = grid_info_from_numpy(grid_3d)

        metrics_np = compute_all_metrics(B, grid_3d)
        R_np = metrics_np['R_universal']

        R_jax, _ = compute_R_universal_jax(jnp.array(B), gi)

        np.testing.assert_allclose(np.asarray(R_jax), R_np,
                                   atol=ATOL, rtol=RTOL)


# ═══════════════════════════════════════════════════════════════════════
# Test 6: Free energy and Lyapunov
# ═══════════════════════════════════════════════════════════════════════

class TestFreeEnergy:

    def test_energy_density(self):
        rng = np.random.default_rng(55)
        B = rng.standard_normal((20, 1, 1, 3))

        u_np = magnetic_energy_density(B)
        u_jax = np.asarray(magnetic_energy_density_jax(jnp.array(B)))

        np.testing.assert_allclose(u_jax, u_np, atol=1e-12)

    def test_free_energy_1d(self, grid_1d):
        B = _make_zpinch_B(grid_1d)
        gi = grid_info_from_numpy(grid_1d)

        F_np, u_np = free_energy_cylindrical(B, grid_1d)
        F_jax, u_jax = free_energy_cylindrical_jax(jnp.array(B), gi)

        np.testing.assert_allclose(float(F_jax), F_np, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(u_jax), u_np, atol=1e-12)

    def test_free_energy_3d(self, grid_3d):
        B = _make_dipole_B(grid_3d)
        gi = grid_info_from_numpy(grid_3d)

        F_np, u_np = free_energy_cylindrical(B, grid_3d)
        F_jax, u_jax = free_energy_cylindrical_jax(jnp.array(B), gi)

        np.testing.assert_allclose(float(F_jax), F_np, rtol=1e-6)

    def test_lyapunov_1d(self, grid_1d):
        B = _make_zpinch_B(grid_1d)
        gi = grid_info_from_numpy(grid_1d)

        metrics = compute_all_metrics(B, grid_1d)
        R_np = metrics['R_universal']

        L_np, dev_np = lyapunov_R_cylindrical(R_np, grid_1d)
        L_jax, dev_jax = lyapunov_R_jax(jnp.array(R_np), gi)

        np.testing.assert_allclose(float(L_jax), L_np, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(dev_jax), dev_np, atol=1e-12)

    def test_lyapunov_3d(self, grid_3d):
        B = _make_dipole_B(grid_3d)
        gi = grid_info_from_numpy(grid_3d)

        metrics = compute_all_metrics(B, grid_3d)
        R_np = metrics['R_universal']

        L_np, dev_np = lyapunov_R_cylindrical(R_np, grid_3d)
        L_jax, dev_jax = lyapunov_R_jax(jnp.array(R_np), gi)

        np.testing.assert_allclose(float(L_jax), L_np, rtol=1e-6)


# ═══════════════════════════════════════════════════════════════════════
# Test 7: Biot-Savart sanity checks
# ═══════════════════════════════════════════════════════════════════════

class TestBiotSavart:

    def test_circular_coil_on_axis(self):
        """B on axis of a circular coil should match analytic formula."""
        from magrot.jax_core.biot_savart import (
            fourier_to_coil_points,
            _biot_savart_single_coil,
        )
        from scipy.constants import mu_0

        R_coil = 1.0  # Coil radius
        I = 1.0       # Current

        # Fourier params for a circle of radius R in the xy-plane
        params = jnp.zeros((2, 3, 2))  # 2 harmonics, 3 coords, cos/sin
        params = params.at[1, 0, 0].set(R_coil)  # x = R cos(t)
        params = params.at[1, 1, 1].set(R_coil)  # y = R sin(t)

        coil_pts = fourier_to_coil_points(params, N_segments=256)

        # Evaluate on axis at z=0
        eval_pts = jnp.array([[0.0, 0.0, 0.0]])
        B = _biot_savart_single_coil(coil_pts, eval_pts, current=I,
                                      softening=1e-14)

        # Analytic: B_z = mu_0 I / (2 R)
        B_analytic = mu_0 * I / (2 * R_coil)
        B_z = float(B[0, 2])

        np.testing.assert_allclose(B_z, B_analytic, rtol=0.01,
                                   err_msg="On-axis field doesn't match analytic")

    def test_field_is_differentiable(self):
        """Verify that jax.grad flows through the Biot-Savart chain."""
        from magrot.jax_core.biot_savart import (
            fourier_to_coil_points,
            _biot_savart_single_coil,
        )

        def loss_fn(params):
            coil_pts = fourier_to_coil_points(params, N_segments=64)
            eval_pts = jnp.array([[0.0, 0.0, 0.0]])
            B = _biot_savart_single_coil(coil_pts, eval_pts)
            return jnp.sum(B ** 2)

        params = jnp.zeros((3, 3, 2))
        params = params.at[1, 0, 0].set(1.0)
        params = params.at[1, 1, 1].set(1.0)

        grads = jax.grad(loss_fn)(params)
        assert grads.shape == params.shape
        assert jnp.any(grads != 0.0), "Gradients should be non-zero"


# ═══════════════════════════════════════════════════════════════════════
# Test 8: End-to-end gradient flow
# ═══════════════════════════════════════════════════════════════════════

class TestGradientFlow:

    def test_forge_loss_gradient(self):
        """Verify jax.grad flows through the complete forge_loss chain."""
        from magrot.jax_core.forge_loss import forge_loss_scalar
        from magrot.jax_core.grid_jax import make_grid_info
        from magrot.jax_core.biot_savart import (
            init_circular_coils,
            grid_to_eval_points,
        )

        # Very small grid for fast CPU test
        gi = make_grid_info(Nr=10, Ntheta=8, Nz=5,
                            r_range=(0.3, 1.2),
                            z_range=(-0.3, 0.3))
        eval_cart = grid_to_eval_points(gi)

        coil_params = init_circular_coils(
            N_coils=3, major_radius=0.7, minor_radius=0.4,
            N_fourier=5
        )

        # Compute loss
        loss_val = forge_loss_scalar(coil_params, gi, eval_cart,
                                      N_segments=32)
        assert jnp.isfinite(loss_val), f"Loss is not finite: {loss_val}"
        assert float(loss_val) > 0, "Loss should be positive"

        # Compute gradient
        grad_fn = jax.grad(forge_loss_scalar)
        grads = grad_fn(coil_params, gi, eval_cart, N_segments=32)

        assert grads.shape == coil_params.shape
        assert jnp.all(jnp.isfinite(grads)), "Gradients contain NaN/inf"
        assert jnp.any(grads != 0.0), "Gradients should be non-zero"
