"""
MagRot Forge loss function — physics + engineering constraints.

The total loss combines:
  L_R         : Lyapunov integral |R-1|^2 dV  (force balance)
  F_total     : Stored magnetic energy         (prefer lower energy)
  R_grad_mag  : Spatial smoothness of R        (no sharp instability)
  L_curvature : Coil bend radius penalty       (buildability)
  L_distance  : Min coil-coil gap penalty      (buildability)
  L_length    : Total coil wire length         (simplicity)

All terms are differentiable via jax.grad through the complete chain:
  coil_params -> Fourier -> Biot-Savart -> B_cyl -> R_universal -> loss
"""

import jax
import jax.numpy as jnp
from functools import partial

from .biot_savart import (
    fourier_to_all_coils,
    biot_savart_field,
    cartesian_to_cylindrical_B,
    grid_to_eval_points,
)
from .metrics_jax import compute_R_universal_jax
from .free_energy_jax import (
    lyapunov_R_jax,
    free_energy_cylindrical_jax,
)
from .numerics_jax import diff_4th_jax


# ── Engineering penalty functions ────────────────────────────────────────

def coil_curvature_penalty(all_coil_points):
    """Penalize coils with tight bends (small radius of curvature).

    Uses the discrete curvature: kappa = |d²r/dt²| / |dr/dt|² at each
    segment.  Penalizes max curvature exceeding a threshold.

    Parameters
    ----------
    all_coil_points : jnp.ndarray, shape (N_coils, N_seg, 3)

    Returns
    -------
    penalty : scalar
    """
    # First derivative (tangent)
    dr = jnp.roll(all_coil_points, -1, axis=1) - all_coil_points
    # Second derivative
    d2r = jnp.roll(dr, -1, axis=1) - dr

    dr_mag = jnp.sqrt(jnp.sum(dr ** 2, axis=-1) + 1e-12)
    d2r_mag = jnp.sqrt(jnp.sum(d2r ** 2, axis=-1) + 1e-12)

    # Curvature kappa = |d2r| / |dr|^2
    kappa = d2r_mag / (dr_mag ** 2 + 1e-12)

    # Penalize mean of kappa^2 (smooth, differentiable)
    return jnp.mean(kappa ** 2)


def coil_distance_penalty(all_coil_points, d_min=0.05):
    """Penalize coils that get too close to each other.

    For each pair of coils, compute the minimum distance between their
    segment midpoints and penalize if below d_min.

    Parameters
    ----------
    all_coil_points : jnp.ndarray, shape (N_coils, N_seg, 3)
    d_min : float
        Minimum allowed coil-coil distance (meters).

    Returns
    -------
    penalty : scalar
    """
    N_coils = all_coil_points.shape[0]
    penalty = 0.0

    for i in range(N_coils):
        for j in range(i + 1, N_coils):
            # Pairwise distances between segment midpoints
            # Use a subsampled version for efficiency (every 4th point)
            pts_i = all_coil_points[i, ::4, :]  # (N_seg//4, 3)
            pts_j = all_coil_points[j, ::4, :]

            # (Ni, Nj)
            diff = pts_i[:, jnp.newaxis, :] - pts_j[jnp.newaxis, :, :]
            dist2 = jnp.sum(diff ** 2, axis=-1)
            min_dist2 = jnp.min(dist2)

            # Smooth penalty: max(0, d_min^2 - min_dist^2)^2
            violation = jnp.maximum(d_min ** 2 - min_dist2, 0.0)
            penalty = penalty + violation ** 2

    return penalty


def coil_length_penalty(all_coil_points):
    """Penalize total coil wire length (prefer simpler coil sets).

    Parameters
    ----------
    all_coil_points : jnp.ndarray, shape (N_coils, N_seg, 3)

    Returns
    -------
    total_length : scalar
    """
    dl = jnp.roll(all_coil_points, -1, axis=1) - all_coil_points
    seg_lengths = jnp.sqrt(jnp.sum(dl ** 2, axis=-1))
    return jnp.sum(seg_lengths)


# ── R-field spatial gradient penalty ─────────────────────────────────────

def r_gradient_penalty(R_field, grid_info):
    """Penalize sharp spatial gradients in R (instability sites).

    Parameters
    ----------
    R_field : jnp.ndarray, shape (Nr, Ntheta, Nz)
    grid_info : dict

    Returns
    -------
    penalty : scalar (sum of |grad R|^2)
    """
    dR_dr = diff_4th_jax(R_field, grid_info['dr'], axis=0, periodic=False)
    grad_sq = dR_dr ** 2

    if grid_info['Ntheta'] > 1:
        dR_dtheta = diff_4th_jax(R_field, grid_info['dtheta'], axis=1,
                                  periodic=True)
        grad_sq = grad_sq + (dR_dtheta / grid_info['R']) ** 2

    if grid_info['Nz'] > 1:
        dR_dz = diff_4th_jax(R_field, grid_info['dz'], axis=2, periodic=False)
        grad_sq = grad_sq + dR_dz ** 2

    return jnp.sum(grad_sq)


# ── Complete Forge loss ──────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    'w_R': 1.0,         # Lyapunov |R-1|^2 — primary physics objective
    'w_F': 0.01,        # Stored energy — prefer lower energy
    'w_grad': 0.1,      # R smoothness — no sharp instability sites
    'w_curv': 0.1,      # Coil curvature — buildability
    'w_dist': 1.0,      # Coil distance — no overlapping coils
    'w_len': 0.001,     # Coil length — simplicity
}


def forge_loss(coil_params, grid_info, eval_cart, weights=None,
               N_segments=128, currents=None):
    """Total Forge loss: physics + engineering.

    Parameters
    ----------
    coil_params : jnp.ndarray, shape (N_coils, N_fourier, 3, 2)
        Fourier coefficients for all coils.
    grid_info : dict
        Static grid arrays.
    eval_cart : jnp.ndarray, shape (M, 3)
        Cartesian evaluation points (pre-computed from grid).
    weights : dict or None
        Loss term weights. If None, use DEFAULT_WEIGHTS.
    N_segments : int
        Coil discretization resolution.
    currents : jnp.ndarray or None
        Per-coil currents.

    Returns
    -------
    loss : scalar
    aux : dict
        Individual loss components for monitoring.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # 1. Coil params -> physical coil curves
    coil_points = fourier_to_all_coils(coil_params, N_segments)

    # 2. Biot-Savart: coils -> B field at evaluation points (Cartesian)
    B_cart_flat = biot_savart_field(coil_points, eval_cart, currents)

    # 3. Cartesian -> Cylindrical B field
    B_cyl_flat = cartesian_to_cylindrical_B(B_cart_flat, eval_cart)

    # 4. Reshape to grid shape (Nr, Ntheta, Nz, 3)
    Nr, Ntheta, Nz = grid_info['Nr'], grid_info['Ntheta'], grid_info['Nz']
    B_cyl = B_cyl_flat.reshape(Nr, Ntheta, Nz, 3)

    # 5. R_universal
    R_universal, phys = compute_R_universal_jax(B_cyl, grid_info)

    # 6. Physics loss terms
    L_R, _ = lyapunov_R_jax(R_universal, grid_info)
    F_total, _ = free_energy_cylindrical_jax(B_cyl, grid_info)
    R_grad = r_gradient_penalty(R_universal, grid_info)

    # 7. Engineering constraints
    L_curv = coil_curvature_penalty(coil_points)
    L_dist = coil_distance_penalty(coil_points)
    L_len = coil_length_penalty(coil_points)

    # 8. Weighted total
    loss = (weights['w_R'] * L_R
            + weights['w_F'] * F_total
            + weights['w_grad'] * R_grad
            + weights['w_curv'] * L_curv
            + weights['w_dist'] * L_dist
            + weights['w_len'] * L_len)

    aux = {
        'L_R': L_R,
        'F_total': F_total,
        'R_grad': R_grad,
        'L_curv': L_curv,
        'L_dist': L_dist,
        'L_len': L_len,
        'loss': loss,
        'R_mean': jnp.mean(R_universal),
        'R_std': jnp.std(R_universal),
    }

    return loss, aux


def forge_loss_scalar(coil_params, grid_info, eval_cart, weights=None,
                      N_segments=128, currents=None):
    """Scalar-only version for jax.grad (drops aux)."""
    loss, _ = forge_loss(coil_params, grid_info, eval_cart, weights,
                         N_segments, currents)
    return loss
