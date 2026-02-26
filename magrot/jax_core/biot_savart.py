"""
Differentiable Biot-Savart integrator for MagRot Forge.

Computes the magnetic field B(r) produced by a set of current-carrying
coils parameterized as Fourier-described closed curves.  The entire
chain  coil_params -> coil_points -> B_field  is differentiable via
jax.grad, enabling gradient-based coil geometry optimization.

Coil parameterization
---------------------
Each coil is a closed 3D curve given by Fourier descriptors:

    x(t) = sum_n [ xc_n cos(n t) + xs_n sin(n t) ]
    y(t) = sum_n [ yc_n cos(n t) + ys_n sin(n t) ]
    z(t) = sum_n [ zc_n cos(n t) + zs_n sin(n t) ]

with t in [0, 2pi).  The array shape per coil is (N_fourier, 3, 2)
where the last axis is [cos, sin] coefficients.
"""

import jax
import jax.numpy as jnp
from functools import partial
from scipy.constants import mu_0
import numpy as np


# ── Fourier -> coil points ───────────────────────────────────────────────

def fourier_to_coil_points(coil_params, N_segments=128):
    """Convert Fourier coefficients to discretized coil centerline.

    Parameters
    ----------
    coil_params : jnp.ndarray, shape (N_fourier, 3, 2)
        Fourier coefficients for one coil.
        coil_params[n, coord, 0] = cos coefficient
        coil_params[n, coord, 1] = sin coefficient
        coord: 0=x, 1=y, 2=z; n: harmonic index (0..N_fourier-1)
    N_segments : int
        Number of discrete points along the coil.

    Returns
    -------
    points : jnp.ndarray, shape (N_segments, 3)
        Coil centerline in Cartesian (x, y, z).
    """
    t = jnp.linspace(0, 2 * jnp.pi, N_segments, endpoint=False)
    N_fourier = coil_params.shape[0]
    n = jnp.arange(N_fourier)  # (N_fourier,)

    # (N_segments, N_fourier)
    cos_nt = jnp.cos(n[jnp.newaxis, :] * t[:, jnp.newaxis])
    sin_nt = jnp.sin(n[jnp.newaxis, :] * t[:, jnp.newaxis])

    # coil_params[:, :, 0] = cos coeffs, shape (N_fourier, 3)
    # coil_params[:, :, 1] = sin coeffs, shape (N_fourier, 3)
    # points = cos_nt @ cos_coeffs + sin_nt @ sin_coeffs
    points = cos_nt @ coil_params[:, :, 0] + sin_nt @ coil_params[:, :, 1]

    return points  # (N_segments, 3)


def fourier_to_all_coils(all_coil_params, N_segments=128):
    """Convert a batch of coil Fourier params to coil point arrays.

    Parameters
    ----------
    all_coil_params : jnp.ndarray, shape (N_coils, N_fourier, 3, 2)
    N_segments : int

    Returns
    -------
    all_points : jnp.ndarray, shape (N_coils, N_segments, 3)
    """
    return jax.vmap(fourier_to_coil_points, in_axes=(0, None))(
        all_coil_params, N_segments
    )


# ── Biot-Savart field computation ────────────────────────────────────────

def _biot_savart_single_coil(coil_points, eval_points, current=1.0,
                              softening=1e-7):
    """B-field from one coil at all evaluation points.

    Parameters
    ----------
    coil_points : jnp.ndarray, shape (N_seg, 3)
        Discretized coil centerline.
    eval_points : jnp.ndarray, shape (M, 3)
        Points where B is evaluated.
    current : float
        Coil current (A).
    softening : float
        Softening length^2 to prevent singularities.

    Returns
    -------
    B : jnp.ndarray, shape (M, 3)
    """
    # dl = coil_points[i+1] - coil_points[i], wrapping at end
    dl = jnp.roll(coil_points, -1, axis=0) - coil_points  # (N_seg, 3)
    # Midpoint of each segment
    midpoints = (coil_points + jnp.roll(coil_points, -1, axis=0)) / 2.0

    # r_vec = eval_points - midpoints: (M, N_seg, 3)
    r_vec = eval_points[:, jnp.newaxis, :] - midpoints[jnp.newaxis, :, :]
    r_mag2 = jnp.sum(r_vec ** 2, axis=-1) + softening  # (M, N_seg)
    r_mag3 = r_mag2 * jnp.sqrt(r_mag2)  # |r|^3

    # dl x r_vec: (M, N_seg, 3)
    cross = jnp.cross(dl[jnp.newaxis, :, :], r_vec)

    # Sum over segments: B = (mu_0 I / 4pi) sum (dl x r_hat / |r|^2)
    #                      = (mu_0 I / 4pi) sum (dl x r / |r|^3)
    prefactor = mu_0 * current / (4.0 * jnp.pi)
    B = prefactor * jnp.sum(cross / r_mag3[..., jnp.newaxis], axis=1)

    return B  # (M, 3)


def biot_savart_field(all_coil_points, eval_points, currents=None,
                      softening=1e-7):
    """Total B-field from all coils at all evaluation points.

    Parameters
    ----------
    all_coil_points : jnp.ndarray, shape (N_coils, N_seg, 3)
    eval_points : jnp.ndarray, shape (M, 3)
    currents : jnp.ndarray, shape (N_coils,) or None
        Per-coil currents. Default: 1.0 A for all coils.
    softening : float

    Returns
    -------
    B_total : jnp.ndarray, shape (M, 3)
    """
    N_coils = all_coil_points.shape[0]
    if currents is None:
        currents = jnp.ones(N_coils)

    def single_coil_field(coil_pts, current):
        return _biot_savart_single_coil(coil_pts, eval_points, current,
                                         softening)

    # vmap over coils, sum contributions
    all_B = jax.vmap(single_coil_field)(all_coil_points, currents)  # (N_coils, M, 3)
    return jnp.sum(all_B, axis=0)  # (M, 3)


# ── Coordinate transform: Cartesian -> Cylindrical ──────────────────────

def cartesian_to_cylindrical_B(B_cart, eval_points_cart):
    """Transform B from Cartesian (Bx, By, Bz) to cylindrical (Br, Btheta, Bz).

    Parameters
    ----------
    B_cart : jnp.ndarray, shape (M, 3)
        B field in Cartesian components.
    eval_points_cart : jnp.ndarray, shape (M, 3)
        Evaluation points in Cartesian (x, y, z).

    Returns
    -------
    B_cyl : jnp.ndarray, shape (M, 3)
        B field in cylindrical (Br, Btheta, Bz).
    """
    x = eval_points_cart[:, 0]
    y = eval_points_cart[:, 1]
    phi = jnp.arctan2(y, x)

    cos_phi = jnp.cos(phi)
    sin_phi = jnp.sin(phi)

    Bx = B_cart[:, 0]
    By = B_cart[:, 1]
    Bz = B_cart[:, 2]

    Br = Bx * cos_phi + By * sin_phi
    Btheta = -Bx * sin_phi + By * cos_phi

    return jnp.stack([Br, Btheta, Bz], axis=-1)


# ── Grid point generation (cylindrical -> Cartesian eval points) ─────────

def grid_to_eval_points(grid_info):
    """Convert cylindrical grid to flat Cartesian eval points for Biot-Savart.

    Parameters
    ----------
    grid_info : dict

    Returns
    -------
    eval_cart : jnp.ndarray, shape (Nr*Ntheta*Nz, 3)
        Evaluation points in (x, y, z).
    """
    R = grid_info['R']          # (Nr, Ntheta, Nz)
    THETA = grid_info['THETA']
    Z = grid_info['Z']

    X = R * jnp.cos(THETA)
    Y = R * jnp.sin(THETA)

    return jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)


# ── Coil initialization with physical priors ─────────────────────────────

def init_circular_coils(N_coils, major_radius=1.0, minor_radius=0.3,
                        N_fourier=10, key=None):
    """Initialize coil params as circular coils with optional perturbation.

    Each coil is a circle at a different toroidal angle, tilted to
    approximate a simple stellarator-like geometry.

    Parameters
    ----------
    N_coils : int
    major_radius : float
        Major radius of the torus.
    minor_radius : float
        Distance from the magnetic axis to each coil center.
    N_fourier : int
        Number of Fourier harmonics.
    key : jax.random.PRNGKey or None
        If given, add small random perturbations to the coils.

    Returns
    -------
    coil_params : jnp.ndarray, shape (N_coils, N_fourier, 3, 2)
    """
    params = jnp.zeros((N_coils, N_fourier, 3, 2))

    for i in range(N_coils):
        phi = 2 * np.pi * i / N_coils  # Toroidal angle of this coil

        # Mode 1 cos = circle center offset in x,y
        # The coil center is at (R cos(phi), R sin(phi), 0) on the torus
        # Mode 1 gives the circle shape: cos(t) and sin(t)
        # x(t) = R cos(phi) + a cos(t) cos(phi)  (radial direction)
        # y(t) = R sin(phi) + a cos(t) sin(phi)
        # z(t) = a sin(t)

        # Mode 0 (constant offset = coil center)
        params = params.at[i, 0, 0, 0].set(major_radius * np.cos(phi))  # x center
        params = params.at[i, 0, 1, 0].set(major_radius * np.sin(phi))  # y center

        # Mode 1 (circle shape)
        params = params.at[i, 1, 0, 0].set(minor_radius * np.cos(phi))  # x cos
        params = params.at[i, 1, 1, 0].set(minor_radius * np.sin(phi))  # y cos
        params = params.at[i, 1, 2, 1].set(minor_radius)                # z sin

    if key is not None:
        noise = jax.random.normal(key, params.shape) * 0.01 * minor_radius
        # Don't perturb mode 0 and mode 1 too much
        noise = noise.at[:, :2, :, :].multiply(0.1)
        params = params + noise

    return params
