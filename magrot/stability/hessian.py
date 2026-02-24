"""
Multi-axis perturbation response and Hessian eigenvalue computation (Test A).

Goal: Determine if R = 1 is a local minimum, maximum, or saddle point of
the free energy functional F.

Method:
    1. Start from equilibrium (R_core ~ 1)
    2. Define perturbation axes (pressure, current, shape, field, ...)
    3. Sweep perturbation amplitude epsilon from -delta to +delta
    4. Compute F(epsilon) = integral |R - 1|^2 dV at each level
    5. Fit parabola: F(epsilon) ~ F_0 + 0.5 * H * epsilon^2
    6. Extract Hessian eigenvalue H for each axis

Success criteria:
    H > 0 for all axes -> R = 1 is a true minimum (stable attractor)
    H > 0 some, H < 0 others -> saddle point (MHD instability modes)
    H < 0 for all -> maximum (unstable, contradicts v1 evidence)
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class HessianResult:
    """Result of a single-axis perturbation sweep."""
    axis_name: str
    epsilon_values: np.ndarray
    F_values: np.ndarray
    H: float                  # Hessian eigenvalue (curvature)
    F0: float                 # Free energy at epsilon = 0
    is_stable: bool           # H > 0
    fit_residual: float       # Quality of parabolic fit


def fit_hessian_eigenvalue(epsilon_values, F_values):
    """Fit F(epsilon) = F0 + 0.5 * H * epsilon^2 and extract H.

    Parameters
    ----------
    epsilon_values : ndarray
        Perturbation amplitudes.
    F_values : ndarray
        Lyapunov functional values at each epsilon.

    Returns
    -------
    H : float
        Hessian eigenvalue (curvature of F parabola).
    F0 : float
        Free energy at epsilon = 0.
    residual : float
        RMS residual of the parabolic fit.
    """
    eps = np.asarray(epsilon_values, dtype=float)
    F = np.asarray(F_values, dtype=float)

    # Fit: F = c0 + c1 * eps + c2 * eps^2
    # Using least squares: [1, eps, eps^2] . [c0, c1, c2] = F
    A = np.column_stack([np.ones_like(eps), eps, eps ** 2])
    coeffs, residuals, _, _ = np.linalg.lstsq(A, F, rcond=None)
    c0, c1, c2 = coeffs

    # H = 2 * c2 (since F ~ F0 + 0.5 * H * eps^2, so c2 = H/2)
    H = 2.0 * c2
    F0 = c0

    F_fit = A @ coeffs
    rms_residual = float(np.sqrt(np.mean((F - F_fit) ** 2)))

    return H, F0, rms_residual


# ── Perturbation generators for cylindrical Z-pinch ─────────────────────────

def perturb_pressure_scale(B, p, grid, epsilon):
    """Axis 1: Scale plasma pressure by (1 + epsilon).

    Parameters
    ----------
    B : ndarray
    p : ndarray
    grid : CylindricalGrid
    epsilon : float

    Returns
    -------
    B_pert, p_pert : ndarray
    """
    return B.copy(), p * (1.0 + epsilon)


def perturb_current_profile(B, p, grid, epsilon, a=0.01, I=1e4):
    """Axis 2: Perturb current density profile from uniform toward peaked/flat.

    epsilon > 0: more peaked (parabolic weighting)
    epsilon < 0: flatter (inverse parabolic)

    Parameters
    ----------
    B : ndarray
    p : ndarray
    grid : CylindricalGrid
    epsilon : float
    a : float
        Plasma radius.
    I : float
        Total current.

    Returns
    -------
    B_pert, p_pert : ndarray
    """
    from scipy.constants import mu_0

    B_pert = B.copy()
    r = grid.R

    inside = r < a
    r_norm = r / a

    # Modify the current profile weighting
    # Base: uniform J -> B_theta ~ r inside
    # Perturbed: J(r) = J0 * (1 + epsilon * (1 - (r/a)^2))
    # This changes the integral to keep total I constant
    weight = 1.0 + epsilon * (1.0 - r_norm ** 2)
    # Normalize to preserve total current
    # integral J * 2 pi r dr = I from 0 to a
    # For uniform: J0 = I / (pi a^2)
    # For weighted: integral weight * r dr / integral r dr should = 1
    # weight_avg over [0,a] with r weighting:
    # integral_0^a weight(r) * r dr = integral_0^a (1 + eps(1-r^2/a^2)) r dr
    #   = a^2/2 + eps * (a^2/2 - a^2/4) = a^2/2 * (1 + eps/2)
    norm_factor = 1.0 / (1.0 + epsilon / 2.0) if abs(1.0 + epsilon / 2.0) > 1e-10 else 1.0

    # Reconstruct B_theta from modified current profile
    # B_theta(r) = (mu_0 / (2 pi r)) * integral_0^r J(r') * 2 pi r' dr'
    # For J(r') = J0 * norm * (1 + eps(1 - r'^2/a^2)):
    # integral = J0 * norm * pi * [r^2 + eps(r^2 - r^4/(2a^2))]
    # = I * norm * [r^2/a^2 + eps * r^2/a^2 * (1 - r^2/(2a^2))]
    # B_theta = mu_0 I / (2 pi a^2) * norm * r * [1 + eps(1 - r^2/(2a^2))]
    J0_factor = mu_0 * I / (2 * np.pi * a ** 2) * norm_factor
    B_theta_inside = J0_factor * r * (1.0 + epsilon * (1.0 - r_norm ** 2 / 2.0))
    B_theta_outside = mu_0 * I / (2 * np.pi * r)

    B_pert[..., 1] = np.where(inside, B_theta_inside, B_theta_outside)

    return B_pert, p.copy() if p is not None else None


def perturb_boundary_radius(B, p, grid, epsilon, a=0.01, I=1e4):
    """Axis 3: Perturb the plasma boundary radius by factor (1 + epsilon).

    Parameters
    ----------
    B : ndarray
    p : ndarray
    grid : CylindricalGrid
    epsilon : float
    a : float
    I : float

    Returns
    -------
    B_pert, p_pert : ndarray
    """
    from ..fields.analytic import field_zpinch_bennett
    a_pert = a * (1.0 + epsilon)
    return field_zpinch_bennett(grid, I=I, a=a_pert)


def perturb_field_strength(B, p, grid, epsilon):
    """Axis 4: Scale magnetic field magnitude by (1 + epsilon).

    Parameters
    ----------
    B : ndarray
    p : ndarray
    grid : CylindricalGrid
    epsilon : float

    Returns
    -------
    B_pert, p_pert : ndarray
    """
    return B * (1.0 + epsilon), p.copy() if p is not None else None


# ── Perturbation generators for toroidal equilibrium ─────────────────────────

def perturb_tokamak_pressure(eq, epsilon):
    """Axis 1 (toroidal): Scale tokamak pressure by (1 + epsilon).

    Parameters
    ----------
    eq : TokamakEquilibrium-like object
    epsilon : float

    Returns
    -------
    p_pert : ndarray
    """
    return eq.p_mat * (1.0 + epsilon)


def perturb_tokamak_toroidal_field(eq, epsilon):
    """Axis 4 (toroidal): Scale toroidal field by (1 + epsilon).

    Parameters
    ----------
    eq : TokamakEquilibrium-like object
    epsilon : float

    Returns
    -------
    Bphi_pert : ndarray
    """
    return eq.Bphi * (1.0 + epsilon)


# ── Hessian sweep driver ────────────────────────────────────────────────────

def hessian_sweep_cylindrical(B, p, grid, axis_name, perturbation_func,
                              n_eps=21, eps_range=0.3,
                              compute_R_func=None, **perturbation_kwargs):
    """Run a single-axis perturbation sweep and compute the Hessian eigenvalue.

    Parameters
    ----------
    B : ndarray
        Equilibrium magnetic field.
    p : ndarray
        Equilibrium pressure.
    grid : CylindricalGrid
    axis_name : str
        Human-readable name for this perturbation axis.
    perturbation_func : callable
        Function(B, p, grid, epsilon, **kwargs) -> (B_pert, p_pert).
    n_eps : int
        Number of epsilon values to sweep.
    eps_range : float
        Range of epsilon: sweep from -eps_range to +eps_range.
    compute_R_func : callable or None
        Function(B, grid, p) -> R_field.  If None, uses R_universal.
    **perturbation_kwargs
        Extra keyword arguments passed to perturbation_func.

    Returns
    -------
    HessianResult
    """
    from ..rotation.metrics import compute_all_metrics
    from ..thermodynamics.free_energy import lyapunov_R_cylindrical

    epsilon_values = np.linspace(-eps_range, eps_range, n_eps)
    F_values = np.zeros(n_eps)

    for i, eps in enumerate(epsilon_values):
        B_pert, p_pert = perturbation_func(B, p, grid, eps, **perturbation_kwargs)

        if compute_R_func is not None:
            R_field = compute_R_func(B_pert, grid, p_pert)
        else:
            res = compute_all_metrics(B_pert, grid, p_mat=p_pert)
            R_field = res['R_universal']

        L, _ = lyapunov_R_cylindrical(R_field, grid)
        F_values[i] = L

    H, F0, residual = fit_hessian_eigenvalue(epsilon_values, F_values)

    return HessianResult(
        axis_name=axis_name,
        epsilon_values=epsilon_values,
        F_values=F_values,
        H=H,
        F0=F0,
        is_stable=H > 0,
        fit_residual=residual,
    )


def hessian_sweep_toroidal(eq, axis_name, perturbation_func,
                           compute_R_func, compute_L_func,
                           n_eps=21, eps_range=0.3, **kwargs):
    """Run a single-axis perturbation sweep on a tokamak equilibrium.

    Parameters
    ----------
    eq : TokamakEquilibrium-like object
    axis_name : str
    perturbation_func : callable
        Function(eq, epsilon) -> perturbed field/pressure arrays.
    compute_R_func : callable
        Function(eq, perturbed_data) -> R_field.
    compute_L_func : callable
        Function(R_field, eq) -> L (Lyapunov scalar).
    n_eps : int
    eps_range : float

    Returns
    -------
    HessianResult
    """
    epsilon_values = np.linspace(-eps_range, eps_range, n_eps)
    F_values = np.zeros(n_eps)

    for i, eps in enumerate(epsilon_values):
        pert_data = perturbation_func(eq, eps)
        R_field = compute_R_func(eq, pert_data)
        L = compute_L_func(R_field, eq)
        F_values[i] = L

    H, F0, residual = fit_hessian_eigenvalue(epsilon_values, F_values)

    return HessianResult(
        axis_name=axis_name,
        epsilon_values=epsilon_values,
        F_values=F_values,
        H=H,
        F0=F0,
        is_stable=H > 0,
        fit_residual=residual,
    )


def multi_axis_hessian_summary(results):
    """Summarize Hessian eigenvalues from multiple perturbation axes.

    Parameters
    ----------
    results : list of HessianResult

    Returns
    -------
    summary : dict
        Contains eigenvalue spectrum, stability classification, and
        unstable direction list.
    """
    eigenvalues = {r.axis_name: r.H for r in results}
    all_stable = all(r.is_stable for r in results)
    unstable_axes = [r.axis_name for r in results if not r.is_stable]

    if all_stable:
        classification = "minimum"
        interpretation = ("R = 1 is a true minimum of the free energy "
                          "functional (stable attractor in all tested directions)")
    elif any(r.is_stable for r in results):
        classification = "saddle"
        interpretation = (f"R = 1 is a saddle point: stable in "
                          f"{sum(r.is_stable for r in results)} directions, "
                          f"unstable in {len(unstable_axes)} directions "
                          f"({', '.join(unstable_axes)})")
    else:
        classification = "maximum"
        interpretation = ("R = 1 is a maximum of the free energy functional "
                          "(unstable in all tested directions)")

    return {
        'eigenvalues': eigenvalues,
        'classification': classification,
        'interpretation': interpretation,
        'all_stable': all_stable,
        'unstable_axes': unstable_axes,
    }
