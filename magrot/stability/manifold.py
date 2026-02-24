"""
Constraint boundary mapping (Test C): unconstrained relaxation.

Goal: Determine what happens when the constraints that define the R = 1
manifold are violated.

Method:
    1. Start from equilibrium
    2. Simulate constraint removal scenarios:
       - Remove external toroidal field (B_tor decay)
       - Allow flux diffusion through boundary (resistive wall)
       - Remove pressure source (radiative cooling)
    3. Track R(x, sigma) as system evolves
    4. Map the "mesa edge" -- the constraint boundary beyond which R = 1
       ceases to exist

Success criteria:
    - Smooth departure + gradual R drift -> soft boundaries
    - Catastrophic departure + rapid R explosion -> sharp boundaries (disruption-like)
    - Different scenarios hit different boundaries -> constraint landscape has structure
"""

import numpy as np
from scipy.constants import mu_0
from dataclasses import dataclass


@dataclass
class ConstraintRelaxationResult:
    """Result of a constraint removal simulation."""
    scenario_name: str
    alpha_values: np.ndarray     # Constraint removal parameter (0 = full, 1 = removed)
    R_core: np.ndarray           # Core R at each alpha
    R_edge: np.ndarray           # Edge R at each alpha
    F_values: np.ndarray         # Free energy at each alpha
    L_values: np.ndarray         # Lyapunov |R-1|^2 at each alpha
    departure_alpha: float       # Alpha where |R - 1| exceeds threshold
    is_catastrophic: bool        # True if departure is sharp (large dL/dalpha)


# ── Constraint removal scenarios for cylindrical Z-pinch ─────────────────────

def remove_pressure_zpinch(B, p, grid, alpha):
    """Radiative cooling: scale pressure by (1 - alpha).

    alpha = 0: full equilibrium pressure
    alpha = 1: zero pressure (unconstrained)

    Parameters
    ----------
    B : ndarray
    p : ndarray
    grid : CylindricalGrid
    alpha : float in [0, 1]

    Returns
    -------
    B_pert, p_pert : ndarray
    """
    return B.copy(), p * (1.0 - alpha)


def remove_field_zpinch(B, p, grid, alpha):
    """Magnetic field decay: scale B by (1 - alpha).

    alpha = 0: full field
    alpha = 1: zero field

    Parameters
    ----------
    B : ndarray
    p : ndarray
    grid : CylindricalGrid
    alpha : float in [0, 1]

    Returns
    -------
    B_pert, p_pert : ndarray
    """
    return B * (1.0 - alpha), p.copy() if p is not None else None


def diffuse_boundary_zpinch(B, p, grid, alpha, a=0.01):
    """Flux diffusion: spread the boundary over a wider region.

    Simulates resistive wall effect by broadening the current channel.

    alpha = 0: sharp boundary at r = a
    alpha = 1: current spread uniformly to grid edge

    Parameters
    ----------
    B : ndarray
    p : ndarray
    grid : CylindricalGrid
    alpha : float
    a : float
        Nominal plasma radius.

    Returns
    -------
    B_pert, p_pert : ndarray
    """
    from ..fields.analytic import field_zpinch_bennett

    # Effective radius expands from a to grid edge
    a_eff = a + alpha * (grid.r[-1] * 0.8 - a)
    return field_zpinch_bennett(grid, I=1e4, a=a_eff)


# ── Constraint removal scenarios for toroidal equilibrium ────────────────────

def remove_toroidal_field(eq, alpha):
    """Toroidal field decay: scale Bphi by (1 - alpha).

    Parameters
    ----------
    eq : TokamakEquilibrium-like
    alpha : float in [0, 1]

    Returns
    -------
    Bphi_pert : ndarray
    """
    return eq.Bphi * (1.0 - alpha)


def remove_pressure_toroidal(eq, alpha):
    """Radiative cooling: scale pressure by (1 - alpha).

    Parameters
    ----------
    eq : TokamakEquilibrium-like
    alpha : float in [0, 1]

    Returns
    -------
    p_pert : ndarray
    """
    return eq.p_mat * (1.0 - alpha)


# ── Constraint relaxation sweep driver ───────────────────────────────────────

def constraint_relaxation_sweep_cylindrical(
    B, p, grid, scenario_name, removal_func,
    compute_R_func=None, n_alpha=21, alpha_range=(0.0, 1.0),
    threshold=0.5, **kwargs
):
    """Sweep a constraint removal parameter and track R departure.

    Parameters
    ----------
    B : ndarray
        Equilibrium magnetic field.
    p : ndarray
        Equilibrium pressure.
    grid : CylindricalGrid
    scenario_name : str
    removal_func : callable
        Function(B, p, grid, alpha, **kwargs) -> (B_pert, p_pert).
    compute_R_func : callable or None
        Function(B, grid, p) -> R_field. Uses R_universal if None.
    n_alpha : int
    alpha_range : tuple
    threshold : float
        |R - 1| threshold for "departure" detection.
    **kwargs
        Extra args passed to removal_func.

    Returns
    -------
    ConstraintRelaxationResult
    """
    from ..rotation.metrics import compute_all_metrics
    from ..thermodynamics.free_energy import (
        lyapunov_R_cylindrical, free_energy_cylindrical,
    )

    alphas = np.linspace(alpha_range[0], alpha_range[1], n_alpha)
    R_core = np.zeros(n_alpha)
    R_edge = np.zeros(n_alpha)
    F_vals = np.zeros(n_alpha)
    L_vals = np.zeros(n_alpha)

    r = grid.r
    s = (slice(None), 0, 0)  # 1D slice for cylindrical

    for i, alpha in enumerate(alphas):
        B_pert, p_pert = removal_func(B, p, grid, alpha, **kwargs)

        if compute_R_func is not None:
            R_field = compute_R_func(B_pert, grid, p_pert)
        else:
            res = compute_all_metrics(B_pert, grid, p_mat=p_pert)
            R_field = res['R_universal']

        R_1d = R_field[s]

        # Core = inner 20%, edge = outer 20%
        n_r = len(r)
        core_slice = slice(0, max(1, n_r // 5))
        edge_slice = slice(max(1, 4 * n_r // 5), n_r)
        R_core[i] = float(np.mean(R_1d[core_slice]))
        R_edge[i] = float(np.mean(R_1d[edge_slice]))

        L, _ = lyapunov_R_cylindrical(R_field, grid)
        L_vals[i] = L

        F, _ = free_energy_cylindrical(B_pert, grid, p_pert)
        F_vals[i] = F

    # Find departure point
    R_dev = np.abs(R_core - 1.0)
    departure_idx = np.argmax(R_dev > threshold)
    departure_alpha = float(alphas[departure_idx]) if R_dev[departure_idx] > threshold else float(alphas[-1])

    # Catastrophic if dL/dalpha spikes (normalized rate of change > 10)
    dL = np.diff(L_vals)
    dalpha = np.diff(alphas)
    dL_dalpha = dL / np.maximum(dalpha, 1e-30)
    if len(dL_dalpha) > 0 and L_vals[0] > 0:
        max_rate = float(np.max(np.abs(dL_dalpha))) / max(L_vals[0], 1e-30)
        is_catastrophic = max_rate > 10.0
    else:
        is_catastrophic = False

    return ConstraintRelaxationResult(
        scenario_name=scenario_name,
        alpha_values=alphas,
        R_core=R_core,
        R_edge=R_edge,
        F_values=F_vals,
        L_values=L_vals,
        departure_alpha=departure_alpha,
        is_catastrophic=is_catastrophic,
    )


def mesa_edge_summary(results):
    """Summarize constraint boundary characteristics from multiple scenarios.

    Parameters
    ----------
    results : list of ConstraintRelaxationResult

    Returns
    -------
    summary : dict
    """
    boundaries = {}
    for r in results:
        boundaries[r.scenario_name] = {
            'departure_alpha': r.departure_alpha,
            'is_catastrophic': r.is_catastrophic,
            'max_L': float(np.max(r.L_values)),
        }

    has_sharp = any(r.is_catastrophic for r in results)
    has_soft = any(not r.is_catastrophic for r in results)

    if has_sharp and has_soft:
        landscape = "structured"
        interpretation = ("The constraint landscape has structure: some "
                          "boundaries are sharp (disruption-like) while "
                          "others are soft (gradual departure)")
    elif has_sharp:
        landscape = "sharp"
        interpretation = ("All tested constraint boundaries are sharp "
                          "(disruption-like behavior)")
    else:
        landscape = "soft"
        interpretation = ("All tested constraint boundaries are soft "
                          "(gradual R departure)")

    return {
        'boundaries': boundaries,
        'landscape': landscape,
        'interpretation': interpretation,
    }
