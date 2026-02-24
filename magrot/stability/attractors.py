"""
Basin of attraction characterization for R = 1.

Synthesizes results from Tests A, B, C to determine:
    - How far the system can be perturbed before it doesn't return to R = 1
    - The topology of the attraction basin (deep well, shallow bowl, mesa)
    - Whether different perturbation directions have different basin widths

This module provides the analysis framework; the actual perturbation data
comes from hessian.py (Test A) and manifold.py (Test C).
"""

import numpy as np
from dataclasses import dataclass


@dataclass
class BasinCharacterization:
    """Complete characterization of the R = 1 attraction basin."""
    hessian_eigenvalues: dict          # axis_name -> H value
    basin_widths: dict                 # axis_name -> epsilon at which L > threshold
    mesa_edges: dict                   # constraint -> alpha at departure
    classification: str                # 'valley-on-mesa', 'pure-attractor', etc.
    interpretation: str                # Human-readable summary
    is_driven_state: bool              # From entropy audit (Test B)
    S_dot_at_equilibrium: float        # Entropy production rate at R ~ 1


def estimate_basin_width(epsilon_values, F_values, threshold_factor=2.0):
    """Estimate the perturbation amplitude at which the system leaves the
    basin of attraction.

    Defines "leaving the basin" as F(epsilon) > threshold_factor * F(0).

    Parameters
    ----------
    epsilon_values : ndarray
    F_values : ndarray
    threshold_factor : float

    Returns
    -------
    width : float
        Estimated basin half-width in epsilon.
    """
    eps = np.asarray(epsilon_values)
    F = np.asarray(F_values)

    # Find minimum (should be near eps = 0)
    i_min = np.argmin(F)
    F_min = F[i_min]

    if F_min <= 0:
        F_min = np.mean(F)

    threshold = threshold_factor * F_min

    # Find where F exceeds threshold on positive side
    pos_mask = eps > eps[i_min]
    if np.any(pos_mask):
        F_pos = F[pos_mask]
        eps_pos = eps[pos_mask]
        exceed = np.where(F_pos > threshold)[0]
        width_pos = float(eps_pos[exceed[0]]) if len(exceed) > 0 else float(eps_pos[-1])
    else:
        width_pos = float(eps[-1])

    # Find where F exceeds threshold on negative side
    neg_mask = eps < eps[i_min]
    if np.any(neg_mask):
        F_neg = F[neg_mask]
        eps_neg = eps[neg_mask]
        exceed = np.where(F_neg > threshold)[0]
        width_neg = float(abs(eps_neg[exceed[-1]])) if len(exceed) > 0 else float(abs(eps_neg[0]))
    else:
        width_neg = float(abs(eps[0]))

    return min(width_pos, width_neg)


def classify_attractor(hessian_results, manifold_results, entropy_audit_result):
    """Classify the thermodynamic character of R = 1 based on Tests A, B, C.

    Parameters
    ----------
    hessian_results : list of HessianResult
        From Test A.
    manifold_results : list of ConstraintRelaxationResult
        From Test C.
    entropy_audit_result : EntropyAuditResult
        From Test B.

    Returns
    -------
    BasinCharacterization
    """
    # Hessian eigenvalues
    eigenvalues = {r.axis_name: r.H for r in hessian_results}
    all_stable = all(r.is_stable for r in hessian_results)
    any_stable = any(r.is_stable for r in hessian_results)

    # Basin widths
    widths = {}
    for r in hessian_results:
        widths[r.axis_name] = estimate_basin_width(r.epsilon_values, r.F_values)

    # Mesa edges
    edges = {}
    has_catastrophic = False
    for r in manifold_results:
        edges[r.scenario_name] = r.departure_alpha
        if r.is_catastrophic:
            has_catastrophic = True

    # Entropy audit
    is_driven = entropy_audit_result.is_driven_state
    S_dot_eq = entropy_audit_result.S_dot_total

    # Classification logic (from the plan's Table in Section 7.2)
    if all_stable and is_driven and has_catastrophic:
        classification = "valley-on-mesa"
        interpretation = (
            "R = 1 is a conditional entropy maximum on the constrained MHD "
            "equilibrium manifold (valley). The manifold itself is "
            "negentropic, requiring external energy throughput to maintain "
            "(mesa). Entropy is continuously produced at equilibrium "
            f"(S_dot = {S_dot_eq:.3e} W/K), confirming driven steady state. "
            "Sharp constraint boundaries confirm disruption-like behavior "
            "at the mesa edge."
        )
    elif all_stable and is_driven and not has_catastrophic:
        classification = "valley-soft-mesa"
        interpretation = (
            "R = 1 is a stable attractor within the constrained manifold. "
            "Constraint boundaries are soft (no disruption-like departure). "
            "The system is driven (S_dot > 0) but transitions gradually."
        )
    elif all_stable and not is_driven:
        classification = "pure-attractor"
        interpretation = (
            "R = 1 is the relaxation endpoint with no entropy production "
            "at equilibrium. This is a true thermodynamic equilibrium, "
            "not a driven state."
        )
    elif any_stable and not all_stable:
        unstable = [r.axis_name for r in hessian_results if not r.is_stable]
        classification = "saddle-point"
        interpretation = (
            f"R = 1 is a saddle point: stable in some directions, unstable "
            f"in {unstable}. The unstable directions may correspond to "
            "known MHD instability modes."
        )
    else:
        classification = "maximum"
        interpretation = (
            "R = 1 appears to be a maximum of the free energy functional. "
            "This contradicts v1 evidence and may indicate a formulation issue."
        )

    return BasinCharacterization(
        hessian_eigenvalues=eigenvalues,
        basin_widths=widths,
        mesa_edges=edges,
        classification=classification,
        interpretation=interpretation,
        is_driven_state=is_driven,
        S_dot_at_equilibrium=S_dot_eq,
    )
