"""
Thermodynamic diagnostics for state flow tracking.

Tracks F(sigma), S_dot(sigma), Lyapunov functional verification, and
the correlation between |R - 1| and local entropy production / free energy.

Used both during state flow evolution (Phase 2C) and for post-hoc
analysis of existing equilibria (Phase 2A).
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class StateFlowRecord:
    """Accumulates thermodynamic quantities along an entropy-parameterized
    evolution trajectory.

    Each entry corresponds to one step in sigma (total entropy produced).
    """
    sigma: list = field(default_factory=list)
    F: list = field(default_factory=list)
    S_dot: list = field(default_factory=list)
    L_R: list = field(default_factory=list)         # Lyapunov |R-1|^2
    R_core: list = field(default_factory=list)
    R_edge: list = field(default_factory=list)
    K: list = field(default_factory=list)            # helicity
    Phi: list = field(default_factory=list)           # flux
    P_dissipated: list = field(default_factory=list)

    def record(self, sigma_val, F_val=None, S_dot_val=None, L_R_val=None,
               R_core_val=None, R_edge_val=None, K_val=None, Phi_val=None,
               P_val=None):
        """Append one snapshot to the trajectory."""
        self.sigma.append(sigma_val)
        self.F.append(F_val)
        self.S_dot.append(S_dot_val)
        self.L_R.append(L_R_val)
        self.R_core.append(R_core_val)
        self.R_edge.append(R_edge_val)
        self.K.append(K_val)
        self.Phi.append(Phi_val)
        self.P_dissipated.append(P_val)

    def as_arrays(self):
        """Return all trajectories as numpy arrays."""
        return {
            'sigma': np.array(self.sigma, dtype=float),
            'F': np.array(self.F, dtype=float),
            'S_dot': np.array(self.S_dot, dtype=float),
            'L_R': np.array(self.L_R, dtype=float),
            'R_core': np.array(self.R_core, dtype=float),
            'R_edge': np.array(self.R_edge, dtype=float),
            'K': np.array(self.K, dtype=float),
            'Phi': np.array(self.Phi, dtype=float),
            'P_dissipated': np.array(self.P_dissipated, dtype=float),
        }


def verify_monotonic_decrease(F_trajectory, atol=1e-12):
    """Check that F(sigma) is monotonically decreasing (second law).

    Parameters
    ----------
    F_trajectory : array-like
        Free energy values along the trajectory.
    atol : float
        Absolute tolerance for numerical noise.

    Returns
    -------
    is_monotonic : bool
        True if F never increases beyond tolerance.
    violations : ndarray
        Indices where F increased.
    max_violation : float
        Largest upward step in F.
    """
    F = np.asarray(F_trajectory, dtype=float)
    dF = np.diff(F)
    violations = np.where(dF > atol)[0]
    max_violation = float(np.max(dF)) if len(dF) > 0 else 0.0
    return len(violations) == 0, violations, max_violation


def verify_constraint_conservation(trajectory_values, rtol=0.01):
    """Check that a conserved quantity stays within rtol of its initial value.

    Parameters
    ----------
    trajectory_values : array-like
        Values of the conserved quantity at each step.
    rtol : float
        Relative tolerance.

    Returns
    -------
    is_conserved : bool
    max_deviation : float
        Maximum relative deviation from initial value.
    """
    vals = np.asarray(trajectory_values, dtype=float)
    if len(vals) == 0:
        return True, 0.0
    ref = vals[0] if abs(vals[0]) > 1e-30 else 1.0
    deviations = np.abs(vals - vals[0]) / abs(ref)
    max_dev = float(np.max(deviations))
    return max_dev < rtol, max_dev


def correlation_R_sdot(R_field, s_dot_field, mask=None):
    """Compute the Pearson correlation between |R - 1| and s_dot.

    The plan predicts these should be strongly correlated: regions with
    larger force imbalance (|R - 1|) should produce entropy faster.

    Parameters
    ----------
    R_field : ndarray
    s_dot_field : ndarray
    mask : ndarray of bool or None
        If given, only include points where mask is True.

    Returns
    -------
    r_pearson : float
        Pearson correlation coefficient.
    """
    R_dev = np.abs(R_field - 1.0).ravel()
    s_flat = s_dot_field.ravel()

    if mask is not None:
        m = mask.ravel()
        R_dev = R_dev[m]
        s_flat = s_flat[m]

    # Filter out near-zero entries that would skew the correlation
    valid = (s_flat > 0) & np.isfinite(R_dev) & np.isfinite(s_flat)
    if np.sum(valid) < 10:
        return float('nan')

    R_dev = R_dev[valid]
    s_flat = s_flat[valid]

    R_mean = np.mean(R_dev)
    s_mean = np.mean(s_flat)
    cov = np.mean((R_dev - R_mean) * (s_flat - s_mean))
    std_R = np.std(R_dev)
    std_s = np.std(s_flat)

    if std_R < 1e-30 or std_s < 1e-30:
        return float('nan')

    return float(cov / (std_R * std_s))


def correlation_R_free_energy(R_field, u_field, mask=None):
    """Compute the Pearson correlation between |R - 1| and local energy density.

    Tests whether |R - 1| is proportional to the local free energy density
    gradient, as predicted by the R-F connection in the plan.

    Parameters
    ----------
    R_field : ndarray
    u_field : ndarray
        Local energy density (from free_energy.total_energy_density).
    mask : ndarray of bool or None

    Returns
    -------
    r_pearson : float
    """
    return correlation_R_sdot(R_field, u_field, mask=mask)
