"""
MagRot: Rotational-Vector Framework for Magnetic Field Dynamics
===============================================================

A dimensionless rotational parameter R(x) that encodes the local tendency
toward collapse, expansion, or equilibrium in magnetic field configurations.

Metrics:
    R_kappa     -- Metric A: curvature ratio (kappa / kappa_eq)
    R_collapse  -- Metric B: collapse indicator (1 + C)
    R_misalign  -- Metric C: current-field misalignment |B x curl(B)| / |B|^2
    R_universal -- Universal: |f_inward| / |f_outward|

Convention (force-ratio):
    R < 1  ->  Expanding (outward forces dominate)
    R = 1  ->  Equilibrium (force balance)
    R > 1  ->  Contracting (inward forces dominate)

v2 additions:
    thermodynamics/  -- Entropy-based state flow (replaces time-based dynamics)
        free_energy      -- F[B, p] functional computation
        entropy          -- Local entropy production rate s_dot(x)
        constraints      -- Helicity K, flux Phi, mass M conservation
        diagnostics      -- F(sigma) tracking, Lyapunov verification
        state_flow       -- Variational relaxation engine
    stability/       -- Entropic hypothesis tests
        hessian          -- Multi-axis perturbation Hessian eigenvalues (Test A)
        entropy_audit    -- Entropy production accounting at equilibrium (Test B)
        manifold         -- Constraint boundary mapping (Test C)
        attractors       -- Basin of attraction characterization

Author: WCNEGENTROPY HOLDINGS LLC
License: MIT
"""

__version__ = "0.2.0-dev"
