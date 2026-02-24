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

Author: Mikeal Clark / WCNEGENTROPY HOLDINGS LLC
License: MIT
"""

__version__ = "0.1.0"
