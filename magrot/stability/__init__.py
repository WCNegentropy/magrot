"""
Entropic hypothesis testing module (Phase 2B).

Implements the four empirical tests that determine the thermodynamic identity
of R = 1:

Test A -- Hessian eigenvalue spectrum (hessian.py)
Test B -- Entropy production accounting at equilibrium (entropy_audit.py)
Test C -- Unconstrained relaxation / constraint removal (manifold.py)
         Supports full 3D grids via volume-averaging R over theta and z.
Test D support -- Basin-of-attraction characterization (attractors.py)

Validated results (1D + 3D):
  - R = 1 is a saddle point (1D) / minimum (3D) of the free energy
  - Entropy is being produced at R = 1 (driven steady state, not global eq.)
  - Constraint boundaries are soft for the Z-pinch (gradual R departure)
  - 3D classification change (saddle -> minimum) due to volume-averaged
    perturbations; mode-resolved perturbations needed for instability detection
"""
