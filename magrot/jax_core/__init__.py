"""
MagRot Forge — JAX-accelerated physics core.

Pure-functional JAX reimplementation of the MAGROT physics pipeline,
validated against the NumPy reference to 1e-6 tolerance.  Every function
is JIT-compilable and fully differentiable via jax.grad.

Modules
-------
numerics_jax    : 4th-order finite differences (functional, periodic-aware)
curvature_jax   : Field-line curvature kappa = (b . nabla) b
maxwell_jax     : curl(B), J x B Lorentz force
decompose_jax   : Inward / outward force decomposition
metrics_jax     : R_universal (the loss-critical metric)
free_energy_jax : F[B,p], Lyapunov L_R = integral |R-1|^2 dV
grid_jax        : Static grid arrays for JIT (no dataclass)
biot_savart     : Differentiable coil-to-B-field integrator
"""
