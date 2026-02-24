"""
Thermodynamic state flow module (Phase 2A/2C).

Replaces the v1 time-based dynamics with entropy-parameterized state evolution.
Evolution parameter is sigma (total irreversible entropy produced), not clock
time t.  This guarantees monotonic free energy decrease (second law) and gives
physical meaning to every step in the evolution.

Submodules
----------
free_energy   -- F[B, p] functional computation
entropy       -- Local entropy production rate density s_dot(x)
constraints   -- Magnetic helicity K, total flux Phi, total mass M
diagnostics   -- F(sigma) tracking, Lyapunov verification
state_flow    -- Variational relaxation engine (Phase 2C)
"""
