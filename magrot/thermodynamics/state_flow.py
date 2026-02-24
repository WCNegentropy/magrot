"""
Entropy-parameterized state flow engine (Phase 2C, Test D).

Replaces time-based evolution with variational relaxation along the free
energy gradient, parameterized by total entropy produced sigma.

Key evolution principle:
    dF/d_sigma <= 0      (second law -- guaranteed monotonic)
    Near equilibrium:    Onsager minimum entropy production -> smooth relaxation
    Far from equilibrium: maximum entropy production -> rapid reconfiguration

Relaxation schemes:
    1. Steepest descent:  delta_B proportional to -delta_F/delta_B
    2. L-BFGS on F[B, p]  (faster convergence for smooth landscapes)
    3. Onsager linear response near equilibrium

For the thin-shell Z-pinch (Test D), the state is {a(sigma), v(sigma)} and
the evolution reduces to an ODE system in sigma rather than t.
"""

import numpy as np
from scipy.constants import mu_0, pi
from scipy.integrate import solve_ivp

from .diagnostics import StateFlowRecord


# ── Thin-shell Z-pinch: entropy-parameterized evolution (Test D) ─────────

def zpinch_entropy_flow(
    I=1e4, a_eq=0.01, a0=None, v0=0.0, gamma=5.0 / 3.0,
    rho_L=None, nu=0.0, T_plasma=1e7, eta=1e-6,
    sigma_max=None, n_points=5000, rtol=1e-10, atol=1e-13,
):
    """Integrate Z-pinch thin-shell dynamics parameterized by entropy sigma.

    This reproduces the v1 damped Z-pinch (mhd_1d.zpinch_thin_shell) but
    uses total entropy produced sigma as the independent variable instead
    of clock time t.

    The mapping from time to entropy:
        d_sigma = S_dot * dt
        dt / d_sigma = 1 / S_dot

    where S_dot is the instantaneous total entropy production rate.

    State vector: y = [a, v, t]  (radius, radial velocity, accumulated time)
    Independent variable: sigma (total entropy produced)

    ODE system:
        da/d_sigma = v * (dt/d_sigma) = v / S_dot
        dv/d_sigma = F_net/m * (dt/d_sigma) = F_net / (rho_L * S_dot)
        dt/d_sigma = 1 / S_dot

    Entropy production for resistive Z-pinch:
        S_dot = integral eta |J|^2 / T dV
    For thin shell with uniform current J = I / (pi a^2):
        S_dot = eta * (I/(pi a^2))^2 / T * pi a^2 * L_z
    where L_z is the length per unit length = 1.
        S_dot = eta * I^2 / (pi a^2 T)

    Plus the kinematic damping nu contributes to entropy:
        S_dot_damp = nu * v^2 / T

    Parameters
    ----------
    I : float
        Plasma current (A).
    a_eq : float
        Bennett equilibrium radius (m).
    a0 : float or None
        Initial radius (defaults to 0.5 * a_eq for compressed start).
    v0 : float
        Initial radial velocity (m/s).
    gamma : float
        Adiabatic index.
    rho_L : float or None
        Linear mass density (kg/m).
    nu : float
        Damping coefficient.
    T_plasma : float
        Plasma temperature (K) for entropy computation.
    eta : float
        Resistivity (Ohm m) for entropy computation.
    sigma_max : float or None
        Maximum total entropy to produce.  If None, auto-estimated.
    n_points : int
        Number of output points.
    rtol, atol : float
        Integrator tolerances.

    Returns
    -------
    dict with sigma, a, v, t, R, F, S_dot arrays.
    """
    if a0 is None:
        a0 = 0.5 * a_eq

    pB_eq = mu_0 * I ** 2 / (8 * pi ** 2 * a_eq ** 2)

    if rho_L is None:
        rho_L = 3.3e-7 * pi * a_eq ** 2

    def compute_forces(a, v):
        a = max(a, 1e-10)
        pB = mu_0 * I ** 2 / (8 * pi ** 2 * a ** 2)
        p_th = pB_eq * (a_eq / a) ** (2 * gamma)
        F_net = 2 * pi * a * (p_th - pB) - nu * v
        return F_net, pB, p_th

    def compute_R(a):
        pB = mu_0 * I ** 2 / (8 * pi ** 2 * a ** 2)
        p_th = pB_eq * (a_eq / a) ** (2 * gamma)
        return pB / max(p_th, 1e-30)

    def compute_F(a):
        """Free energy: magnetic + thermal - equilibrium value."""
        pB = mu_0 * I ** 2 / (8 * pi ** 2 * a ** 2)
        p_th = pB_eq * (a_eq / a) ** (2 * gamma)
        # Energy per unit length (integrated over cross section)
        E_mag = pB * pi * a ** 2  # approximate
        E_th = p_th * pi * a ** 2 / (gamma - 1)
        # Equilibrium values
        E_mag_eq = pB_eq * pi * a_eq ** 2
        E_th_eq = pB_eq * pi * a_eq ** 2 / (gamma - 1)
        # Kinetic energy
        E_kin = 0.5 * rho_L * 0  # evaluated separately
        return (E_mag + E_th) - (E_mag_eq + E_th_eq)

    def compute_S_dot(a, v):
        """Total entropy production rate (W/K per unit length)."""
        a = max(a, 1e-10)
        # Ohmic: eta * J^2 / T * volume
        J = I / (pi * a ** 2)
        S_dot_ohmic = eta * J ** 2 / T_plasma * pi * a ** 2

        # Damping dissipation (converts to heat -> entropy)
        S_dot_damp = nu * v ** 2 / T_plasma if nu > 0 else 0.0

        # Total entropy production
        S_dot_total = S_dot_ohmic + S_dot_damp

        return max(S_dot_total, 1e-30)

    # Estimate sigma_max from characteristic dissipation
    if sigma_max is None:
        S0 = compute_S_dot(a0, 0.0)
        # Estimate how much entropy to produce: ~ F_initial / T
        F0 = abs(compute_F(a0))
        sigma_max = max(10.0 * F0 / T_plasma, S0 * 1e-2)
        # Ensure a reasonable range
        sigma_max = max(sigma_max, 1e-10)

    def ode(sigma, y):
        a, v, t_acc = y
        a = max(a, 1e-10)

        F_net, _, _ = compute_forces(a, v)
        S_dot = compute_S_dot(a, v)
        dt_ds = 1.0 / S_dot

        da_ds = v * dt_ds
        dv_ds = F_net / rho_L * dt_ds
        dt_ds_val = dt_ds

        return [da_ds, dv_ds, dt_ds_val]

    sigma_span = (0, sigma_max)
    sigma_eval = np.linspace(0, sigma_max, n_points)

    sol = solve_ivp(
        ode, sigma_span, [a0, v0, 0.0],
        t_eval=sigma_eval, method='RK45', rtol=rtol, atol=atol,
        max_step=sigma_max / 100,
    )

    if not sol.success:
        # Retry with tighter settings
        sol = solve_ivp(
            ode, sigma_span, [a0, v0, 0.0],
            t_eval=sigma_eval, method='DOP853', rtol=rtol * 0.1, atol=atol * 0.1,
            max_step=sigma_max / 500,
        )

    a_arr = sol.y[0]
    v_arr = sol.y[1]
    t_arr = sol.y[2]
    sigma_arr = sol.t

    R_arr = np.array([compute_R(a) for a in a_arr])
    F_arr = np.array([compute_F(a) for a in a_arr])
    S_dot_arr = np.array([compute_S_dot(a, v) for a, v in zip(a_arr, v_arr)])

    return {
        'sigma': sigma_arr,
        'a': a_arr,
        'v': v_arr,
        't': t_arr,
        'R': R_arr,
        'F': F_arr,
        'S_dot': S_dot_arr,
        'a_eq': a_eq,
        'converged': sol.success,
    }


# ── Field-based steepest descent relaxation ──────────────────────────────────

def steepest_descent_step_cylindrical(B, grid, p=None, gamma=5.0 / 3.0,
                                      eta=1e-6, T=1e7, d_sigma=1e-6):
    """One step of steepest descent relaxation on a cylindrical grid.

    Adjusts B to reduce F by an amount corresponding to entropy d_sigma.

    delta_B proportional to -J (current drives the relaxation toward
    force-free state).  The step size is set by the entropy increment.

    Parameters
    ----------
    B : ndarray, shape (*grid.R.shape, 3)
        Current magnetic field.
    grid : CylindricalGrid
    p : ndarray or None
    gamma : float
    eta : float
        Resistivity.
    T : float or ndarray
        Temperature.
    d_sigma : float
        Entropy increment for this step.

    Returns
    -------
    B_new : ndarray
        Updated magnetic field.
    dF : float
        Change in free energy (should be negative).
    s_dot_total : float
        Entropy production rate at this step.
    """
    from ..stress.maxwell import compute_forces_conservative_cyl
    from .free_energy import free_energy_cylindrical, _volume_elements_cylindrical
    from .entropy import entropy_production_resistive, total_entropy_production_cylindrical

    # Current state
    phys = compute_forces_conservative_cyl(B, grid, p)
    J = phys['J']

    # Entropy production
    T_arr = np.full_like(phys['Bmag'], T) if np.isscalar(T) else T
    s_dot, _, _ = entropy_production_resistive(J, T_arr, eta=eta)
    S_dot_total = total_entropy_production_cylindrical(s_dot, grid)

    # Time equivalent of this entropy step
    dt_equiv = d_sigma / max(S_dot_total, 1e-30)

    # Steepest descent: B evolves to reduce |J x B|
    # In resistive MHD: dB/dt = -curl(eta J) = -eta curl(J) (for uniform eta)
    # Simplified: delta_B = -eta * J * dt
    delta_B = -eta * J * dt_equiv

    B_new = B + delta_B

    # Compute free energy change
    F_old, _ = free_energy_cylindrical(B, grid, p, gamma)
    F_new, _ = free_energy_cylindrical(B_new, grid, p, gamma)
    dF = F_new - F_old

    return B_new, float(dF), float(S_dot_total)


def relaxation_flow_cylindrical(B_init, grid, p=None, gamma=5.0 / 3.0,
                                eta=1e-6, T=1e7, n_steps=500,
                                d_sigma=None, sigma_max=None,
                                record_every=1):
    """Run steepest descent relaxation on a cylindrical grid, parameterized
    by total entropy produced sigma.

    Parameters
    ----------
    B_init : ndarray
        Initial magnetic field.
    grid : CylindricalGrid
    p : ndarray or None
    gamma : float
    eta : float
    T : float or ndarray
    n_steps : int
    d_sigma : float or None
        Entropy per step. If None, auto-estimated.
    sigma_max : float or None
        Stop when total entropy reaches this. If None, run all n_steps.
    record_every : int
        Record diagnostics every N steps.

    Returns
    -------
    B_final : ndarray
        Relaxed magnetic field.
    record : StateFlowRecord
        Trajectory diagnostics.
    """
    from .free_energy import free_energy_cylindrical

    B = B_init.copy()
    record = StateFlowRecord()

    # Auto-estimate d_sigma
    if d_sigma is None:
        F0, _ = free_energy_cylindrical(B, grid, p, gamma)
        d_sigma = max(abs(F0) / (T * n_steps), 1e-15)

    sigma_total = 0.0

    for step in range(n_steps):
        if sigma_max is not None and sigma_total >= sigma_max:
            break

        B, dF, S_dot = steepest_descent_step_cylindrical(
            B, grid, p, gamma, eta, T, d_sigma
        )
        sigma_total += d_sigma

        if step % record_every == 0:
            F_curr, _ = free_energy_cylindrical(B, grid, p, gamma)
            record.record(
                sigma_val=sigma_total,
                F_val=F_curr,
                S_dot_val=S_dot,
            )

    return B, record
