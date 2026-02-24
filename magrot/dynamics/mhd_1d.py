"""
1D radial MHD toy model for Z-pinch dynamics.

Thin-shell model: d^2a/dt^2 = (1/rho_L) * [p_thermal - p_magnetic]
with adiabatic pressure scaling and optional resistive damping.
"""

import numpy as np
from scipy.constants import mu_0, pi
from scipy.integrate import solve_ivp


def zpinch_thin_shell(
    I=1e4, a_eq=0.01, a0=None, v0=0.0, gamma=5.0 / 3.0,
    rho_L=None, nu=0.0, t_periods=5, n_points=5000,
):
    """Integrate Z-pinch thin-shell radial dynamics.

    Parameters
    ----------
    I : float
        Plasma current (A).
    a_eq : float
        Bennett equilibrium radius (m).
    a0 : float or None
        Initial radius (defaults to a_eq).
    v0 : float
        Initial radial velocity (m/s).
    gamma : float
        Adiabatic index.
    rho_L : float or None
        Linear mass density (kg/m). Auto-computed if None.
    nu : float
        Damping coefficient for resistive relaxation.
    t_periods : int
        Number of estimated oscillation periods to simulate.
    n_points : int
        Number of output time-steps.

    Returns
    -------
    dict with t, a, v, R (time, radius, velocity, R_universal arrays).
    """
    if a0 is None:
        a0 = a_eq
    pB_eq = mu_0 * I**2 / (8 * pi**2 * a_eq**2)
    if rho_L is None:
        rho_L = 3.3e-7 * pi * a_eq**2

    def ode(t, y):
        a, v = y
        a = max(a, 1e-8)
        pB = mu_0 * I**2 / (8 * pi**2 * a**2)
        p_th = pB_eq * (a_eq / a) ** (2 * gamma)
        F_net = 2 * pi * a * (p_th - pB) - nu * v
        return [v, F_net / rho_L]

    def compute_R(a):
        pB = mu_0 * I**2 / (8 * pi**2 * a**2)
        p_th = pB_eq * (a_eq / a) ** (2 * gamma)
        return pB / max(p_th, 1e-30)

    omega_est = np.sqrt((2 * gamma - 2) * 2 * pi * pB_eq / rho_L)
    T_est = 2 * pi / omega_est
    t_span = (0, t_periods * T_est)
    t_eval = np.linspace(*t_span, n_points)

    sol = solve_ivp(ode, t_span, [a0, v0], t_eval=t_eval,
                    method='RK45', rtol=1e-12, atol=1e-15)

    R_t = np.array([compute_R(a) for a in sol.y[0]])

    return {
        't': sol.t,
        'a': sol.y[0],
        'v': sol.y[1],
        'R': R_t,
        'T_est': T_est,
    }
