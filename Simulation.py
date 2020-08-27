
from Model import Model
import numpy as np
from scipy.integrate import solve_ivp

MAX_STEP = 0.01


def simulate(model, t_span, theta0):
    """
    Run a simulation of the model.

    Parameters
    ----------
    model : Model
        Model to simulate.
    t_span : Tuple of two floats
        Interval of integration (t0, tf).
    theta0 : (n,) NumPy array
        Initial angle at each bus.

    Returns
    -------
    t : (T,) NumPy array
        Array of time points.
    angles : (n, T) NumPy array
        Array of angle points.
    freqs : (n, T) NumPy array
        Array of nodal frequencies.
    """

    # Simulate trajectory
    fun = lambda t, x: model.bus_status(x)[1]
    sol = solve_ivp(fun, t_span, theta0, max_step=MAX_STEP)
    angles = sol.y

    # Analyze trajectory
    freqs = np.empty(angles.shape)
    for t in range(angles.shape[1]):
        _, freqs[:, t] = model.bus_status(angles[:, t])
    return sol.t, angles, freqs
