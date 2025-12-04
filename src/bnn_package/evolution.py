#!/usr/bin/env/python3

# ======================= Libraries

import numpy as np
import numba

# ======================= Functions


def get_coupling_operator(Adjacence, type_diff: str):
    """
    Pre-computes the normalized adjacency matrix.
    Run this once in pure python rather jit compilation.
    """
    if type_diff == "Diffusive":
        weights = np.sum(Adjacence, axis=1)
        if np.any(weights == 0):
            raise ValueError("Nodes with degree 0 detected.")
        Coupling_op = (
            Adjacence / weights[:, np.newaxis]
        )  # We use broadcasting to divide each row by its weight
    elif type_diff == "Laplacian":
        degrees = np.sum(Adjacence, axis=1)
        Deg = np.diag(degrees)
        # The Laplacian
        Coupling_op = Deg - Adjacence
    else:
        raise ValueError("No method gave for coupling")
    # === CRITICAL PERFORMANCE POINT ===
    # We force the matrix to be laid out contiguously in memory.
    # BLAS (the math engine behind @) runs much faster on contiguous arrays.
    return np.ascontiguousarray(Coupling_op)


@numba.jit(
    nopython=True
)  # option mode "fastmath=True" can be swith on if data are clean enough
def coupling_func(State_voltage_active, eps, Coupling_op, Model, type_diff: str):
    """
    Applies diffusive coupling to the State.

    State: (N_nodes, Dimension)
    Model : type of coupling : 'Henon' or 'FhN'
    Norm_Adj: Pre-computed (N_nodes, N_nodes) matrix
    """
    result = np.zeros(State_voltage_active.shape)

    if Model == "Henon":
        interaction = (
            Coupling_op @ State_voltage_active
        )  # For sparse or huge network (more than 1000 nodes with only few neighbors by node use an explicit for loop with numba)
        result = (1.0 - eps) * State_voltage_active + eps * interaction

    if Model == "FhN":
        if type_diff == "Laplacian":
            result = eps * (Coupling_op @ State_voltage_active)
        elif type_diff == "Diffusive":
            neighbor_avg = Coupling_op @ State_voltage_active
            result = (1.0 - eps) * State_voltage_active + eps * neighbor_avg
        else:
            raise ValueError("No method gave for the coupling")

    return result


@numba.jit(nopython=True)
def fhn_derivatives(State, params, N_p, Coupling_op, C_r, type_diff, I_input=0):
    """
    Computes the derivative dS/dt for a system of oscillators.
    State shape: (N_nodes, 3) -> col 0 is v_e (voltage of active node), col 1 is g (recovery), col 2 is v_p (voltage of passive node)
    Params: bibliothèque (Note: dt is not used here, but passed in params)
    A : adjacency matrix (N*N)
    N_p : 1D array which represents a diagonal matrix
    """
    v_e = np.ascontiguousarray(State[:, 0])
    g = State[:, 1]
    v_p = State[:, 2]

    A = params[0]
    alpha = params[1]
    Eps = params[2]
    K = params[3]
    V_RP = params[4]

    dState = np.empty_like(State)
    # Termes de couplage
    coupling_term = coupling_func(v_e, Eps, Coupling_op, "FhN", type_diff)

    # dv_e/dt = A*v_e*(v_e-alpha)*(1-v_e) - g + N_p*C_r*(v_p-v_e) - D*coupling_term[:,0]
    dState[:, 0] = (
        A * v_e * (v_e - alpha) * (1 - v_e)
        - g
        + ((C_r * (v_p - v_e)) * N_p)
        - coupling_term
        + I_input
    )

    # dg/dt = Eps(v_e - g)
    dState[:, 1] = Eps * (v_e - g)

    # dv_p/dt = K(V_RP - v_p) - C_r(v_p - v_e)
    dState[:, 2] = K * (V_RP - v_p) - C_r * (v_p - v_e)
    return dState


@numba.jit(nopython=True)
def step_fhn_rk4(State, params, N_p, Coupling_op, C_r, type_diff):
    """
    Performs one RK4 step.
    Matches signature: (State, params) -> New_State
    """
    dt = params[5]  # We assume dt is stored in the params array
    # RK4 Integration Logic

    # k1 = f(y)
    k1 = fhn_derivatives(State, params, N_p, Coupling_op, C_r, type_diff)
    # k2 = f(y + 0.5*dt*k1)
    k2 = fhn_derivatives(
        State + 0.5 * dt * k1, params, N_p, Coupling_op, C_r, type_diff
    )
    # k3 = f(y + 0.5*dt*k2)
    k3 = fhn_derivatives(
        State + 0.5 * dt * k2, params, N_p, Coupling_op, C_r, type_diff
    )
    # k4 = f(y + dt*k3)
    k4 = fhn_derivatives(State + dt * k3, params, N_p, Coupling_op, C_r, type_diff)
    # y_new = y + (dt/6) * (k1 + 2k2 + 2k3 + k4)
    State_next = State + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return State_next


@numba.jit(nopython=True)
def step_henon(State, params):
    """
    Performs one Henon map step.
    Matches signature: (State, params) -> New_State
    """
    x = State[:, 0]
    y = State[:, 1]
    # Unpack parameters
    a = params[0]
    b = params[1]
    # Pre-allocate derivative array
    State_next = np.empty_like(State)
    # x(n+1) = 1-a*x(n)^2 + y(n)
    State_next[:, 0] = 1 - a * x * x + y
    # y(n+1) = b*x(n)
    State_next[:, 1] = b * x
    return State_next


@numba.jit(nopython=True)
def evolve_system(
    State_0, N_steps, params, model_step_func, N_p, Coupling_op, C_r, type_diff
):
    """
    Main generic solver.

    State_0 = Initial conditions
    N_steps = length of the simulation (total time)
    params = matrix of parameters for a given system
    model_step_function = dynamical system's function
    coupling_func = function defining the coupling
    coupling_op = operator needed for the coupling_func
    eps = value of the coupling

    return the tensor of evolution state with the shape (N_steps, n_nodes, dim)
    """
    n_nodes, dim = State_0.shape
    Trajectory = np.zeros((N_steps, n_nodes, dim), dtype=State_0.dtype)
    Trajectory[0] = State_0
    progress_stride = max(1, N_steps // 10)
    for t in range(0, N_steps - 1):
        if t % progress_stride == 0:
            print("Simulation Progress:", (t * 100) // N_steps, "%")
        Trajectory[t + 1] = model_step_func(
            Trajectory[t], params, N_p, Coupling_op, C_r, type_diff
        )
    print("Simulation Progress: 100%")
    return Trajectory
