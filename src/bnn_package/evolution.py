#!/usr/bin/env/python3

# ======================= Libraries

import numpy as np
import numba

# ======================= Functions


def get_coupling_operator(Adjacence):
    """
    Pre-computes the normalized adjacency matrix.
    Run this once in pure python rather jit compilation.
    """
    weights = np.sum(Adjacence, axis=1)
    if np.any(weights == 0):
        raise ValueError("Nodes with degree 0 detected.")
    normalized_adj = (
        Adjacence / weights[:, np.newaxis]
    )  # We use broadcasting to divide each row by its weight

    # === CRITICAL PERFORMANCE POINT ===
    # We force the matrix to be laid out contiguously in memory.
    # BLAS (the math engine behind @) runs much faster on contiguous arrays.
    return np.ascontiguousarray(normalized_adj)


@numba.jit(
    nopython=True
)  # option mode "fastmath=True" can be swith on if data are clean enough
def coupling_func(State, eps, Norm_Adj):
    """
    Applies diffusive coupling to the State.

    State: (N_nodes, Dimension)
    Norm_Adj: Pre-computed (N_nodes, N_nodes) matrix
    """
    interaction = (
        Norm_Adj @ State
    )  # For sparse or huge network (more than 1000 nodes with only few neighbors by node use an explicit for loop with numba)
    State_new = (1.0 - eps) * State + eps * interaction
    return State_new


@numba.jit(nopython=True)
def evolve_system(
    State_0, N_steps, params, model_step_func, coupling_func, coupling_op, eps
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
    Trajectory = np.zeros((N_steps, n_nodes, dim))
    Trajectory[0, :, :] = State_0
    # .copy() here ensures Current_State is contiguous in memory to start with.
    Current_State = Trajectory[0, :, :].copy()
    Current_State = coupling_func(Current_State, eps, coupling_op)
    for t in range(1, N_steps):
        Next_State = model_step_func(Current_State, params)
        Trajectory[t, :, :] = Next_State
        Current_State = coupling_func(Next_State, eps, coupling_op)
    return Trajectory


@numba.jit(nopython=True)
def fhn_derivatives(State, params):
    """
    Computes the derivative dS/dt for a system of oscillators.
    State shape: (N_nodes, 2) -> col 0 is v (voltage), col 1 is w (recovery)
    Params: [a, b, tau, I_ext, dt] (Note: dt is not used here, but passed in params)
    """
    v = State[:, 0]
    w = State[:, 1]
    a = params[0]
    b = params[1]
    tau = params[2]
    I_ext = params[3]
    dState = np.empty_like(State)
    # dv/dt = v - v^3/3 - w + I
    dState[:, 0] = v - (v**3 / 3.0) - w + I_ext
    # dw/dt = (v + a - b*w) / tau
    dState[:, 1] = (v + a - b * w) / tau
    return dState


@numba.jit(nopython=True)
def step_fhn_rk4(State, params):
    """
    Performs one RK4 step.
    Matches signature: (State, params) -> New_State
    """
    dt = params[4]  # We assume dt is stored in the params array
    # RK4 Integration Logic
    # k1 = f(y)
    k1 = fhn_derivatives(State, params)
    # k2 = f(y + 0.5*dt*k1)
    k2 = fhn_derivatives(State + 0.5 * dt * k1, params)
    # k3 = f(y + 0.5*dt*k2)
    k3 = fhn_derivatives(State + 0.5 * dt * k2, params)
    # k4 = f(y + dt*k3)
    k4 = fhn_derivatives(State + dt * k3, params)
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
def transfo_coupling_vec(X, Y, Eps, Adjacence):
    m, n = len(X), len(Y)
    assert m == n, "missmatch dimension"
    weights = np.sum(Adjacence, axis=1)
    if (weights == 0).any():
        raise ValueError(
            "some nodes have no edges/neighbors/connections with other nodes"
        )  # comme ça pas de problème de type sur la sortie
    transition = Eps * Adjacence * 1 / weights[:, np.newaxis] + np.identity(m)
    X_new = np.dot(transition, X) - Eps * X
    Y_new = np.dot(transition, Y) - Eps * Y
    return X_new, Y_new


@numba.jit(nopython=True)
def evolution_vec(X_0, Y_0, N, list_ab, Eps, Adjacence):
    """
    Warning: slicing on rows gives contiguous array while on columns it is not.

    Return the evolution with respect to time of the nodes.
    """
    n, m = len(X_0), len(Y_0)
    p = len(Adjacence)
    assert n == m, "wrong dimensions, both initial conditions don't have the same size"
    assert p == n, "missmatch dimensions"
    X = np.zeros((N, n))
    Y = np.zeros((N, n))
    UN = np.ones(p)
    X[0, :] = X_0
    Y[0, :] = Y_0
    X_c, Y_c = transfo_coupling_vec(X[0, :], Y[0, :], Eps, Adjacence)
    for t in range(1, N):
        X[t, :] = -list_ab[0, :] * X_c**2 + UN + Y_c
        Y[t, :] = list_ab[1, :] * X_c
        # This slice (a row) is contiguous by default!
        X_c, Y_c = transfo_coupling_vec(X[t, :], Y[t, :], Eps, Adjacence)
    return X, Y
