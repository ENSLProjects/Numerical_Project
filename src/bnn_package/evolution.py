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
def coupling_func(State, eps, Adj, Model):
    """
    Applies diffusive coupling to the State.

    State: (N_nodes, Dimension)
    Model : type of coupling : 'Henon' or 'FN'
    Norm_Adj: Pre-computed (N_nodes, N_nodes) matrix
    """

    if Model == "Henon":
        Adj = get_coupling_operator(Adj)
        interaction = (
            Adj @ State
        )  # For sparse or huge network (more than 1000 nodes with only few neighbors by node use an explicit for loop with numba)
        State_new = (1.0 - eps) * State + eps * interaction
    if Model == "FN":
        degrees = np.sum(Adj, axis=1)
        Deg = np.diag(degrees)
        L = Deg - Adj
        State_new = np.zeros(State.shape)
        State_new[:, 0] = L @ State[:, 0]
        State_new[:, 1] = State[:, 1]
        State_new[:, 2] = State[:, 2]
    return State_new


@numba.jit(nopython=True)
def evolve_system(State_0, N_steps, params, model_step_func, eps, N_p, Adj, C_r, D):
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
    Trajectory[0] = State_0.copy()

    Current_State = State_0.copy()

    for t in range(1, N_steps):
        # Assure la contiguïté mémoire
        Current_State = np.ascontiguousarray(Current_State)
        Next_State = model_step_func(Current_State, params, N_p, Adj, C_r, D)
        Next_State = np.ascontiguousarray(Next_State)
        Trajectory[t] = Next_State
        Current_State = Next_State

    return Trajectory


@numba.jit(nopython=True)
def fhn_derivatives(State, params, N_p, Adj, C_r, D):
    """
    Computes the derivative dS/dt for a system of oscillators.
    State shape: (N_nodes, 3) -> col 0 is v_e (voltage of active node), col 1 is g (recovery), col 2 is v_p (voltage of passive node)
    Params: bibliothèque (Note: dt is not used here, but passed in params)
    A : adjacency matrix (N*N)
    N_p : diagonal matrix -> number of passive node for each active node
    """
    v_e = State[:, 0]
    g = State[:, 1]
    v_p = State[:, 2]

    K = params[3]
    A = params[0]
    Eps = params[2]
    alpha = params[1]
    V_RP = params[4]

    dState = np.empty_like(State)
    # Termes de couplage
    coupling_term = coupling_func(State, 0, Adj, "FN")

    # dv_e/dt = A*v_e*(v_e-alpha)*(1-v_e) - g + N_p*C_r*(v_p-v_e) - D*coupling_term[:,0]
    dState[:, 0] = (
        A * v_e * (v_e - alpha) * (1 - v_e)
        - g
        + (N_p @ (C_r * (v_p - v_e)))
        - D * coupling_term[:, 0]
    )

    # dg/dt = Eps(v_e - g)
    dState[:, 1] = Eps * (v_e - g)

    # dv_p/dt = K(V_RP - v_p) - C_r(v_p - v_e)
    dState[:, 2] = K * (V_RP - v_p) - C_r * (v_p - v_e)
    return dState


@numba.jit(nopython=True)
def step_fhn_rk4(State, params, N_p, Adj, C_r, D):
    """
    Performs one RK4 step.
    Matches signature: (State, params) -> New_State
    """
    dt = params[5]  # We assume dt is stored in the params array
    # RK4 Integration Logic
    # k1 = f(y)
    k1 = fhn_derivatives(State, params, N_p, Adj, C_r, D)
    # k2 = f(y + 0.5*dt*k1)
    k2 = fhn_derivatives(State + 0.5 * dt * k1, params, N_p, Adj, C_r, D)
    # k3 = f(y + 0.5*dt*k2)
    k3 = fhn_derivatives(State + 0.5 * dt * k2, params, N_p, Adj, C_r, D)
    # k4 = f(y + dt*k3)
    k4 = fhn_derivatives(State + dt * k3, params, N_p, Adj, C_r, D)
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
