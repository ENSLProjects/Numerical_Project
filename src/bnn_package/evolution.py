#!/usr/bin/env python3

# ======================= Libraries


from typing import List, Tuple, Any
import numpy as np
import numba
from numba import float64, int32
from numba.experimental import jitclass


# ======================= Helper Functions =======================


def get_coupling_operator(adjacency: np.ndarray, type_diff: str) -> np.ndarray:
    """
    Pre-computes the normalized adjacency matrix.
    Run this ONCE in pure Python before the simulation.
    """
    if type_diff == "Diffusive":
        weights = np.sum(adjacency, axis=1)
        if np.any(weights == 0):
            raise ValueError("Nodes with degree 0 detected.")
        coupling_op = adjacency / weights[:, np.newaxis]
    elif type_diff == "Laplacian":
        degrees = np.sum(adjacency, axis=1)
        deg_matrix = np.diag(degrees)
        coupling_op = deg_matrix - adjacency
    else:
        raise ValueError(f"Unknown coupling method: {type_diff}")

    # Critical for performance: Ensure contiguous memory layout
    return np.ascontiguousarray(coupling_op)


# ======================= 1. MODEL DEFINITIONS =======================

# --- FitzHugh-Nagumo Model ---

fhn_spec: List[Tuple[str, Any]] = [
    ("coupling_op", float64[:, ::1]),  # The matrix (N, N)
    ("a", float64),  # Parameter A (Scalar)
    ("alpha", float64),
    ("fhn_eps", float64),  # Epsilon
    ("k", float64),  # K (Passive)
    ("v_rp", float64),  # Resting Potential
    ("coupling_str", float64),  # Sigma (Coupling strength)
    ("dt", float64),  # Time step
    ("type_diff", int32),  # 0=Diffusive, 1=Laplacian
    ("c_r", float64),  # Resistive Coupling (Scalar)
    ("n_p", float64[:]),  # Passive Node Count (Vector)
]


@jitclass(fhn_spec)  # type: ignore
class FitzHughNagumoModel:
    def __init__(
        self,
        coupling_op,
        alpha,
        a,
        fhn_eps,
        k,
        vrp,
        coupling_str,
        dt,
        type_diff_code,
        cr,
        np_vec,
    ):
        self.coupling_op = coupling_op
        self.alpha = alpha
        self.a = a
        self.fhn_eps = fhn_eps
        self.k = k
        self.v_rp = vrp
        self.coupling_str = coupling_str
        self.dt = dt
        self.type_diff = type_diff_code
        self.c_r = cr
        self.n_p = np_vec

    def derivatives(self, state, t, current_input):
        """
        Computes dState/dt.
        State shape: (3, N) -> [v_active, w, v_passive]
        """
        v = state[0]
        w = state[1]
        v_p = state[2]

        interaction = self.coupling_op @ v

        if self.type_diff == 1:  # Laplacian
            coupling_term = self.coupling_str * interaction
        else:  # Diffusive
            coupling_term = (
                1.0 - self.coupling_str
            ) * v + self.coupling_str * interaction

        passive_interaction = self.c_r * (v_p - v)

        # dv/dt
        dv = (
            self.a * (v * (v - self.alpha) * (1.0 - v))  # Local Dynamics
            - w  # Recovery
            + coupling_term  # Network Diffusion
            + current_input  # External Input
            + (passive_interaction * self.n_p)  # Feedback from passive nodes
        )

        # dw/dt
        dw = self.fhn_eps * (v - w)

        # dv_p/dt
        dv_p = self.k * (self.v_rp - v_p) - passive_interaction

        d_state = np.empty_like(state)
        d_state[0] = dv
        d_state[1] = dw
        d_state[2] = dv_p

        return d_state


# --- Henon Map Model ---

henon_spec: List[Tuple[str, Any]] = [
    ("coupling_op", float64[:, ::1]),
    ("a", float64),
    ("b", float64),
    ("coupling_str", float64),
    ("dt", float64),  # Dummy for compatibility
    ("type_diff", int32),
]


@jitclass(henon_spec)  # type: ignore
class HenonMapModel:
    def __init__(self, coupling_op, a, b, coupling_str, type_diff_code):
        self.coupling_op = coupling_op
        self.a = a
        self.b = b
        self.coupling_str = coupling_str
        self.dt = 1.0
        self.type_diff = type_diff_code

    def compute_next_step(self, state, current_input):
        """
        Computes x(t+1), y(t+1).
        State shape: (2, N)
        """
        x = state[0]
        y = state[1]

        interaction = self.coupling_op @ x

        if self.type_diff == 1:
            x_coupled = x + self.coupling_str * interaction
        else:
            x_coupled = (1.0 - self.coupling_str) * x + self.coupling_str * interaction

        # Add Input
        x_coupled += current_input

        # Map Dynamics
        x_next = 1.0 - self.a * (x_coupled**2) + y
        y_next = self.b * x_coupled

        next_state = np.empty_like(state)
        next_state[0] = x_next
        next_state[1] = y_next

        return next_state


# ======================= 2. SOLVER ENGINES =======================


@numba.jit(nopython=True, nogil=True)
def rk4_step(model, state, t, current_input):
    """Runge-Kutta 4 Integrator for ODEs."""
    dt = model.dt
    k1 = model.derivatives(state, t, current_input)
    k2 = model.derivatives(state + 0.5 * dt * k1, t + 0.5 * dt, current_input)
    k3 = model.derivatives(state + 0.5 * dt * k2, t + 0.5 * dt, current_input)
    k4 = model.derivatives(state + dt * k3, t + dt, current_input)

    return state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


@numba.jit(nopython=True, nogil=True)
def euler_step(model, state, t, current_input):
    """Euler Integrator (Faster, less accurate)."""
    dt = model.dt
    d_state = model.derivatives(state, t, current_input)
    return state + dt * d_state


@numba.jit(nopython=True, nogil=True)
def map_step(model, state, t, current_input):
    """Wrapper for Discrete Maps (Henon)."""
    return model.compute_next_step(state, current_input)


# ======================= 3. SIMULATION MANAGER =======================


@numba.jit(nopython=True, nogil=True)
def evolve_system(model, state_0, final_time, input_signal, stepper_func):
    """
    Generic Evolution Function.

    Args:
        model: Physics Object (FHN, Henon...)
        state_0: Initial state (Dim, N_nodes)
        final_time: Total steps (int)
        input_signal: (Time, Nodes) or None
        stepper_func: Function to use (rk4_step, map_step...)
    """
    dim = state_0.shape[0]
    n_nodes = state_0.shape[1]

    trajectory = np.zeros((final_time, dim, n_nodes), dtype=np.float64)
    trajectory[0] = state_0

    current_state = state_0.copy()
    has_input = input_signal is not None

    for t in range(final_time - 1):
        u_in = input_signal[t] if has_input else 0.0

        current_state = stepper_func(model, current_state, t * model.dt, u_in)

        trajectory[t + 1] = current_state

    return trajectory
