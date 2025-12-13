#!/usr/bin/env/python3
import numpy as np
import numba
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
import cma
import h5py

from bnn_package import evolution
from bnn_package import buildgraph


# ======================= PART 1: SOLVER (Unchanged) =======================


@numba.jit(nopython=True)
def run_reservoir_epoch(
    State_0, input_signal, input_weights, params, N_p, Coupling_op, C_r, type_diff
):
    steps = input_signal.shape[0]
    n_nodes = State_0.shape[0]
    dt = params[5]
    states_history = np.zeros((steps, n_nodes), dtype=np.float64)
    current = State_0.copy()

    for t in range(steps):
        # I_injected = W_in @ u(t)
        I_curr = input_weights @ np.ascontiguousarray(input_signal[t])

        k1 = evolution.fhn_derivatives(
            current, params, N_p, Coupling_op, C_r, type_diff, I_curr
        )
        k2 = evolution.fhn_derivatives(
            current + 0.5 * dt * k1, params, N_p, Coupling_op, C_r, type_diff, I_curr
        )
        k3 = evolution.fhn_derivatives(
            current + 0.5 * dt * k2, params, N_p, Coupling_op, C_r, type_diff, I_curr
        )
        k4 = evolution.fhn_derivatives(
            current + dt * k3, params, N_p, Coupling_op, C_r, type_diff, I_curr
        )

        current += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        states_history[t] = current[:, 0]

    return states_history


# ======================= PART 2: RESERVOIR WITH I/O MASKS =======================
class FhnReservoir:
    def __init__(
        self, n_nodes, weighted_adjacency, fhn_params, input_nodes, readout_nodes
    ):
        self.n_nodes = n_nodes
        self.A = weighted_adjacency
        self.params = fhn_params
        self.coupling_op = evolution.get_coupling_operator(self.A, "Diffusive")
        self.N_p = np.zeros(n_nodes)
        self.C_r = 0.0
        self.readout = Ridge(alpha=1e-3)  # Stronger regularization

        # DEFINING THE GAP
        self.input_node_indices = input_nodes  # e.g., [0,1,2]
        self.readout_node_indices = readout_nodes  # e.g., [17,18,19]

        self.input_weights = None

    def fit(self, X_signals, Y_labels):
        input_dim = X_signals[0].shape[1]
        rng = np.random.default_rng(42)

        # 1. Input Weights: ONLY project to 'input_node_indices'
        if self.input_weights is None:
            self.input_weights = np.zeros((self.n_nodes, input_dim))
            # Random weights only for input zone
            active_weights = rng.normal(
                0, 10.0, (len(self.input_node_indices), input_dim)
            )
            self.input_weights[self.input_node_indices] = active_weights

        all_features = []
        for sig in X_signals:
            state_0 = np.zeros((self.n_nodes, 3))
            states = run_reservoir_epoch(
                state_0,
                sig,
                self.input_weights,
                self.params,
                self.N_p,
                self.coupling_op,
                self.C_r,
                "Diffusive",
            )

            # 2. Readout: ONLY see 'readout_node_indices'
            # We ignore the activity of the input nodes!
            # The signal MUST travel through the graph to be seen here.
            readout_activity = states[50:, self.readout_node_indices]

            # Feature: Standard Deviation of the Output Zone
            features = np.std(readout_activity, axis=0)
            all_features.append(features)

        X_train = np.array(all_features)
        self.readout.fit(X_train, Y_labels)
        return self.readout.score(X_train, Y_labels)

    def predict(self, X_signals):
        all_features = []
        for sig in X_signals:
            state_0 = np.zeros((self.n_nodes, 3))
            states = run_reservoir_epoch(
                state_0,
                sig,
                self.input_weights,
                self.params,
                self.N_p,
                self.coupling_op,
                self.C_r,
                "Diffusive",
            )
            readout_activity = states[50:, self.readout_node_indices]
            features = np.std(readout_activity, axis=0)
            all_features.append(features)
        return self.readout.predict(np.array(all_features))


# ======================= PART 3: VISUALIZATION & OBJECTIVE =======================
def plot_synaptic_plasticity(history, filename="evolution_trajectories.png"):
    history = np.abs(np.array(history))
    plt.figure(figsize=(10, 6))
    plt.plot(history, color="blue", alpha=0.15, linewidth=1)
    plt.plot(
        np.mean(history, axis=1), color="red", linewidth=2, linestyle="--", label="Mean"
    )
    plt.title("Synaptic Plasticity: The Bridge Builders")
    plt.xlabel("Generation")
    plt.ylabel("Weight Strength")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.savefig(filename, dpi=150)
    plt.close()


def plot_weight_distribution(history, filename="evolution_heatmap.png"):
    history = np.abs(np.array(history))
    bins = np.linspace(0, np.max(history) * 1.1, 50)
    density = np.array([np.histogram(row, bins, density=True)[0] for row in history]).T
    plt.figure(figsize=(10, 6))
    plt.imshow(
        density,
        aspect="auto",
        origin="lower",
        cmap="magma",
        extent=(0, len(history), bins[0], bins[-1]),
    )
    plt.colorbar(label="Density")
    plt.title("Evolution of Weight Distribution")
    plt.xlabel("Generation")
    plt.ylabel("Weight")
    plt.savefig(filename, dpi=150)
    plt.close()


def objective_function(genome, base_adj, edge_idx, X_train, y_train, X_val, y_val, ctx):
    # Decode
    W_Adj = np.zeros_like(base_adj, dtype=np.float64)
    weights = np.abs(genome)
    W_Adj[edge_idx] = weights
    W_Adj += W_Adj.T

    # Build Reservoir with GAP constraints
    res = FhnReservoir(
        ctx["n_nodes"],
        W_Adj,
        ctx["fhn_params"],
        input_nodes=ctx["in_nodes"],
        readout_nodes=ctx["out_nodes"],
    )
    try:
        res.fit(X_train, y_train)
        preds = res.predict(X_val)
        acc = accuracy_score(y_val, (preds > 0.5).astype(int))

        # Penalize disconnection (Average weight shouldn't be 0)
        # But we REMOVE the sparsity penalty to encourage bridges
        return -acc
    except Exception:
        return 10.0


# ======================= PART 4: MAIN =======================
def logistic_map_series(r, steps, x0):
    """Generates chaotic time series"""
    x = np.zeros(steps)
    x[0] = x0
    for i in range(steps - 1):
        x[i + 1] = r * x[i] * (1 - x[i])
    return x


def run_evolutionary_optimization():
    # --- 1. DATA (Logistic Map - Same as before) ---
    print(">>> Generating HARD Data (Logistic Map)...")
    n_samples = 80
    steps = 400
    X_data, y_data = [], []
    for i in range(n_samples):
        r = 3.7 if i % 2 == 0 else 3.9
        target = 0 if i % 2 == 0 else 1
        sig = logistic_map_series(r, steps, np.random.rand())
        sig = (sig - 0.5) * 5.0
        X_data.append(sig.reshape(-1, 1))
        y_data.append(target)
    split = int(0.8 * n_samples)
    X_train, X_val = X_data[:split], X_data[split:]
    y_train, y_val = y_data[:split], y_data[split:]

    # --- 2. THE CANYON TOPOLOGY ---
    N_NODES = 25
    rng = np.random.default_rng()

    print(">>> Building The Canyon (Gap between In and Out)...")

    # Custom Positioning to force separation
    # Node 0-2 (Input)  -> Left Side (x=0)
    # Node 3-21 (Hidden)-> Random Middle (x=2 to 8)
    # Node 22-24 (Read) -> Right Side (x=10)
    pos = np.zeros((2, N_NODES))

    # Inputs at X=0, Spread along Y
    pos[0, :3] = 0.0
    pos[1, :3] = rng.uniform(0, 10, 3)

    # Readouts at X=10, Spread along Y
    pos[0, -3:] = 10.0
    pos[1, -3:] = rng.uniform(0, 10, 3)

    # Hidden Layer in the "Canyon"
    pos[0, 3:-3] = rng.uniform(2.0, 8.0, N_NODES - 6)  # X between 2 and 8
    pos[1, 3:-3] = rng.uniform(0, 10, N_NODES - 6)  # Y random

    # Create Connectivity based on these positions
    # sigma=2.5 ensures Input (x=0) CANNOT reach Output (x=10) directly.
    # It must hop at least 3-4 times.
    Base_Adj = buildgraph.connexion_normal_deterministic(pos, rng, std=2.3)

    # Define Zones
    in_nodes = np.array([0, 1, 2])
    out_nodes = np.array([N_NODES - 3, N_NODES - 2, N_NODES - 1])

    # Verify no direct connection exists (Safety Check)
    if np.any(Base_Adj[in_nodes][:, out_nodes]):
        print("WARNING: Direct leak detected! Reducing sigma...")
        Base_Adj = buildgraph.connexion_normal_deterministic(pos, rng, std=1.5)

    rows, cols = np.triu_indices(N_NODES, k=1)
    mask = Base_Adj[rows, cols] > 0
    edge_idx = (rows[mask], cols[mask])
    n_edges = len(rows[mask])

    print(f"    Nodes: {N_NODES}")
    print(f"    Trainable Edges: {n_edges}")
    print(f"    Input Nodes: {in_nodes}")
    print(f"    Output Nodes: {out_nodes}")

    # Context
    # eps=0.01: Weak coupling requires strong edges to propagate signal
    fhn_params = np.array([1.0, 0.1, 0.01, 1.0, 0.0, 0.1])
    ctx = {
        "n_nodes": N_NODES,
        "fhn_params": fhn_params,
        "in_nodes": in_nodes,
        "out_nodes": out_nodes,
    }

    # --- 3. EVOLUTION (Start from Darkness) ---
    # Initialize weights to effectively ZERO.
    # The bridge is strictly broken. The readout should see NOTHING (Acc ~50%).
    # Evolution must 'turn on' the lights path by path.
    initial_genome = np.ones(n_edges) * 1e-6

    weight_hist = []
    # sigma0=0.5 allows the weights to jump up quickly if needed
    es = cma.CMAEvolutionStrategy(initial_genome, 0.5, {"popsize": 24, "maxiter": 50})

    print("\n>>> Starting Evolution (Expect ~50% accuracy at first)...")
    while not es.stop():
        sols = es.ask()
        # ABSOLUTE VALUE wrapper to ensure non-negative weights during trial
        fits = [
            objective_function(
                np.abs(s), Base_Adj, edge_idx, X_train, y_train, X_val, y_val, ctx
            )
            for s in sols
        ]
        es.tell(sols, fits)

        best_acc = -es.result.fbest
        weight_hist.append(es.result.xbest)
        print(f"Gen {es.result.iterations}: Best Acc = {best_acc:.2%}")

        # Stop early if solved (to save time)
        if best_acc > 0.999:
            print("Solved!")
            break

    print(">>> Evolution Complete. Generating Plots...")
    plot_synaptic_plasticity(weight_hist)
    plot_weight_distribution(weight_hist)

    # --- 5. SAVE THE BRAIN ---

    # Reconstruct the Final Adjacency Matrix
    Final_Adj = np.zeros_like(Base_Adj, dtype=np.float64)
    best_weights = (
        np.abs(es.result.xbest) if es.result.xbest is not None else np.zeros(n_edges)
    )

    # Apply weights
    rows, cols = edge_idx
    Final_Adj[rows, cols] = best_weights
    Final_Adj = Final_Adj + Final_Adj.T

    # Save to HDF5
    filename = "data_simulation/Evolved_Canyon_Brain.h5"
    with h5py.File(filename, "w") as f:
        f.create_dataset("adjacency", data=Final_Adj)
        f.create_dataset("positions", data=pos)  # Save positions to plot them later
        f.attrs["accuracy"] = best_acc
        f.attrs["generations"] = es.result.iterations

    print(f"\n>>> [SAVED] Brain stored in {filename}")


if __name__ == "__main__":
    run_evolutionary_optimization()
