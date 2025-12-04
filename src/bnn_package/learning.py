#!/usr/bin/env/python3
import numpy as np
import numba
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
import cma
from bnn_package import evolution
from bnn_package import buildgraph


@numba.jit(nopython=True)
def run_reservoir_epoch(
    State_0, input_signal, input_weights, params, N_p, Coupling_op, C_r, type_diff
):
    steps = input_signal.shape[0]
    n_nodes = State_0.shape[0]
    dt = params[5]

    states_history = np.zeros((steps, n_nodes), dtype=np.float64)
    current_state = State_0.copy()

    for t in range(steps):
        u_t = np.ascontiguousarray(input_signal[t])
        I_current = input_weights @ u_t

        k1 = evolution.fhn_derivatives(
            current_state, params, N_p, Coupling_op, C_r, type_diff, I_current
        )
        k2 = evolution.fhn_derivatives(
            current_state + 0.5 * dt * k1,
            params,
            N_p,
            Coupling_op,
            C_r,
            type_diff,
            I_current,
        )
        k3 = evolution.fhn_derivatives(
            current_state + 0.5 * dt * k2,
            params,
            N_p,
            Coupling_op,
            C_r,
            type_diff,
            I_current,
        )
        k4 = evolution.fhn_derivatives(
            current_state + dt * k3, params, N_p, Coupling_op, C_r, type_diff, I_current
        )

        current_state = current_state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        states_history[t] = current_state[:, 0]

    return states_history


# ==============================================================================
# PART 2: THE RESERVOIR
# ==============================================================================


class FhnReservoir:
    def __init__(self, n_nodes, weighted_adjacency, fhn_params):
        self.n_nodes = n_nodes
        self.A = weighted_adjacency
        self.params = fhn_params
        # Important: We recalculate the Coupling Operator with the NEW weights
        self.coupling_op = evolution.get_coupling_operator(self.A, "Diffusive")
        self.N_p = np.zeros(n_nodes)
        self.C_r = 0.0
        self.readout = Ridge(alpha=1e-5)
        self.input_weights = None

    def fit(self, X_signals, Y_labels, input_connectivity=0.3):
        input_dim = X_signals[0].shape[1]
        rng_in = np.random.default_rng(42)

        if self.input_weights is None:
            self.input_weights = rng_in.normal(0, 5.0, (self.n_nodes, input_dim))
            mask = rng_in.random((self.n_nodes, input_dim)) > input_connectivity
            self.input_weights[mask] = 0.0

        all_features = []
        for signal in X_signals:
            state_0 = np.zeros((self.n_nodes, 3))
            states = run_reservoir_epoch(
                state_0,
                signal,
                self.input_weights,
                self.params,
                self.N_p,
                self.coupling_op,
                self.C_r,
                "Diffusive",
            )
            features = np.std(states[50:], axis=0)
            all_features.append(features)

        X_train = np.array(all_features)
        self.readout.fit(X_train, Y_labels)
        return self.readout.score(X_train, Y_labels)

    def predict(self, X_signals):
        all_features = []
        for signal in X_signals:
            state_0 = np.zeros((self.n_nodes, 3))
            states = run_reservoir_epoch(
                state_0,
                signal,
                self.input_weights,
                self.params,
                self.N_p,
                self.coupling_op,
                self.C_r,
                "Diffusive",
            )
            features = np.std(states[50:], axis=0)
            all_features.append(features)
        return self.readout.predict(np.array(all_features))


# ==============================================================================
# PART 3: THE SYNAPTIC EVOLUTION LOOP
# ==============================================================================


def objective_function_weights(
    genome, base_adjacency, edge_indices, X_train, y_train, X_val, y_val, static_params
):
    """
    Decodes GENOME -> WEIGHTS -> ADJACENCY MATRIX
    """
    N = static_params["n_nodes"]

    # 1. Map Genome (Weights) back to the Matrix
    # We clone the base mask to keep the structure
    Weighted_Adj = np.zeros_like(base_adjacency, dtype=np.float64)

    # Apply weights to the specific edges that exist
    # We take absolute value because negative adjacency is usually invalid in standard FhN diffusion
    weights = np.abs(genome)

    # Assign weights to the upper triangle edges
    rows, cols = edge_indices
    Weighted_Adj[rows, cols] = weights

    # Symmetrize (assuming undirected graph)
    Weighted_Adj = Weighted_Adj + Weighted_Adj.T

    # 2. Build Reservoir with this Weighted Matrix
    res = FhnReservoir(N, Weighted_Adj, static_params["fhn_params"])

    try:
        # Train Readout
        res.fit(X_train, y_train)
        # Validate
        preds = res.predict(X_val)

        preds_class = (preds > 0.5).astype(int)
        acc = accuracy_score(y_val, preds_class)

        # Optional: Add L1 penalty to encourage sparsity (pruning non-essential edges)
        # loss = -accuracy + lambda * sum(|weights|)
        sparsity_penalty = 0.001 * np.mean(weights)

        return -(acc - sparsity_penalty)  # We minimize this

    except ValueError:
        return 10.0


def run_evolutionary_optimization():
    # --- 1. SETUP DATA ---
    print(">>> Generating Synthetic Data...")
    n_samples = 60
    time_steps = 300
    X_data = []
    y_data = []
    rng = np.random.default_rng(42)

    for i in range(n_samples):
        t = np.linspace(0, 20, time_steps)
        if i % 2 == 0:
            sig = np.sin(t) * 2.0
            target = 0
        else:
            sig = rng.normal(0, 1.0, time_steps)
            target = 1
        X_data.append(sig.reshape(-1, 1))
        y_data.append(target)

    split = int(0.8 * n_samples)
    X_train, X_val = X_data[:split], X_data[split:]
    y_train, y_val = y_data[:split], y_data[split:]

    # --- 2. INITIALIZE BASE TOPOLOGY ---
    N_NODES = 20
    print(f">>> Initializing Base Graph for {N_NODES} nodes...")

    # We use your buildgraph to create the "Skeleton"
    # We place nodes randomly and connect them
    pos = buildgraph.pos_nodes_uniform(N_NODES, 10.0, 10.0, rng)
    # std=3.0 means reasonable connectivity
    Base_Adj = buildgraph.connexion_normal_deterministic(pos, rng, std=3.0)

    # Identify the edges we can train (Upper Triangle only to enforce symmetry later)
    # We only train edges that ALREADY EXIST (Base_Adj == 1)
    rows, cols = np.triu_indices(N_NODES, k=1)
    existing_edges_mask = Base_Adj[rows, cols] > 0

    trainable_rows = rows[existing_edges_mask]
    trainable_cols = cols[existing_edges_mask]
    edge_indices = (trainable_rows, trainable_cols)

    num_trainable_edges = len(trainable_rows)
    print(f"    Base Graph created. Found {num_trainable_edges} trainable edges.")

    # --- 3. CMA-ES SETUP ---
    # Genome: One float value per trainable edge
    initial_genome = np.ones(num_trainable_edges)  # Start with weight 1.0

    fhn_params = np.array([1.0, 0.1, 0.05, 1.0, 0.0, 0.1], dtype=np.float64)
    static_context = {"n_nodes": N_NODES, "fhn_params": fhn_params}

    print("\n>>> Starting Synaptic Weight Evolution")
    es = cma.CMAEvolutionStrategy(initial_genome, 0.5, {"popsize": 12, "maxiter": 15})

    while not es.stop():
        solutions = es.ask()
        fitnesses = [
            objective_function_weights(
                s,
                Base_Adj,
                edge_indices,
                X_train,
                y_train,
                X_val,
                y_val,
                static_context,
            )
            for s in solutions
        ]
        es.tell(solutions, fitnesses)

        best_score = -es.result.fbest
        print(f"Gen {es.result.iterations}: Best Score = {best_score:.4f}")

    print("\n>>> Evolution Complete.")

    # Extract Best Graph
    if es.result.xbest is not None:
        best_weights = np.abs(es.result.xbest)
        Final_Adj = np.zeros_like(Base_Adj, dtype=np.float64)
        Final_Adj[edge_indices] = best_weights
        Final_Adj = Final_Adj + Final_Adj.T

        print(
            "Optimization finished. You can now analyze which weights went to 0 (pruned)."
        )
    else:
        print("Warning: No valid solution found during optimization.")


if __name__ == "__main__":
    run_evolutionary_optimization()
