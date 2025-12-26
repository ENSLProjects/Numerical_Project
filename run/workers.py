#!/usr/bin/env python3

# ======================= Libraries


import numpy as np
import os
from numpy.random import default_rng
from scipy.linalg import expm
from bnn_package import (
    get_coupling_operator,
    evolve_system,
    rk4_step,
    prepare_data,
    compute_te_over_lags,
    FitzHughNagumoModel,
    save_simulation_data,
    AVAILABLE_METRICS_ORDER_PARAMETER,
    RESEARCH_METRICS,
)
import random
import networkx as nx


# ======================= Helper Functions =======================


def load_graph_topology(path):
    """Loads the graph data safely."""
    data = np.load(path)
    # Handle older graph files that might not have uuid
    graph_uuid = str(data["uuid"]) if "uuid" in data else "unknown"
    return (data["adjacency"], data["positions"], data["passive_counts"], graph_uuid)


def get_stable_dt(params, adjacency):
    """
    Heuristic to determine a stable dt for FitzHugh-Nagumo.
    Accounts for stiffness (fhn_eps) and network coupling (spectral radius).
    """
    # 1. Local Dynamics: Stability is limited by the fast/slow timescale ratio
    # If fhn_eps is very small, the recovery variable 'w' is much slower than 'v'
    fhn_eps = float(params.get("fhn_eps", 0.08))
    tau_local = 1.0 / fhn_eps

    # 2. Network Dynamics: Stability is limited by the coupling strength and graph
    # For Laplacian/Diffusive operators, we estimate the spectral radius
    coupling_str = float(params.get("epsilon", 0.01))
    diff_type = params.get("diffusive_operator", "Diffusive")

    if diff_type == "Laplacian":
        # Gershgorin circle theorem upper bound for Laplacian: 2 * max_degree
        degrees = np.sum(adjacency, axis=1)
        spec_radius = 2.0 * np.max(degrees)
    else:
        # For diffusive/row-normalized matrices, the max eigenvalue is ~1
        spec_radius = 1.0

    tau_net = 1.0 / (coupling_str * spec_radius)

    # 3. Apply safety margin (RK4 typically requires dt < 10% of fastest timescale)
    # We take the minimum of local, network, and a baseline safety value (0.05)
    suggested_dt = min(tau_local, tau_net, 1.0) * 0.05

    # Ensure it's not too small (performance) or too large (instability)
    return max(min(suggested_dt, 0.01), 0.001)


def create_model(params, adjacency, N_p):
    """
    Instantiates the JIT-compiled Physics Model with CORRECT parameter mapping.
    """
    # 1. Determine Coupling Operator Type
    diff_type = params.get("diffusive_operator", "Diffusive")
    type_diff_code = 1 if diff_type == "Laplacian" else 0
    coupling_op = np.ascontiguousarray(
        get_coupling_operator(adjacency, diff_type), dtype=np.float64
    )
    coupling_str = float(params.get("epsilon", 0.01))
    fhn_eps = float(params.get("fhn_eps", 0.08))
    alpha = float(params.get("alpha", 0.2))
    k = float(params.get("k", 0.25))
    dt = get_stable_dt(params, adjacency)
    cr = float(params.get("cr", 1.0))
    a = float(params.get("a", 3.0))
    vrp = float(params.get("vrp", 1.5))
    model = FitzHughNagumoModel(
        coupling_op=coupling_op,
        alpha=alpha,
        fhn_eps=fhn_eps,  # Internal Physics (mapped from fhn_eps)
        k=k,
        vrp=vrp,
        coupling_str=coupling_str,  # Network Coupling (mapped from epsilon)
        dt=dt,
        type_diff_code=type_diff_code,
        cr=cr,
        np_vec=N_p.astype(np.float64),  # Passive node counts vector
        a=a,
    )

    return model


def common_worker_setup(params):
    """Shared setup logic."""
    graph_path = params["graph_file_path"]
    adjacency, pos, N_p, graph_uuid = load_graph_topology(graph_path)
    model = create_model(params, adjacency, N_p)
    return model, graph_uuid


def get_stratified_pairs(adjacency, config):
    """
    Selects pairs based on topological distance to bypass N^2 cost.
    """
    G = nx.from_numpy_array(adjacency)
    all_pairs = {}

    # 1. Direct Neighbors (Dist 1)
    edges = list(G.edges())
    n_d1 = config.get("n_dist1", 1000)
    # Sample if we have more edges than requested
    if len(edges) > n_d1:
        all_pairs["dist1"] = random.sample(edges, n_d1)
    else:
        all_pairs["dist1"] = edges

    # 2. Dist 2 and Dist 3+ (BFS sampling)
    nodes = list(G.nodes())
    n_d2 = config.get("n_dist2", 1000)
    n_d3 = config.get("n_dist3", 1000)

    d2_found, d3_found = [], []
    attempts = 0
    max_attempts = (n_d2 + n_d3) * 100  # Safety break

    while (len(d2_found) < n_d2 or len(d3_found) < n_d3) and attempts < max_attempts:
        u, v = random.sample(nodes, 2)
        try:
            d = nx.shortest_path_length(G, source=u, target=v)
            if d == 2 and len(d2_found) < n_d2:
                d2_found.append((u, v))
            elif d >= 3 and len(d3_found) < n_d3:
                d3_found.append((u, v))
        except nx.NetworkXNoPath:
            pass
        attempts += 1

    all_pairs["dist2"] = d2_found
    all_pairs["dist3"] = d3_found

    return all_pairs


# ======================= Simulation Functions =======================


def time_series(params):
    """
    Runs a full simulation and saves the trajectory.
    """
    # 1. Setup
    model, graph_uuid = common_worker_setup(params)
    n_nodes = params["number_of_nodes"]

    # Total steps (Time / dt)
    total_time_steps = int(params.get("total_time", 1000))

    # 2. Random State Init
    rng = default_rng(params.get("seed", None))

    State_0 = np.zeros((3, n_nodes), dtype=np.float64)
    State_0[0] = 0.1 + 0.1 * rng.standard_normal(n_nodes)
    State_0[1] = 0.3 + 0.1 * rng.standard_normal(n_nodes)
    State_0[2] = 1.0 + 0.1 * rng.standard_normal(n_nodes)

    # 3. Run
    full_data = evolve_system(
        model, State_0, total_time_steps, input_signal=None, stepper_func=rk4_step
    )

    # 4. Save Logic
    output_folder = params.get("output_folder", "Data_output")

    # Filename using Coupling Constant
    coupling_val = model.coupling_str
    cr_val = model.c_r
    filename = f"ts_N{n_nodes}_Coup{coupling_val:.3f}_cr{cr_val:.3f}_G-{graph_uuid}.h5"
    save_path = os.path.join(output_folder, filename)

    transitory = int(params.get("transitory_time", total_time_steps * 0.1))

    with np.errstate(over="ignore"):
        data_to_save = full_data[transitory:].astype(np.float32)

    save_simulation_data(save_path, data_to_save, graph_uuid)

    return {}


def run_order_parameter(params):
    """
    Runs simulation and returns scalar metrics.
    """
    # 1. Setup
    model, graph_uuid = common_worker_setup(params)
    n_nodes = params["number_of_nodes"]
    total_time_steps = int(params["total_time"])

    rng = default_rng(params.get("seed", None))

    # State Init
    State_0 = np.zeros((3, n_nodes), dtype=np.float64)
    State_0[0] = 0.1 + 0.1 * rng.standard_normal(n_nodes)

    # 2. Run
    traj = evolve_system(
        model, State_0, total_time_steps, input_signal=None, stepper_func=rk4_step
    )

    # 3. Measure
    start = int(params["transitory_time"])

    # Extract Voltage (Index 0) for metrics
    # Shape becomes (Time_Subset, N_Nodes)
    voltage_data = traj[start:, 0, :]

    results = {
        # We save 'epsilon' as the coupling strength because that matches your config file keys
        "epsilon": model.coupling_str,
        "fhn_eps": model.fhn_eps,  # Explicitly log the physics parameter
        "cr": model.c_r,
        "graph_uuid": graph_uuid,
    }

    metrics = params.get("metrics", ["sync_error"])
    for m in metrics:
        if m in AVAILABLE_METRICS_ORDER_PARAMETER:
            # Metrics must handle (Time, Nodes) input
            results[m] = AVAILABLE_METRICS_ORDER_PARAMETER[m](voltage_data)

    return results


def research_alignment_worker(params):
    """
    Simulates FHN, loops over LAGS, computes Stratified TE vs Theory,
    and returns flat research metrics. (RAM Optimized)
    """
    # 1. Setup
    model, graph_uuid = common_worker_setup(params)

    # We strictly need adjacency for sampling and theory
    adj, _, _, _ = load_graph_topology(params["graph_file_path"])

    # 2. Simulate (In-Memory)
    total_time = int(params["total_time"])
    rng = default_rng(params.get("seed", None))
    n_nodes = params["number_of_nodes"]

    State_0 = np.zeros((3, n_nodes), dtype=np.float64)
    State_0[0] = 0.1 + 0.1 * rng.standard_normal(n_nodes)
    State_0[1] = 0.3 + 0.1 * rng.standard_normal(n_nodes)
    State_0[2] = 1.0 + 0.1 * rng.standard_normal(n_nodes)

    traj = evolve_system(model, State_0, total_time, None, rk4_step)

    # Extract Voltage (Post-Transitory)
    start = int(params.get("transitory_time", 1000))
    voltage_data = traj[start:, 0, :]

    results = {
        "epsilon": model.coupling_str,
        "cr": model.c_r,
        "fhn_eps": model.fhn_eps,
        "graph_uuid": graph_uuid,
    }

    if not np.isfinite(voltage_data).all():
        print(
            f"!!! CRASH DETECTED: Simulation exploded for eps={model.coupling_str}, cr={model.c_r}"
        )
        # Return a 'failed' result so the runner doesn't crash
        return {
            "epsilon": model.coupling_str,
            "cr": model.c_r,
            "error": "Simulation exploded (NaN/Inf values), try to look at rk4 dt",
        }

    # Also check for empty/silent data (std dev approx 0)
    var = np.std(voltage_data)
    if var < 1e-9:
        print(
            f">>> SKIPPING TE: Fixed point detected (Std={var:.2e}) for cr={model.c_r}"
        )

        # Fill results with dummy zeros to keep the CSV valid
        analysis_cfg = params.get("research_analysis", {})
        lags = analysis_cfg.get("te_lags", [1])
        metrics = analysis_cfg.get("metrics", ["kl_divergence"])

        for tau in lags:
            results[f"te_uncertainty_lag{tau}"] = 0.0
            for m in metrics:
                results[f"{m}_lag{tau}"] = (
                    0.0  # KL 0 means perfect alignment (trivial), or just 0 info
                )

        return results

    # 3. Analyze
    analysis_cfg = params.get("research_analysis", {})
    if not analysis_cfg.get("active", False):
        return {"error": "Research analysis inactive in config"}

    # A. Get Pairs
    pairs_dict = get_stratified_pairs(adj, analysis_cfg["stratified_sampling"])

    # B. Loop over Lags (Flattening dimensions)
    lags_to_test = analysis_cfg.get("te_lags", [1])

    n_real = analysis_cfg.get("n_real", 1)

    for tau in lags_to_test:
        L_exp = expm(-model.coupling_op * model.dt * tau)

        measured_means = []
        measured_stds = []
        theory_vals = []

        for group, pairs in pairs_dict.items():
            for u, v in pairs:
                x = prepare_data(voltage_data[:, u])
                y = prepare_data(voltage_data[:, v])

                # Pass n_real to the optimized measure function
                val_means, val_stds = compute_te_over_lags(
                    x,
                    y,
                    [tau],
                    n_real=n_real,
                    n_eff=analysis_cfg["n_eff"],
                    kNN=analysis_cfg["kNN"],
                    verbose=False,
                )

                measured_means.append(val_means[0])
                measured_stds.append(val_stds[0])
                theory_vals.append(L_exp[u, v])

        # Convert to arrays and Compute Metrics
        vec_means = np.array(measured_means)
        vec_stds = np.array(measured_stds)
        vec_theory = np.array(theory_vals)

        # Save uncertainty metric (Good for error bars in plots later)
        results[f"te_uncertainty_lag{tau}"] = np.mean(vec_stds)

        # Compute Metrics using the ROBUST MEAN
        requested_metrics = analysis_cfg.get("metrics", ["kl_divergence"])
        for metric_name in requested_metrics:
            if metric_name in RESEARCH_METRICS:
                score = RESEARCH_METRICS[metric_name](vec_means, vec_theory)
                results[f"{metric_name}_lag{tau}"] = score

    return results
