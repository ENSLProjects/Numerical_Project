#!/usr/bin/env/python3
import numpy as np
import networkx as nx
import os
import time
from numpy.random import default_rng

from bnn_package import (
    pos_nodes_uniform,
    connexion_normal_deterministic,
    get_coupling_operator,
    evolve_system,
    step_fhn_rk4,
    add_passive_nodes,
    get_simulation_path,
    save_simulation_data,
)
from bnn_package.measure import AVAILABLE_METRICS

# ==============================================================================
# 1. LIGHTWEIGHT WORKER (Phase Scans)
# ==============================================================================


def run_order_parameter(params):
    """Standardized Worker: Accepts ONE dictionary 'params'.
    Extracts seed and metrics from inside that dictionary.

    Args:
        params (dic): dic from the .yaml config file

    Returns:
        csv: csv of the output, value of the order parameter for the set of parameters params
    """
    # 1. Unpack Setup
    # Extract seed from params (default if missing)
    seed = params.get("seed", 123456789)
    rng = default_rng(seed)

    # 2. Extract Physics
    N_nodes = params["Number of Nodes"]
    (xmax, ymax) = params["Square for the graph"]
    Diffusion = params["Diffusive Operator"]

    p_solver = np.array(
        [
            params["A"],
            params["Alpha"],
            params["Epsilon"],
            params["K"],
            params["Vrp"],
            params["dt"],
        ],
        dtype=np.float64,
    )

    # 3. Build Graph
    pos = pos_nodes_uniform(N_nodes, xmax, ymax, rng)
    Adjacency = connexion_normal_deterministic(pos, rng, params["Std"])
    DiffusionOp = get_coupling_operator(Adjacency, Diffusion)

    # Passive Nodes
    G = nx.from_numpy_array(Adjacency)
    nx.set_node_attributes(G, {n: "active" for n in G.nodes()}, name="type")
    _, N_p = add_passive_nodes(G, params["Mean Poisson"], rng)

    # 4. Run Evolution
    traj = evolve_system(
        State_0=np.zeros((N_nodes, 3)),
        N_steps=params["Total time"],
        params=p_solver,
        model_step_func=step_fhn_rk4,
        N_p=N_p,
        Coupling_op=DiffusionOp,
        C_r=params["Cr"],
        type_diff=Diffusion,
    )

    # 5. Measure Metrics
    start = params["transient"]
    X = traj[start::10, :, 0]
    Y = traj[start::10, :, 1]

    # Initialize results with the sweep variables
    results = {"epsilon": params["Epsilon"], "Cr": params["Cr"]}

    # Extract requested metrics from YAML (default to 'sync_error' if missing)
    # The runner no longer passes 'order_parameter_name' as an argument,
    # so we look for it in the config dict.
    requested_metrics = params.get("metrics", ["sync_error"])

    for metric_name in requested_metrics:
        if metric_name not in AVAILABLE_METRICS:
            # Fallback or Skip
            continue

        metric_func = AVAILABLE_METRICS[metric_name]
        values = []

        for t in range(X.shape[0]):
            val = metric_func(X, Y, N_nodes, t)
            values.append(val)

        results[metric_name] = np.mean(values)

    return results


# ==============================================================================
# 2. HEAVY WORKER (Time Series)
# ==============================================================================


def time_series(params):
    """
    Standardized Worker: Accepts ONE dictionary 'params'.
    """
    print(f">>> Running Time Series for Eps={params.get('Epsilon')}")

    # 1. Setup
    seed = params.get("seed", None)
    rng = default_rng(seed)
    N_nodes = params["Number of nodes"]

    # Graph Params
    (xmax, ymax) = params["Square for the graph"]
    std = params["Std"]
    f = params["Mean Poisson"]

    # Physics Params
    A = params["A"]
    alpha = params["Alpha"]
    Eps = params["Epsilon"]
    K = params["K"]
    Vrp = params["Vrp"]
    dt = params["dt"]
    C_r = params["Cr"]

    transitoire = params.get("transient", 0)
    N_time = params["Total time"]
    Diffusion_mode = params["Diffusive Operator"]

    # 2. Init State & Graph
    State_0 = np.zeros((N_nodes, 3))
    State_0[:, 0] = 0.1 + 0.1 * rng.standard_normal(N_nodes)
    State_0[:, 1] = 0.3 + 0.1 * rng.standard_normal(N_nodes)
    State_0[:, 2] = 1.0 + 0.1 * rng.standard_normal(N_nodes)

    pos = pos_nodes_uniform(N_nodes, xmax, ymax, rng)
    Adjacency = connexion_normal_deterministic(pos, rng, std)
    DiffusionOp = get_coupling_operator(Adjacency, Diffusion_mode)

    G = nx.from_numpy_array(Adjacency)
    nx.set_node_attributes(G, {n: "active" for n in G.nodes()}, name="type")
    PasG, N_p = add_passive_nodes(G, f, rng)

    p_vec = np.array([A, alpha, Eps, K, Vrp, dt], dtype=np.float64)

    # 3. Evolution
    t_start = time.time()
    FullData = evolve_system(
        State_0, N_time, p_vec, step_fhn_rk4, N_p, DiffusionOp, C_r, "Laplacian"
    )
    print(f"    Done in {time.time() - t_start:.2f}s")

    # 4. Save
    MY_FOLDER = "data_simulation"
    GRAPH_FOLDER = os.path.join(MY_FOLDER, "graph")
    os.makedirs(GRAPH_FOLDER, exist_ok=True)

    params_dict = {
        "number of nodes": N_nodes,
        "std graph": std,
        "epsilon": Eps,
        "time length simulation": N_time,
        "model": "FhN",
        "how to diffuse": "Laplacian",
        "parameters_model": {
            "A": A,
            "alpha": alpha,
            "coupling_active": Eps,
            "K": K,
            "Vrp": Vrp,
            "coupling_passive": C_r,
            "time_step rk4": dt,
            "average Poisson": f,
        },
    }

    save_path = get_simulation_path(MY_FOLDER, "FhN", params_dict)
    graph_filename = f"graph_FhN_{N_nodes}_{Eps}_Poisson{f}.graphml"

    nx.write_graphml(PasG, os.path.join(GRAPH_FOLDER, graph_filename))

    Datacuted = FullData[transitoire:, :, :]
    save_simulation_data(
        save_path, Datacuted, params_dict, os.path.join(GRAPH_FOLDER, graph_filename)
    )

    return {}  # Empty dict to satisfy map
