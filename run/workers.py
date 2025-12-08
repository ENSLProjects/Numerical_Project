#!/usr/bin/env/python3

# ======================= Libraries


import numpy as np
import os

# import time
from numpy.random import default_rng

from bnn_package import (
    get_coupling_operator,
    evolve_system,
    step_fhn_rk4,
    save_simulation_data,
)
from bnn_package.measure import AVAILABLE_METRICS


# ======================= Functions


def load_graph_topology(path):
    """Loads the standard graph format saved by runner.py

    Args:
        path (str): relative path where to dowloand the topoly.

    Returns:
        tupple: the adjacency, nodes positions and number of passive nodes by active nodes data
    """
    data = np.load(path)
    # Extract Graph ID if available
    graph_uuid = str(data["uuid"]) if "uuid" in data else "unknown"
    return (data["adjacency"], data["positions"], data["passive_counts"], graph_uuid)


def common_worker_setup(params):
    """Shared setup logic for both workers.

    Args:
        params (dic): setup of parameters from the .yaml file.

    Returns:
        tupple: returns the array of the FhN parameters, the diffusion operator (Combinatory or Normalized Laplacian), the vector of passive nodes and the uuid of the graph.
    """
    # 1. Load Graph
    graph_path = params["graph_file_path"]
    adjacency, pos, N_p, graph_uuid = load_graph_topology(graph_path)

    # 2. Physics Operators
    diffusion = params["diffusive_operator"]
    diffusionOp = get_coupling_operator(adjacency, diffusion).astype(np.float64)

    # 3. Physics Vector
    p_vec = np.array(
        [
            params["a"],
            params["alpha"],
            params["epsilon"],
            params["k"],
            params["vrp"],
            params["dt"],
        ],
        dtype=np.float64,
    )

    return p_vec, diffusionOp, N_p, graph_uuid, diffusion


def time_series(params):
    """_summary_

    Args:
        params (_type_): _description_

    Returns:
        _type_: _description_
    """

    # 1. Setup from Graph File
    p_vec, diffusionOp, N_p, graph_uuid, diffusion = common_worker_setup(params)
    n_nodes = params["number_of_nodes"]
    full_time = params.get("total_time")

    # 2. Random State Init
    rng = default_rng(params.get("seed", None))
    State_0 = np.zeros((n_nodes, 3))
    State_0[:, 0] = 0.1 + 0.1 * rng.standard_normal(n_nodes)
    State_0[:, 1] = 0.3 + 0.1 * rng.standard_normal(n_nodes)
    State_0[:, 2] = 1.0 + 0.1 * rng.standard_normal(n_nodes)

    # 3. Run
    FullData = evolve_system(
        State_0,
        full_time,
        p_vec,
        step_fhn_rk4,
        N_p,
        diffusionOp,
        params["cr"],
        diffusion,
    )

    # 4. Save
    MY_FOLDER = params.get("output_folder", "Data_output")

    # This allows you to find this specific file later easily
    run_id = params.get("run_id", "manual")
    filename = f"ts_N{n_nodes}_eps{params['epsilon']:.3f}_G-{graph_uuid}.h5"
    save_path = os.path.join(MY_FOLDER, filename)

    # Embed IDs into Metadata
    params_dict = params.copy()
    params_dict["graph_uuid"] = graph_uuid
    params_dict["run_uuid"] = run_id

    transit_inter = full_time * 0.1
    transitoire = params.get("transitory_time", transit_inter)
    save_simulation_data(save_path, FullData[transitoire:], params_dict, "")

    return {}


def run_order_parameter(params):
    """_summary_

    Args:
        params (_type_): _description_

    Returns:
        _type_: _description_
    """
    # 1. Setup
    p_vec, DiffusionOp, N_p, graph_uuid, diffusion = common_worker_setup(params)
    N_nodes = params["number_of_nodes"]

    rng = default_rng(params.get("seed", None))
    State_0 = np.zeros((N_nodes, 3))
    State_0[:, 0] = 0.1 + 0.1 * rng.standard_normal(N_nodes)

    # 2. Run, better to run again the simulation than loading really heavy ones
    traj = evolve_system(
        State_0,
        params["total_time"],
        p_vec,
        step_fhn_rk4,
        N_p,
        DiffusionOp,
        params["cr"],
        diffusion,
    )

    # 3. Measure HAVE TO BE ADAPT TO WORK IN A MORE GENERAL CONTEXT
    start = params["transitory_time"]
    X = traj[start::10, :, 0]
    Y = traj[start::10, :, 1]

    results = {"epsilon": params["epsilon"], "cr": params["cr"]}
    results["graph_uuid"] = graph_uuid  # Save which graph was used

    metrics = params.get("metrics", ["sync_error"])
    for m in metrics:
        if m in AVAILABLE_METRICS:
            val = [AVAILABLE_METRICS[m](X, Y, N_nodes, t) for t in range(X.shape[0])]
            results[m] = np.mean(val)

    return results
