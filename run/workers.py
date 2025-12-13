#!/usr/bin/env python3

# ======================= Libraries


import numpy as np
import os
from numpy.random import default_rng

from bnn_package.evolution import (
    get_coupling_operator,
    evolve_system,
    rk4_step,
    FitzHughNagumoModel,
)
from bnn_package.data_processing import save_simulation_data
from bnn_package.measure import AVAILABLE_METRICS


# ======================= Helper Functions =======================


def load_graph_topology(path):
    """Loads the graph data safely."""
    data = np.load(path)
    # Handle older graph files that might not have uuid
    graph_uuid = str(data["uuid"]) if "uuid" in data else "unknown"
    return (data["adjacency"], data["positions"], data["passive_counts"], graph_uuid)


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
    coupling_str = float(params.get("epsilon", 0.1))
    fhn_time_scale = float(params.get("fhn_eps", 0.01))
    model = FitzHughNagumoModel(
        coupling_op=coupling_op,
        alpha=float(params.get("alpha", 0.5)),
        eps=fhn_time_scale,  # Internal Physics (mapped from fhn_eps)
        k=float(params.get("k", 1.0)),
        vrp=float(params.get("vrp", 0.0)),
        coupling_str=coupling_str,  # Network Coupling (mapped from epsilon)
        dt=float(params.get("dt", 0.01)),
        type_diff_code=type_diff_code,
        cr=float(params.get("cr", 1.0)),
        np_vec=N_p.astype(np.float64),  # Passive node counts vector
    )

    return model


def common_worker_setup(params):
    """Shared setup logic."""
    graph_path = params["graph_file_path"]
    adjacency, pos, N_p, graph_uuid = load_graph_topology(graph_path)
    model = create_model(params, adjacency, N_p)
    return model, graph_uuid


def time_series(params):
    """
    Runs a full simulation and saves the trajectory.
    """
    # 1. Setup
    model, graph_uuid = common_worker_setup(params)
    n_nodes = params["number_of_nodes"]

    # Total steps (Time / dt)
    total_time_steps = int(params.get("total_time", 1000))

    # 2. Random State Init (SoA Layout: 3 x N)
    rng = default_rng(params.get("seed", None))

    State_0 = np.zeros((3, n_nodes), dtype=np.float64)
    State_0[0] = 0.1 + 0.1 * rng.standard_normal(n_nodes)  # Voltage
    State_0[1] = 0.3 + 0.1 * rng.standard_normal(n_nodes)  # Recovery
    State_0[2] = 1.0 + 0.1 * rng.standard_normal(n_nodes)  # Passive

    # 3. Run (Injection: Passing rk4_step)
    full_data = evolve_system(
        model, State_0, total_time_steps, input_signal=None, stepper_func=rk4_step
    )

    # 4. Save Logic
    output_folder = params.get("output_folder", "Data_output")
    run_id = params.get("run_id", "manual")

    # Filename using Coupling Constant (which is model.coupling_str)
    coupling_val = model.coupling_str
    filename = f"ts_N{n_nodes}_Coup{coupling_val:.3f}_G-{graph_uuid}.h5"
    save_path = os.path.join(output_folder, filename)

    # Metadata
    params_dict = params.copy()
    params_dict["graph_uuid"] = graph_uuid
    params_dict["run_uuid"] = run_id
    # Record what we actually ran
    params_dict["actual_fhn_eps"] = model.eps
    params_dict["actual_coupling_str"] = model.coupling_str

    transitory = int(params.get("transitory_time", total_time_steps * 0.1))

    save_simulation_data(save_path, full_data[transitory:], params_dict, "")

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
        "fhn_time_scale": model.eps,  # Explicitly log the physics parameter
        "cr": model.c_r,
        "graph_uuid": graph_uuid,
    }

    metrics = params.get("metrics", ["sync_error"])
    for m in metrics:
        if m in AVAILABLE_METRICS:
            # Metrics must handle (Time, Nodes) input
            results[m] = AVAILABLE_METRICS[m](voltage_data)

    return results
