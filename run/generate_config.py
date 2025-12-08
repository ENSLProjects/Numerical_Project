#!/usr/bin/env/python3
import yaml
import os
import uuid
import numpy as np
import networkx as nx
from datetime import datetime

from bnn_package import (
    pos_nodes_uniform,
    connexion_normal_deterministic,
    add_passive_nodes,
    print_simulation_report,
)


def build_and_save_graph(config, registry_dir):
    """
    Builds a graph based on config parameters and saves it to a central registry.
    Returns the absolute path to the .npz file.
    """
    print("\n>>> GENERATING NEW GRAPH FOR CONFIG...")

    # 1. Setup Parameters
    graph_uuid = str(uuid.uuid4())[:8]
    rng = np.random.default_rng(config.get("seed", 123456789))
    N = config["number_of_nodes"]

    square = config["square_for_graph"]
    if isinstance(square, (list, tuple)):
        xmax = float(square[0])
        ymax = float(square[1])
    elif isinstance(square, (int, float, np.number)):
        val = float(square)
        xmax = val
        ymax = val
    else:
        try:
            val = float(square)
            xmax = val
            ymax = val
        except (ValueError, TypeError):
            raise ValueError(f"Invalid type for 'square_for_graph': {type(square)}")

    # 2. Build Topology
    pos = pos_nodes_uniform(N, xmax, ymax, rng)
    Adjacency = connexion_normal_deterministic(pos, rng, config["std"])
    print("\n ---------> Full graph generated")
    print("\n")
    print_simulation_report(Adjacency, fast_mode=False)

    # 3. Add Passive Nodes
    G = nx.from_numpy_array(Adjacency)
    nx.set_node_attributes(G, {n: "active" for n in G.nodes()}, name="type")
    _, N_p = add_passive_nodes(G, config["mean_poisson"], rng)

    # 4. Save to Registry
    os.makedirs(registry_dir, exist_ok=True)
    filename = f"graph_N{N}_{graph_uuid}.npz"
    filepath = os.path.join(registry_dir, filename)

    np.savez(
        filepath,
        adjacency=Adjacency,
        positions=pos,
        passive_counts=N_p,
        n_nodes=N,
        uuid=graph_uuid,
    )
    print(f"\nGraph Created: {filepath}")
    print("\n")
    return filepath


def create_experiment_config(experiment_name, **kwargs):
    # 1. Base Template
    template = {
        "mode": "sweep",
        "output_file": f"results_{experiment_name}.csv",
        "parallel": True,
        "cores_ratio": 0.8,
        # Graph Params
        "number_of_nodes": 1000,
        "square_for_graph": [10.0, 10.0],
        "diffusive_operator": "Laplacian",
        "std": 1.0,
        "mean_poisson": 3,
        # Physics Params
        "total_time": 300000,
        "transitory_time": 10000,
        "a": 3.0,
        "alpha": 0.2,
        "k": 0.25,
        "vrp": 1.5,
        "dt": 0.01,
        # Sweeps
        "epsilon": 0.1,
        "cr": 1.0,
        "metrics": ["sync_error"],
        # IMPORTANT: This key starts empty or None
        "existing_graph_path": None,
    }

    # 2. Merge User Arguments into Template
    for key, value in kwargs.items():
        template[key] = value

    # 3. SMART LOGIC: Check if we need to build a graph
    # If the user didn't provide a path, we build one NOW.
    if not template.get("existing_graph_path"):
        registry_dir = "Data_output/graphs_registry"
        new_graph_path = build_and_save_graph(template, registry_dir)

        # INJECT THE PATH INTO THE CONFIG
        template["existing_graph_path"] = new_graph_path

    # 4. Save YAML
    os.makedirs("run/configs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"run/configs/{timestamp}_{experiment_name}.yaml"

    with open(filename, "w") as f:
        yaml.dump(template, f, sort_keys=False, default_flow_style=None)

    print(f">>> Config Generated: {filename}")
    print(f">>> Linked Graph:     {template['existing_graph_path']}")
    return filename


if __name__ == "__main__":
    create_experiment_config(
        "debugging",
        mode="time_series",
        epsilon=0.08,
        cr=1.5,
    )
