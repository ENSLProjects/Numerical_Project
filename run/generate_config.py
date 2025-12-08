#!/usr/bin/env/python3

# ======================= Libraries


import yaml
import os
import uuid
import numpy as np
import networkx as nx

from bnn_package import (
    pos_nodes_uniform,
    connexion_normal_deterministic,
    add_passive_nodes,
    print_simulation_report,
)


# ======================= Functions


def build_and_save_graph(config, registry_dir):
    """Build a graph and save it as .npz in a central folder

    Args:
        config (dic): dictionnary of meaningful parameters to build the graph from the .yaml file.
        registry_dir (str): relative path of the folder to save up the graph.

    Raises:
        ValueError: if ever the rectangle parameters have unvalid type

    Returns:
        str: file path where the graph is saved with the graph passive and active information.
    """
    print("\n>>> GENERATING NEW GRAPH FOR CONFIG...")

    # 1. Setup Parameters
    graph_uuid = str(uuid.uuid4())[:8]
    rng = np.random.default_rng(config.get("seed", 123456789))
    n_nodes = config["number_of_nodes"]

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
    pos = pos_nodes_uniform(n_nodes, xmax, ymax, rng)
    adjacency = connexion_normal_deterministic(pos, rng, config["std"])
    print("\n ---------> Full graph generated")
    print("\n")
    print_simulation_report(adjacency, fast_mode=False)

    # 3. Add Passive Nodes
    G = nx.from_numpy_array(adjacency)
    nx.set_node_attributes(G, {n: "active" for n in G.nodes()}, name="type")
    _, N_p = add_passive_nodes(G, config["mean_poisson"], rng)

    # 4. Save to Registry
    os.makedirs(registry_dir, exist_ok=True)
    filename = f"graph_N{n_nodes}_{graph_uuid}.npz"
    filepath = os.path.join(registry_dir, filename)

    np.savez(
        filepath,
        adjacency=adjacency,
        positions=pos,
        passive_counts=N_p,
        n_nodes=n_nodes,
        uuid=graph_uuid,
    )
    print(f"\nGraph Created: {filepath}")
    print("\n")
    return filepath


def create_experiment_config(experiment_name, **kwargs):
    """Create the .yaml config file from the template and store it

    Args:
        experiment_name (str): name of the set of parameters

    Returns:
        str: return the filename as str
    """
    # 1. Base Template
    template = {
        # GENERAL INFORMATION
        "mode": "sweep",
        "output_file": f"results_{experiment_name}.csv",
        "parallel": True,
        "cores_ratio": 0.8,
        "output_folder": "Data_output",
        # GRAPH PARAMETERS
        "number_of_nodes": 1000,
        "square_for_graph": [10.0, 10.0],
        "diffusive_operator": "Laplacian",
        "std": 1.0,
        "mean_poisson": 3,
        # PHYSICS PARAMETERS
        "total_time": 300000,
        "transitory_time": 10000,
        "a": 3.0,
        "alpha": 0.2,
        "k": 0.25,
        "vrp": 1.5,
        "dt": 0.01,
        # SWEEPT PARAMETERS
        "epsilon": 0.1,
        "cr": 1.0,
        "metrics": ["sync_error"],
        # IMPORTANT: This key starts empty or None
        "existing_graph_path": None,
    }

    # 2. Merge User Arguments into Template
    for key, value in kwargs.items():
        template[key] = value

    # 3. Check if we need to build a graph
    if not template.get("existing_graph_path"):
        registry_dir = "Data_output/graphs_registry"
        new_graph_path = build_and_save_graph(template, registry_dir)
        template["existing_graph_path"] = new_graph_path

    # 4. Save YAML
    os.makedirs("run/configs", exist_ok=True)
    filename = f"run/configs/{experiment_name}.yaml"
    with open(filename, "w") as f:
        yaml.dump(template, f, sort_keys=False, default_flow_style=None)

    print(f">>> Config Generated: {filename}")
    print(f">>> Linked Graph:     {template['existing_graph_path']}")
    return filename


if __name__ == "__main__":
    create_experiment_config(
        "debugging",
        mode="time_series",
        epsilon=[0.08, 0.1],
        cr=1.5,
    )
