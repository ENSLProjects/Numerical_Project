#!/usr/bin/env python3

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
    """
    Generates a graph, saves it to .npz, and returns the path.
    This guarantees that every simulation config points to a REAL, existing graph.
    """
    print("\n>>> GENERATING NEW GRAPH FOR CONFIG...")

    # 1. Setup Parameters
    rng = np.random.default_rng(config.get("seed", None))
    n_nodes = config["number_of_nodes"]

    square = config["square_for_graph"]
    if isinstance(square, (list, tuple)):
        xmax, ymax = float(square[0]), float(square[1])
    else:
        xmax = ymax = float(square)

    # 2. Build Topology (Nodes + Edges)
    pos = pos_nodes_uniform(n_nodes, xmax, ymax, rng)
    adjacency = connexion_normal_deterministic(pos, rng, config["std"])

    print("\n ---------> Graph Generated")
    which_analysis = config.get("quick_analyze_graph", True)
    print_simulation_report(
        adjacency, fast_mode=which_analysis
    )  # Fast mode for cleaner logs

    # 3. Add Passive Nodes
    G = nx.from_numpy_array(adjacency)
    nx.set_node_attributes(G, {n: "active" for n in G.nodes()}, name="type")
    _, N_p = add_passive_nodes(G, config["mean_poisson"], rng)

    # 4. Save to Registry (Provenance)
    os.makedirs(registry_dir, exist_ok=True)

    graph_uuid = str(uuid.uuid4())[:8]
    filename = f"graph_N{n_nodes}_std{config['std']}_{graph_uuid}.npz"
    filepath = os.path.join(registry_dir, filename)

    np.savez(
        filepath,
        adjacency=adjacency,
        positions=pos,
        passive_counts=N_p,
        n_nodes=n_nodes,
        uuid=graph_uuid,
    )

    print(f"\n[SAVED] Graph Registry: {filepath}")
    return filepath


def create_experiment_config(experiment_name, **kwargs):
    """
    Creates a standardized .yaml config file for the runner.
    """
    # 1. Base Template (Physics & Execution)
    template = {
        # --- EXECUTION ---
        "mode": "sweep",  # "sweep", "time_series", or "research_alignment"
        "output_file": f"results_{experiment_name}.csv",
        "parallel": True,
        "cores_ratio": 0.8,
        "output_folder": "Data_output",
        "seed": 1234567890,
        "quick_analyze_graph": False,
        # --- GRAPH ARCHITECTURE ---
        "number_of_nodes": 1000,
        "square_for_graph": [10.0, 10.0],
        "diffusive_operator": "Laplacian",
        "std": 1.0,
        "mean_poisson": 3,
        # --- PHYSICS (FHN Model) ---
        "total_time": 300000,
        "transitory_time": 10000,
        "dt": 0.01,
        "alpha": 0.2,
        "a": 3.0,
        "k": 0.25,
        "vrp": 1.5,
        "fhn_eps": 0.08,
        # --- SWEEP PARAMETERS ---
        "epsilon": 0.1,
        "metrics": ["sync_error"],
        "cr": 1.0,
        # --- RESEARCH ANALYSIS ---
        # This block is only utilized if mode == "research_alignment"
        "research_analysis": {
            "active": False,
            "te_lags": [1, 5, 10],
            "stratified_sampling": {"n_dist1": 1000, "n_dist2": 1000, "n_dist3": 1000},
            "kNN": 5,
            "n_eff": 4096,
        },
        # --- PROVENANCE ---
        "existing_graph_path": None,
    }

    # 2. Override with User Arguments
    for key, value in kwargs.items():
        # Handle nested research_analysis updates if provided as a dict
        if key == "research_analysis" and isinstance(value, dict):
            template["research_analysis"].update(value)
        else:
            template[key] = value

    # 3. Graph Provenance Check
    if not template.get("existing_graph_path"):
        registry_dir = "Data_output/graphs_registry"
        new_graph_path = build_and_save_graph(template, registry_dir)
        template["existing_graph_path"] = new_graph_path

    # 4. Save YAML
    config_dir = "run/configs"
    os.makedirs(config_dir, exist_ok=True)
    filename = os.path.join(config_dir, f"{experiment_name}.yaml")

    with open(filename, "w") as f:
        yaml.dump(template, f, sort_keys=False, default_flow_style=None)

    print(f"\n>>> CONFIGURATION READY: {filename}")
    print(f"    Linked Graph:      {template['existing_graph_path']}")
    print(f"    Mode:              {template['mode']}")

    return filename


if __name__ == "__main__":
    create_experiment_config(
        "scan_eps",
        mode="research_alignment",
        quick_analyze_graph=False,
        parallel=True,
        cores_ratio=0.5,
        cr=0.5,
        total_time=100000,
        transitory_time=1000,
        mean_poisson=0.7,
        epsilon=[0.011, 0.012, 0.0125, 0.013],
        std=0.9,
        research_analysis={
            "active": True,
            # Minimal lags to find the minimum quickly
            "te_lags": [1, 5, 10],
            # Fast/Coarse settings
            "n_real": 10,
            "n_eff": 4096,
            "kNN": 3,
            # Light sampling (200 pairs per distance)
            "stratified_sampling": {"n_dist1": 200, "n_dist2": 200, "n_dist3": 200},
            # We only need KL to find the "Goldilocks Zone"
            "metrics": ["kl_divergence", "cca_alignment"],
        },
        existing_graph_path="Data_output/graphs_registry/graph_N1000_std0.9_c267b0d5.npz",
    )
