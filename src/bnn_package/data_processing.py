#!/usr/bin/env/python3

# ======================= Libraries

import os
import h5py
from typing import cast
import numpy as np
import networkx as nx
import json
from datetime import datetime
from pathlib import Path

# ======================= Functions


def prepare_data(arr):
    """
    Prépare les données pour la lib C Entropy.

    La sortie respecte ces règles :

    Règle 1 : Format (1, N_samples) obligatoire (donc 1 ligne, N colonnes).
    Règle 2 : Mémoire contiguë (C-contiguous).
    Règle 3 : Type float64 (double).
    """
    # Si c'est un vecteur plat (N,), on le passe en (1, N)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    # Si c'est (N, 1), on transpose en (1, N)
    elif arr.shape[0] > arr.shape[1]:
        arr = arr.T

    return np.ascontiguousarray(arr, dtype=np.float64)


def get_data(container, key):
    """
    Helper to safely extract a dataset and cast it for the linter.
    """
    return cast(h5py.Dataset, container[key])[:]


def corrupted_simulation(file_path):
    """
    Verifies that the tensor of data has no NaN or Inf data.

    Args:
        file_path (str): Relative path to the simulation file.
    """
    filename = os.path.basename(file_path)

    if not os.path.exists(file_path):
        print(f"CRITICAL ERROR: File not found at {file_path}")
        return True

    with h5py.File(file_path, "r") as f:
        print(f"--- File found: {filename} ---")

        try:
            trajectory = get_data(f, "trajectory")
        except KeyError:
            print("Error: 'trajectory' dataset not found at file root.")
            return True

        print("\n" + "=" * 30)
        print("   NUMERICAL DIAGNOSTIC")
        print("=" * 30)

        has_nan = np.isnan(trajectory).any()
        has_inf = np.isinf(trajectory).any()
        max_val = np.max(trajectory)
        min_val = np.min(trajectory)

        print(f"Data Shape:    {trajectory.shape}" + "(Time, Node, Physical variable)")
        print(f"Contains NaNs? {has_nan}")
        print(f"Contains Infs? {has_inf}")

        if has_nan or has_inf:
            print(
                "\n[CONCLUSION] ❌ The simulation EXPLODED. Try to change the Runge-Kutta time step."
            )
            return True
        elif max_val == 0 and min_val == 0:
            print("\n[CONCLUSION] ⚠️ The simulation is FLAT (All Zeros).")
            return True
        else:
            print("\n[CONCLUSION] ✅ Data looks valid.")
            return False


def json_numpy_serializer(obj):
    """Automatically converts numpy types to standard python types.
    Make Numpy JSON-serializable.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Type {type(obj)} not serializable")


def save_simulation_data(file_path, trajectory, parameters, graph_path):
    """
    Saves heavy data to HDF5, embeds parameters as JSON, and links the graph file.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with h5py.File(file_path, "w") as f:
        f.create_dataset(
            "trajectory",
            data=trajectory.astype(np.float32),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )

        param_str = json.dumps(parameters, default=json_numpy_serializer)
        f.attrs["parameters"] = param_str
        f.attrs["linked_graph_path"] = graph_path

    print(f"[Saved] Data: {file_path}")
    print(f"[Linked] Graph: {graph_path}")


def load_simulation_data(file_path, graph: bool):
    """Loads trajectory, parameters, and the linked graph if wanted.

    Args:
        file_path (str): relative path to the file
        graph (bool): True if you want to load the graph as a networkx graph.

    Raises:
        FileNotFoundError: the path doen't exist

    Returns:
        _type_: return the full data of the simulation.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found.")
    data = {}
    with h5py.File(file_path, "r") as f:
        # 1. Trajectory
        data["trajectory"] = get_data(f, "trajectory")

        # 2. Parameters (JSON String -> Dict)
        param_str = f.attrs["parameters_json"]
        # Fix for HDF5 bytes/string difference
        if isinstance(param_str, bytes):
            param_str = param_str.decode("utf-8")
        elif isinstance(param_str, np.ndarray):
            param_str = str(param_str.item())
        elif not isinstance(param_str, str):
            param_str = str(param_str)
        data["parameters"] = json.loads(param_str)

        # 3. Get Graph Path
        graph_path = f.attrs["linked_graph_path"]
        if isinstance(graph_path, bytes):
            graph_path = graph_path.decode("utf-8")
        elif isinstance(graph_path, np.ndarray):
            graph_path = str(graph_path.item())
        elif not isinstance(graph_path, str):
            graph_path = str(graph_path)
    if graph:
        # 4. Load Graph (External file)
        if os.path.exists(graph_path):
            data["graph"] = nx.read_graphml(graph_path)
        else:
            # If graph is missing, we just return None (or raise Error if you prefer)
            print(f"Warning: Graph file missing at {graph_path}")
            data["graph"] = None
    else:
        print("\nGraph not loaded")

    print("\n DATA SUCCESFULLT LOADED")

    return data


def get_simulation_path(base_folder, sim_name, parameters=None):
    """
    Generates a valid path and ensures the folder exists.

    Args:
        base_folder (str): e.g., "results" or "/home/user/data"
        sim_name (str): General prefix, e.g., "FHN_Run"
        parameters (dict): Optional. Adds param values to filename for easy searching.

    Returns:
        Path: A full Path object ready for h5py
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_dir = Path(base_folder) / date_str

    output_dir.mkdir(parents=True, exist_ok=True)

    time_str = datetime.now().strftime("%H-%M-%S")
    filename = f"{sim_name}_{time_str}"

    # Optional: Append key parameters to filename (e.g., "Sim_12-00-00_eps0.1.h5")
    if parameters:
        # Filter for crucial params to keep filename short
        if "epsilon" in parameters:
            filename += f"_eps{parameters['epsilon']:.2f}"
        if "how to diffuse" in parameters:
            filename += "_" + parameters["how to diffuse"]
        if "time_length_simulation" in parameters:
            filename += f"_finaltime{parameters['time_length_simulation']:.2f}"
        if "number of nodes" in parameters:
            filename += f"_nodes{parameters['number of nodes']:.2f}"

    filename += ".h5"

    return output_dir / filename
