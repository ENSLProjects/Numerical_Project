#!/usr/bin/env python3

# ======================= Libraries


import yaml
import os
import h5py
from typing import cast
import numpy as np
from datetime import datetime
from pathlib import Path
import pandas as pd


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


def save_simulation_data(file_path, trajectory, graph_uuid):
    """
    Saves the trajectory data and the graph UUID to HDF5.

    Args:
        file_path (str): Output path.
        trajectory (np.ndarray): Simulation data.
        graph_uuid (str): The unique identifier of the graph used.
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
        f.attrs["graph_uuid"] = str(graph_uuid)


def save_result(res, i, output_file, mode):
    if mode == "time_series" or not res:
        return
    df_temp = pd.DataFrame([res])
    header = (i == 0) and (not os.path.exists(output_file))
    df_temp.to_csv(output_file, mode="a", header=header, index=False)


def load_simulation_data(file_path, graph: bool = False, load_trajectory: bool = True):
    """
    Loads simulation data.
    Can skip loading the heavy trajectory if only metadata/graph is needed.

    Args:
        file_path (str): Path to the .h5 file.
        graph (bool): If True, loads the .npz graph topology.
        load_trajectory (bool): If False, skips loading the heavy time-series data.

    Returns:
        dict: Contains 'parameters', 'graph_uuid', optionally 'graph' and 'trajectory'.
    """
    path_obj = Path(file_path).resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"{file_path} not found.")

    data = {}
    parent_dir = path_obj.parent

    # ================= 1. HDF5 Access (Lightweight vs Heavy) =================
    with h5py.File(path_obj, "r") as f:
        # Always get the UUID for provenance
        if "graph_uuid" in f.attrs:
            uuid_val = f.attrs["graph_uuid"]
            if isinstance(uuid_val, bytes):
                uuid_val = uuid_val.decode("utf-8")
            data["graph_uuid"] = uuid_val
        else:
            data["graph_uuid"] = "unknown"

        # OPTIONAL: Load heavy trajectory
        if load_trajectory:
            data["trajectory"] = get_data(f, "trajectory")
        else:
            data["trajectory"] = None

    # ================= 2. Load Parameters (YAML) =================
    config_files = list(parent_dir.glob("config_*.yaml"))
    if not config_files:
        raise FileNotFoundError(f"No config_*.yaml found in {parent_dir}")

    config_path = config_files[0]
    data["parameters"] = load_config(str(config_path))

    # ================= 3. Load Graph (.npz) =================
    if graph:
        orig_graph_path = data["parameters"].get("existing_graph_path")
        if orig_graph_path:
            filename = os.path.basename(orig_graph_path)
            local_graph_path = parent_dir / filename

            if local_graph_path.exists() and local_graph_path.suffix == ".npz":
                # Load efficient dict of arrays
                data["graph"] = dict(np.load(str(local_graph_path)))
            else:
                print(f"Warning: Graph .npz not found at {local_graph_path}")
                data["graph"] = None
        else:
            data["graph"] = None

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


def load_config(path):
    """Load the configuration file .yaml

    Args:
        path (str): relative path of the config file

    Returns:
        dic: dictionnary with the parameters as keys and their values as values
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)
