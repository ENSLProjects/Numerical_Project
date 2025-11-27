#!/usr/bin/env/python3

# ======================= Libraries

import numpy as np
import numba
from scipy.spatial.distance import pdist  # noqa: F401
import networkx as nx
from tabulate import tabulate
import time
from datetime import datetime
from pathlib import Path

# ======================= Functions


def prepare_data(arr):
    """
    Prépare les données pour la lib C Entropy.
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


@numba.jit(nopython=True)
def count(data, axis: str):
    assert len(np.shape(data)) == 4, "wrong data shape input"
    if axis == "x":
        p = (0, 0)
    elif axis == "y":
        p = (0, 1)
    elif axis == "u":
        p = (1, 0)
    elif axis == "v":
        p = (1, 1)
    else:
        return "axis not existent"
    simu = np.sort(data[:, p[0], p[1], :], axis=0)
    k = len(simu[0])
    nb_points = [1] * k
    for i in range(k):
        start = simu[0, i]
        for j in range(1, len(simu)):
            if simu[j, i] != start:
                nb_points[i] += 1
                start = simu[j, i]
    return nb_points


def find_settling_time(signal, final_n_samples, tolerance_percent=1):
    """
    Finds the time when the signal settles within a tolerance band of its final value.
    """
    final_value = np.mean(signal[-final_n_samples:])
    total_change = np.max(np.abs(signal - final_value))
    tolerance = (tolerance_percent / 100.0) * total_change
    upper_bound = final_value + tolerance
    lower_bound = final_value - tolerance
    outside_mask = (signal > upper_bound) | (signal < lower_bound)
    if not np.any(outside_mask):
        return 0  # It never started outside the band
    last_index_outside = np.where(outside_mask)[0][-1]
    settling_index = last_index_outside + 1
    if settling_index >= len(
        signal
    ):  # Handle case where it never settles within the given window
        print("Warning: Signal might not have settled.")
        settling_index = len(signal) - 1
    return settling_index, settling_index, (lower_bound, upper_bound)


@numba.jit(nopython=True)
def MSD_xy(G, X, Y):
    """
    Return the MSD for a given epsilon
    """
    n, N = X.shape
    assert n == len(G), "wrong dimension"
    MSD_values = np.zeros(N)
    mean_X = np.mean(X, axis=0)
    mean_Y = np.mean(Y, axis=0)
    for t in range(N):
        MSD_t = np.mean((X[:, t] - mean_X[t]) ** 2 + (Y[:, t] - mean_Y[t]) ** 2)
        MSD_values[t] = MSD_t
    return MSD_values


def MSD_vec_xy(G, X, Y):
    """
    Return the MSD for a given epsilon
    """
    n, N = X.shape
    assert n == len(G), "wrong dimension"
    MSD_values = np.zeros(N)
    mean_X = np.mean(X, axis=0)
    mean_Y = np.mean(Y, axis=0)
    MSD_values = np.mean(
        (X - mean_X[np.newaxis, :]) ** 2 + (Y - mean_Y[np.newaxis, :]) ** 2, axis=1
    )
    return MSD_values


def MSD(G, X, average=True, axe=1):
    """
    Return the MSD in the order:
    -- axe=1: esperance_t(variance_node(t))
    -- axe=0: esperance_node(variance_t(node))
    """
    (n, N) = np.shape(X)
    assert N == len(G), "wrong dimension"
    msd_values = np.std(X, axis=axe)
    if average:
        return np.mean(msd_values)
    else:
        return msd_values


def MSD_inverse(G, X, axe=0):
    """
    Return the MSD in the order:
    -- axe=1: esperance_t(variance_node(t))
    -- axe=0: esperance_node(variance_t(node))
    """
    n, N = X.shape
    assert N == len(G), "wrong dimension"
    MSD_values = np.mean(X, axis=axe)
    return np.std(MSD_values)


@numba.jit(nopython=True)
def Synchronized_error(X, Y, N_nodes, time):  # not easily vectorizable
    assert N_nodes > 1, "Require at least two nodes"
    error = 0
    for i in range(N_nodes - 1):
        for j in range(i + 1, N_nodes):
            error += np.sqrt(
                (X[time, i] - X[time, j]) ** 2 + (Y[time, i] - Y[time, j]) ** 2
            )
    return 2 * error / N_nodes / (N_nodes - 1)


def Synchronized_error_mean():
    return 4


def print_simulation_report(adj_matrix, fast_mode=False):
    """
    Computes graph topology metrics and prints a table to the console.

    Args:
        adj_matrix (np.array): The adjacency matrix.
        sim_name (str): Label for the simulation.
        fast_mode (bool): If True, skips slow metrics (Betweenness, Path Length).
    """
    t_start = time.time()

    # 1. Basic Stats (NumPy - Instant)
    N = adj_matrix.shape[0]
    num_edges = np.sum(adj_matrix) / 2
    avg_degree = np.mean(np.sum(adj_matrix, axis=1))
    density = (2 * num_edges) / (N * (N - 1)) if N > 1 else 0

    # 2. NetworkX Object (Low Overhead)
    G = nx.from_numpy_array(adj_matrix)

    # 3. Connectivity & Clustering
    is_connected = nx.is_connected(G)
    clustering = nx.average_clustering(G)

    # 4. Heavy Metrics (Optional)
    avg_path = "Skipped"
    diameter = "Skipped"
    max_betweenness = "Skipped"

    if not fast_mode:
        # Path Lengths
        if is_connected:
            avg_path = f"{nx.average_shortest_path_length(G):.4f}"
            diameter = nx.diameter(G)
        else:
            avg_path = "Inf (Disconnected)"
            diameter = "Inf"

        # Betweenness (The expensive part)
        # We only take the max value to keep the table clean
        bet_dict = nx.betweenness_centrality(G)
        max_betweenness = f"{max(bet_dict.values()):.4f}"

    # 5. Prepare Data for Tabulate
    # We create a list of lists
    table_data = [
        ["Metric", "Value", "Description"],
        #["Simulation ID", sim_name, "Run Identifier"],
        ["Edges (E)", int(num_edges), "Total Connections"],
        ["Avg Degree (<k>)", f"{avg_degree:.4f}", "Avg neighbors per node"],
        ["Density", f"{density:.4f}", "Actual/Possible edges"],
        ["Connected?", "Yes" if is_connected else "No", "Is graph one component?"],
        ["Clustering Coeff", f"{clustering:.4f}", "Local triangular loops"],
        ["Avg Path Length", avg_path, "Global Integration"],
        ["Diameter", diameter, "Longest shortest path"],
        ["Max Betweenness", max_betweenness, "Centrality of hub node"],
    ]

    # 6. Print using 'fancy_grid' for that scientific look
    print("\n" + "=" * 60)
    print(f">>> SIMULATION INITIALIZATION LOG [{time.strftime('%H:%M:%S')}]")
    print("=" * 60)
    print(tabulate(table_data, headers="firstrow", tablefmt="fancy_grid"))

    t_end = time.time()
    print(f"\n[System] Topology analysis completed in {t_end - t_start:.3f}s")
    print("=" * 60 + "\n")

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
    # 1. Define the Output Directory
    # We use Path() so this works on Windows and Linux automatically
    # We add a date subfolder to keep things organized
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_dir = Path(base_folder) / date_str
    
    # 2. Create the directory if it doesn't exist
    # parents=True allows creating "results/2023-10-27" in one go
    # exist_ok=True prevents error if folder already exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Construct the Filename
    # Start with a Timestamp for uniqueness
    time_str = datetime.now().strftime("%H-%M-%S")
    filename = f"{sim_name}_{time_str}"
    
    # Optional: Append key parameters to filename (e.g., "Sim_12-00-00_eps0.1.h5")
    if parameters:
        # Filter for crucial params to keep filename short
        if 'epsilon' in parameters:
            filename += f"_eps{parameters['epsilon']:.2f}"
        if 'alpha' in parameters:
            filename += f"_a{parameters['alpha']:.2f}"
        if 'time_length_simulation' in parameters:
            filename += f"_finaltime{parameters['time_length_simulation']:.2f}"
        if 'number of nodes' in parameters:
            filename += f"_nodes{parameters['number of nodes']}"
            
    filename += ".h5"
    
    # 4. Join folder and filename
    return output_dir / filename
