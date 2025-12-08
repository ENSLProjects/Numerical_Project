#!/usr/bin/env/python3

# ======================= Libraries


import numpy as np
import numba
from scipy.spatial.distance import pdist  # noqa: F401
import networkx as nx
from tabulate import tabulate
import time
import entropy.entropy as ee
from tqdm import tqdm


# ======================= Functions


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
    print(
        f">>> GRAPH ACTIVE NODES ONLY TOPOLOGY ANALYSIS LOG [{time.strftime('%H:%M:%S')}]"
    )
    print("=" * 60)
    print(tabulate(table_data, headers="firstrow", tablefmt="fancy_grid"))

    t_end = time.time()
    print(f"\n[System] Topology analysis completed in {t_end - t_start:.3f}s")
    print("=" * 60 + "\n")


def compute_te_over_lags(
    x, y, lags, n_real=10, n_eff=4096, kNN=5, embedding=(2, 2), Theiler_correction=15
):
    """
    Computes TE over a range of lags with explicit Theiler correction.
    """
    means = np.zeros(len(lags))
    stds = np.zeros(len(lags))
    ee.multithreading(do_what="auto")
    print(f"Computing TE over {len(lags)} lags (Range: {lags[0]}-{lags[-1]})...")
    for i, tau in enumerate(tqdm(lags)):
        current_lag_values = []

        # We loop manually to capture the STD (distribution) of the TE
        for _ in range(n_real):
            val = ee.compute_TE(
                x,
                y,
                n_embed_x=embedding[0],
                n_embed_y=embedding[1],
                stride=1,
                lag=tau,
                k=kNN,
                N_eff=n_eff,
                N_real=1,  # We handle realizations manually here for the std
                Theiler=Theiler_correction,
            )[0]

            current_lag_values.append(val)

        means[i] = np.mean(current_lag_values)
        stds[i] = np.std(current_lag_values)

    return means, stds


def mean_activity(X, Y, N, t):
    """Simple example metric: Mean global voltage"""
    return np.mean(X)


def kuramoto_order(X, Y, N, t):
    """Example: Kuramoto Order Parameter (Phase coherence)"""
    # Assuming X is phase or can be converted to phase
    phases = np.arctan2(Y, X)
    z = np.mean(np.exp(1j * phases))
    return np.abs(z)


# --- THE REGISTRY ---

AVAILABLE_METRICS = {
    "sync_error": Synchronized_error,
    "mean_activity": mean_activity,
    "kuramoto": kuramoto_order,
    "mean standard deviation": MSD_vec_xy,
    # Add
}
