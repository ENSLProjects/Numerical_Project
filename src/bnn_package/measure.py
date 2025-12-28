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
from sklearn.cross_decomposition import CCA


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


def MSD(G, X, order: str, average=True, axe=1):
    """
    Return the MSD in the order:
    -- axe=1: esperance_t(variance_node(t))
    -- axe=0: esperance_node(variance_t(node))
    """
    (n, N) = np.shape(X)
    assert N == len(G), "wrong dimension"
    if order == "right":
        msd_values = np.std(X, axis=axe)
        if average:
            return np.mean(msd_values)
        else:
            return msd_values
    elif order == "left":
        msd_values = np.mean(X, axis=axe)
        return np.std(msd_values)
    else:
        raise ValueError(
            f"the way to compute MSD is either 'right' for std+mean or 'left' for mean+std but {order} was given"
        )


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
    x,
    y,
    lags,
    epsilon_noise=1e-8,
    n_real=10,
    n_eff=4096,
    kNN=5,
    embedding=(2, 2),
    Theiler_correction=1,
    verbose=True,
):
    """
    Computes TE over a range of lags.
    OPTIMIZED: Uses the C-library's internal 'N_real' for fast statistical error estimation.
    """
    means = np.zeros(len(lags))
    stds = np.zeros(len(lags))

    if np.std(x) < 1e-6 or np.std(y) < 1e-6:
        # Return zeros immediately. Do not touch the C-library.
        return means, stds

    rng = np.random.default_rng()

    # Add noise BEFORE reordering to prevend the TE being strictly zero
    x_noisy = x + epsilon_noise * rng.standard_normal(x.shape)
    y_noisy = y + epsilon_noise * rng.standard_normal(y.shape)

    x_c = np.ascontiguousarray(x_noisy, dtype=np.float64)
    y_c = np.ascontiguousarray(y_noisy, dtype=np.float64)

    ee.set_verbosity(1 if verbose else 0)

    iterator = tqdm(lags) if verbose else lags

    for i, tau in enumerate(iterator):
        # The C-library's internal N_real can be unstable on some architectures.
        realizations = []
        # To get a mean/std, we need to ask the library for N_real > 1 OR loop here.
        for _ in range(max(1, n_real)):
            val = ee.compute_TE(
                x_c,
                y_c,
                n_embed_x=embedding[0],
                n_embed_y=embedding[1],
                stride=1,
                lag=tau,
                k=kNN,
                N_eff=n_eff,
                N_real=1,
                Theiler=Theiler_correction,
            )
            # Handle case where it returns a list/tuple even for N_real=1
            if isinstance(val, (list, tuple)):
                realizations.append(val[0])
            else:
                realizations.append(val)
        # Compute Stats manually
        if n_real > 1:
            means[i] = np.mean(realizations)
            stds[i] = np.std(realizations)
        else:
            means[i] = realizations[0]
            stds[i] = 0.0
    return means, stds


def kuramoto_order(X, Y, N, t):
    """Example: Kuramoto Order Parameter (Phase coherence)"""
    # Assuming X is phase or can be converted to phase
    phases = np.arctan2(Y, X)
    z = np.mean(np.exp(1j * phases))
    return np.abs(z)


def compute_kl_divergence(p_vec, q_vec):
    """Kullback-Leibler Divergence: D(P || Q)"""
    # Theoretical Correction: TE cannot be negative
    # Negative values are estimator bias/noise when True TE approx 0.
    # We clip them to 0 (plus epsilon for log stability).
    p_vec = np.maximum(p_vec, 0.0)
    q_vec = np.maximum(q_vec, 0.0)

    p_sum = np.sum(p_vec)
    q_sum = np.sum(q_vec)

    # Handle empty/silent systems
    if p_sum == 0 or q_sum == 0:
        return 0.0

    p = p_vec / p_sum
    q = q_vec / q_sum

    # 3. Compute KL with epsilon stability
    # We add epsilon INSIDE the log to prevent log(0)
    epsilon = 1e-15

    return np.sum(p * np.log((p + epsilon) / (q + epsilon)))


def compute_cca_score(p_vec, q_vec):
    """Canonical Correlation Analysis (1D Approximation for samples)"""
    cca = CCA(n_components=1)
    X_c, Y_c = cca.fit_transform(p_vec.reshape(-1, 1), q_vec.reshape(-1, 1))
    return np.corrcoef(X_c.T, Y_c.T)[0, 1]


# --- THE REGISTRY ---

AVAILABLE_METRICS_ORDER_PARAMETER = {
    "sync_error": Synchronized_error,
    "kuramoto": kuramoto_order,
    "mean standard deviation": MSD_vec_xy,
    # Add
}


RESEARCH_METRICS = {
    "kl_divergence": compute_kl_divergence,
    "cca_alignment": compute_cca_score,
}
