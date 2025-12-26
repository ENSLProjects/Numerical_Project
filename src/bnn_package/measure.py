#!/usr/bin/env/python3

# ======================= Libraries


import numpy as np
import numba
from scipy.spatial.distance import pdist  # noqa: F401
import networkx as nx
from tabulate import tabulate
import time
#import entropy.entropy as ee
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


def kuramoto_order(X, Y, N, t):
    """Example: Kuramoto Order Parameter (Phase coherence)"""
    # Assuming X is phase or can be converted to phase
    phases = np.arctan2(Y, X)
    z = np.mean(np.exp(1j * phases))
    return np.abs(z)


# ------- The order parameters -------------

def detect_oscillating_nodes(X, params, min_power=1.0e8, f_min=0.01):
    """
    Détection d'oscillations via FFT (Fast Fourier Transform).
    
    Arguments:
    ----------
    X : array (n_time, n_nodes)
        Séries temporelles des potentiels.
    params : dict
        Doit contenir "dt".
    min_power : float
        Puissance spectrale minimale pour considérer que le noeud oscille.
        Remplace 'amp_min'. À ajuster selon l'échelle de tes données.
    f_min : float
        Fréquence minimale ignorée (pour éviter le bruit basse fréquence/dérive).
        
    Retourne:
    ---------
    is_osc : bool array (n_nodes,)
    freq   : float array (n_nodes,) - Fréquence dominante en Hz
    """
    n_time, n_nodes = X.shape
    dt = params["dt"]
    
    X_centered = X - np.mean(X, axis=0)


    fft_spectrum = np.fft.rfft(X_centered, axis=0)
    
    # 3. Calculer le spectre de puissance (Power Spectrum)
    power_spectrum = np.abs(fft_spectrum)**2
    
    # 4. Obtenir les fréquences correspondantes aux indices de la FFT
    freqs = np.fft.rfftfreq(n_time, d=dt)
    
    # --- FILTRAGE DES BASSES FRÉQUENCES ---
    # On ignore les fréquences très basses (drift lent)
    valid_idx = freqs >= f_min
    

    restricted_power = power_spectrum[valid_idx, :]
    restricted_freqs = freqs[valid_idx]
    
    if restricted_power.shape[0] == 0:
        # Cas extrême où tout est sous f_min
        return np.zeros(n_nodes, dtype=bool), np.zeros(n_nodes)

    # 5. Trouver le pic (fréquence dominante) pour chaque noeud
    # argmax retourne l'indice du pic dans la dimension restreinte
    peak_indices = np.argmax(restricted_power, axis=0)
    
    # Récupérer la puissance max et la fréquence correspondante
    max_powers = restricted_power[peak_indices, np.arange(n_nodes)]
    peak_freqs = restricted_freqs[peak_indices]
    
    # 6. Décision : Oscillant ou Bruit ?
    # Si le pic de puissance est trop faible, c'est juste du bruit de fond
    is_osc = max_powers > min_power
    
    # Mettre à NaN ou 0 les fréquences des non-oscillants
    final_freqs = peak_freqs.copy()
    final_freqs[~is_osc] = np.nan
    
    return is_osc, final_freqs

def COH(Trajectory, params):
    threshold = params.get("threshold", 0.0)  # par ex. 0.0 comme défaut
    frac_active = (Trajectory >= threshold).mean(axis=1)   # (n_time,)
    F = np.percentile(frac_active, 95.0)  # max sur le temps de la fraction de neurones "actifs"
    return float(F)

def OSC(Trajectory, params):
    dt = params["dt"]
    is_osc, _ = detect_oscillating_nodes(Trajectory, params)
    osc_fraction = np.mean(is_osc) 
    return osc_fraction

def FMSD(Trajectory, params):
    # 1. Récupérer paramètres temporels

    dt = params["dt"]
    


    # 3. Détection
    is_osc, freq = detect_oscillating_nodes(Trajectory, params)

    # Sécurité si rien n'oscille
    if not np.any(is_osc):
        return 0.0

    freq_use = freq[is_osc]       
    sigma_nu = np.std(freq_use)


        
    return sigma_nu




# --- THE REGISTRY ---

AVAILABLE_METRICS = {
    "COH": COH,
    "OSC": OSC,
    "FMSD": FMSD,
}
