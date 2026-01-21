#!/usr/bin/env/python3

# ======================= Libraries


import numpy as np
import numba
from scipy.spatial.distance import pdist  # noqa: F401
from scipy.signal import hilbert
import networkx as nx
from tabulate import tabulate
import time
import entropy.entropy as ee
from tqdm import tqdm
#from sklearn.cross_decomposition import CCA


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
        bet_dict = nx.betweenness_centrality(G)
        max_betweenness = f"{max(bet_dict.values()):.4f}"

    # 5. Prepare Data for Tabulate
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
    embedding=(1, 1),
    Theiler_correction=4,
    verbose=True,
):
    """
    Computes TE over a range of lags.
    OPTIMIZED: Uses the C-library's internal 'N_real' for fast statistical error estimation.
    """
    val = np.zeros(len(lags))
    std = np.zeros(len(lags))

    if np.std(x) < 1e-6 and np.std(y) < 1e-6:
        return val, std



    x_c = np.ascontiguousarray(x, dtype=np.float64)
    y_c = np.ascontiguousarray(y, dtype=np.float64)

    ee.set_verbosity(1 if verbose else 0)

    iterator = tqdm(lags) if verbose else lags

    for i, tau in enumerate(iterator):
        
        val[i] = ee.compute_TE(
            x_c,
            y_c,
            stride=5,
            lag=tau,
            k=kNN,
            N_eff=n_eff,
            N_real=n_real,
            Theiler=Theiler_correction,
        )[0]
        std[i] = ee.get_last_info()[0]
            

    return val, std





def compute_kl_divergence(p_vec, q_vec):
    """Kullback-Leibler Divergence: D(P || Q)
     Theoretical Correction: TE cannot be negative
     Negative values are estimator bias/noise when True TE approx 0."""

    p_vec = np.maximum(p_vec, 0.0)
    q_vec = np.maximum(q_vec, 0.0)

    p_sum = np.sum(p_vec)
    q_sum = np.sum(q_vec)


    if p_sum == 0 or q_sum == 0:
        return 0.0

    p = p_vec / p_sum
    q = q_vec / q_sum


    epsilon = 1e-15

    return np.sum(p * np.log((p + epsilon) / (q + epsilon)))


# ------- The order parameters -------------

def detect_oscillating_nodes(X, params, min_power=1.0e8, f_min=0.0001):
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
    
    #  Calculer le spectre de puissance (Power Spectrum)
    power_spectrum = np.abs(fft_spectrum)**2
    
    #  Obtenir les fréquences correspondantes aux indices de la FFT
    freqs = np.fft.rfftfreq(n_time, d=dt)
    
    valid_idx = freqs >= f_min
    
    restricted_power = power_spectrum[valid_idx, :]
    restricted_freqs = freqs[valid_idx]
    
    if restricted_power.shape[0] == 0:
        return np.zeros(n_nodes, dtype=bool), np.zeros(n_nodes)

    #  Trouver le pic (fréquence dominante) pour chaque noeud
    peak_indices = np.argmax(restricted_power, axis=0)
    

    max_powers = restricted_power[peak_indices, np.arange(n_nodes)]
    peak_freqs = restricted_freqs[peak_indices]
    
    #  Décision : Oscillant ou Bruit ?
    is_osc = max_powers > min_power
    
    # Mettre à NaN ou 0 les fréquences des non-oscillants
    final_freqs = peak_freqs.copy()
    final_freqs[~is_osc] = np.nan
    
    return is_osc, final_freqs

def COH(Trajectory, params):
    ''' fraction of nodes that spike together once '''
    threshold = params.get("threshold", 0.0) 
    frac_active = (Trajectory >= threshold).mean(axis=1)   
    F = np.percentile(frac_active, 95.0)  
    return float(F)

def OSC(Trajectory, params):
    '''Fraction of of oscilating nodes'''
    dt = params["dt"]
    is_osc, _ = detect_oscillating_nodes(Trajectory, params)
    osc_fraction = np.mean(is_osc) 
    return osc_fraction

def FMSD(Trajectory, params):

    '''Ecart-type des fréquences des noeuds oscillants'''

    dt = params["dt"]
    is_osc, freq = detect_oscillating_nodes(Trajectory, params)

    if not np.any(is_osc):
        return 0.0

    freq_use = freq[is_osc]       
    sigma_nu = np.std(freq_use)

    return sigma_nu



def CRITICALITY_CORRELATION_MEAN(Trajectory,params):

    dt = params['dt']

    correlation_matrix = np.corrcoef(Trajectory, rowvar=False)

    upper_tri_indices = np.triu_indices_from(correlation_matrix, k=1)
    correlation_values = correlation_matrix[upper_tri_indices]

    return np.mean(correlation_values)

def CRITICALITY_CORRELATION_VAR(Trajectory,params):

    dt = params['dt']

    correlation_matrix = np.corrcoef(Trajectory, rowvar=False)
    

    upper_tri_indices = np.triu_indices_from(correlation_matrix, k=1)
    correlation_values = correlation_matrix[upper_tri_indices]
    

    return np.var(correlation_values)

# --- THE REGISTRY ---

AVAILABLE_METRICS_ORDER_PARAMETER = {
    "COH": COH,
    "OSC": OSC,
    "FMSD": FMSD,
    "Mean" : CRITICALITY_CORRELATION_MEAN,
    "Var" : CRITICALITY_CORRELATION_VAR,
    # Add
}


RESEARCH_METRICS = {
    "kl_divergence": compute_kl_divergence
}