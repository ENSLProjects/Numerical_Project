#!/usr/bin/env python3

# ======================= Libraries


import numpy as np
import os
import sys
import traceback
from numpy.random import default_rng
from scipy.linalg import expm
from bnn_package import (
    generate_poisson_input,
    get_coupling_operator,
    evolve_system,
    rk4_step,
    prepare_data,
    compute_te_over_lags,
    FitzHughNagumoModel,
    save_simulation_data,
    AVAILABLE_METRICS_ORDER_PARAMETER,
    RESEARCH_METRICS,
)
import random
import networkx as nx 
from multiprocessing import shared_memory



# --- VARIABLES GLOBALES ---
_global_graph = None
_global_passive_counts = None
_global_input_signal = None 

# --- INITIALIZER (Called once) ---
def init_worker_globals(graph, passive_counts, input_signal):
    """
    Stocke les données lourdes dans la mémoire globale du worker.
    Ceci évite de les renvoyer via Pickle à chaque tâche.
    """
    global _global_graph, _global_passive_counts, _global_input_signal
    _global_graph = graph
    _global_passive_counts = passive_counts
    _global_input_signal = input_signal 

# --- WRAPPER (Exécuté pour chaque tâche) ---
def worker_wrapper(task):
    """
    Reçoit les paramètres légers (epsilon, gamma, etc.) dans 'task'.
    Récupère les données lourdes depuis les globales.
    Lance la simulation.
    """
    try:
        if _global_graph is not None:
            task["preloaded_graph"] = _global_graph
        
        if _global_passive_counts is not None:
            task["passive_counts"] = _global_passive_counts

        if _global_input_signal is not None:
            task["input_signal"] = _global_input_signal

        
        mode = task.get("mode", "TE")
        
        if mode == "TE":
            return research_alignment_worker_TE_tensor(task)
        elif mode == "ORDER":
            return run_order_parameter(task)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    except Exception as e:
        print(f"!!! Worker Error in task {task.get('id', '?')}: {e}")
        traceback.print_exc()
        return None


def load_graph_topology(path):
    """Loads the graph data safely."""
    data = np.load(path)
    graph_uuid = str(data["uuid"]) if "uuid" in data else "unknown"
    return (data["adjacency"], data["positions"], data["passive_counts"], graph_uuid)


def get_stable_dt(params, adjacency):
    """
    Heuristic to determine a stable dt for FitzHugh-Nagumo.
    Accounts for stiffness (fhn_eps) and network coupling (spectral radius).
    """
    # 1. Local Dynamics: Stability is limited by the fast/slow timescale ratio
    # If fhn_eps is very small, the recovery variable 'w' is much slower than 'v'
    fhn_eps = float(params.get("fhn_eps", 0.08))
    tau_local = 1.0 / fhn_eps

    # 2. Network Dynamics: Stability is limited by the coupling strength and graph
    # For Laplacian/Diffusive operators, we estimate the spectral radius
    coupling_str = float(params.get("epsilon", 0.01))
    diff_type = params.get("diffusive_operator", "Diffusive")

    if diff_type == "Laplacian":
        # Gershgorin circle theorem upper bound for Laplacian: 2 * max_degree
        degrees = np.sum(adjacency, axis=1)
        spec_radius = 2.0 * np.max(degrees)
    else:
        # For diffusive/row-normalized matrices, the max eigenvalue is ~1
        spec_radius = 1.0

    tau_net = 1.0 / (coupling_str * spec_radius)

    # 3. Apply safety margin (RK4 typically requires dt < 10% of fastest timescale)
    # We take the minimum of local, network, and a baseline safety value (0.05)
    suggested_dt = min(tau_local, tau_net, 1.0) * 0.05

    # Ensure it's not too small (performance) or too large (instability)
    return max(min(suggested_dt, 0.01), 0.001)


def create_model(params, adjacency, N_p):
    """
    Instantiates the JIT-compiled Physics Model with CORRECT parameter mapping.
    """
    # 1. Determine Coupling Operator Type
    diff_type = params.get("diffusive_operator", "Diffusive")
    type_diff_code = 1 if diff_type == "Laplacian" else 0
    coupling_op = np.ascontiguousarray(
        get_coupling_operator(adjacency, diff_type), dtype=np.float64
    )
    coupling_str = float(params.get("epsilon", 0.01))
    fhn_eps = float(params.get("fhn_eps", 0.08))
    alpha = float(params.get("alpha", 0.2))
    k = float(params.get("k", 0.25))
    dt = get_stable_dt(params, adjacency)
    cr = float(params.get("cr", 1.0))
    a = float(params.get("a", 3.0))
    vrp = float(params.get("vrp", 1.5))
    model = FitzHughNagumoModel(
        coupling_op=coupling_op,
        alpha=alpha,
        fhn_eps=fhn_eps,  
        k=k,
        vrp=vrp,
        coupling_str=coupling_str,  # Network Coupling (mapped from epsilon)
        dt=dt,
        type_diff_code=type_diff_code,
        cr=cr,
        np_vec=N_p.astype(np.float64),  # Passive node counts vector
        a=a,
    )

    return model


def common_worker_setup(params):
    """Shared setup logic: Charge le graphe et initialise le modèle."""
    
    # 1. RECUPERATION DU GRAPHE ET DES PASSIFS
    if "preloaded_graph" in params:

        adjacency = params["preloaded_graph"]
        graph_uuid = params.get("graph_uuid", "unknown")
        

        if params.get("passive_counts") is not None:
            N_p = params["passive_counts"]
            print(f"[DEBUG] Passive counts loaded. Total passive nodes: {np.sum(N_p)}")
        else:


            mean_p = params.get("mean_poisson", 0.7)
            rng = np.random.default_rng(params.get("seed"))
            N_p = rng.poisson(mean_p, len(adjacency))
            
    else:

        graph_path = params["graph_file_path"]
        adjacency, pos, N_p, graph_uuid = load_graph_topology(graph_path)

    # 2. PREPARATION DES MATRICES (Nécessaire pour FHN)
    N_p = N_p.astype(np.float64) 
    
    if params.get("diffusive_operator", "Diffusive") == "Laplacian":
        degrees = np.sum(adjacency, axis=1)
        coupling_op = np.diag(degrees) - adjacency 
        type_diff_code = 1 
    else:
        coupling_op = adjacency
        type_diff_code = 0
    
    coupling_op = np.ascontiguousarray(coupling_op, dtype=np.float64)

    # 3. CREATION DU MODELE
    model = FitzHughNagumoModel(
        coupling_op=coupling_op,
        alpha=params.get("alpha", 0.2),
        a=params.get("a", 3.0),
        fhn_eps=params.get("fhn_eps", 0.08),
        k=params.get("k", 0.25),
        vrp=params.get("vrp", 1.5),
        coupling_str=params.get("epsilon"),
        dt=params.get("dt", 0.01),
        type_diff_code=type_diff_code,
        cr=params.get("cr"),
        np_vec=N_p 
    )

    return model, graph_uuid

def get_stratified_pairs(adjacency, config):
    """
    Selects pairs based on topological distance to bypass N^2 cost.
    """
    G = nx.from_numpy_array(adjacency)
    all_pairs = {}

    # 1. Direct Neighbors (Dist 1)
    edges = list(G.edges())
    n_d1 = config.get("n_dist1", 1000)
    if len(edges) > n_d1:
        all_pairs["dist1"] = random.sample(edges, n_d1)
    else:
        all_pairs["dist1"] = edges

    # 2. Dist 2 and Dist 3+ (BFS sampling)
    nodes = list(G.nodes())
    n_d2 = config.get("n_dist2", 1000)
    n_d3 = config.get("n_dist3", 1000)

    d2_found, d3_found = [], []
    attempts = 0
    max_attempts = (n_d2 + n_d3) * 100 

    while (len(d2_found) < n_d2 or len(d3_found) < n_d3) and attempts < max_attempts:
        u, v = random.sample(nodes, 2)
        try:
            d = nx.shortest_path_length(G, source=u, target=v)
            if d == 2 and len(d2_found) < n_d2:
                d2_found.append((u, v))
            elif d >= 3 and len(d3_found) < n_d3:
                d3_found.append((u, v))
        except nx.NetworkXNoPath:
            pass
        attempts += 1

    all_pairs["dist2"] = d2_found
    all_pairs["dist3"] = d3_found

    return all_pairs




# ======================= Simulation Functions =======================


def time_series(params):
    """
    Runs a full simulation and saves the trajectory.
    """

    # 1. Setup & Topology 
    model, graph_uuid = common_worker_setup(params)
    n_nodes = params["number_of_nodes"]
    total_time = int(params["total_time"])
    res_topo = None


    if "preloaded_graph" in params:
        adj = params["preloaded_graph"]  
    else:
        print("[DEBUG] Fallback to file loading (preloaded_graph not found)")
        res_topo = load_graph_topology(params["graph_file_path"])
        adj = res_topo[0]
    coordinates = None
    
    if len(res_topo) > 1 and isinstance(res_topo[1], (np.ndarray, list)):
        possible_coords = np.array(res_topo[1])
        if possible_coords.shape[0] == params["number_of_nodes"] and possible_coords.shape[1] in [2, 3]:
            coordinates = possible_coords

    if coordinates is None:
        try:
            G_tmp = nx.read_gml(params["graph_file_path"]) 
            pos_dict = nx.get_node_attributes(G_tmp, 'pos')
            if not pos_dict:
                coordinates = np.random.rand(params["number_of_nodes"], 2)
            else:
                coordinates = np.array([pos_dict[i] for i in range(len(G_tmp))])
        except Exception as e:
            print(f"Warning: Could not load coords ({e}).Using something else.")
            coordinates = np.random.rand(params["number_of_nodes"], 2)

    # 2. Simulate (In-Memory)
    total_time = int(params["total_time"])
    rng = default_rng(params.get("seed", None))
    n_nodes = params["number_of_nodes"]

    State_0 = np.zeros((3, n_nodes), dtype=np.float64)
    State_0[0] = 0.1 + noise * rng.standard_normal(n_nodes)
    State_0[1] = 0.3 + noise * rng.standard_normal(n_nodes)
    State_0[2] = 1.0 + noise * rng.standard_normal(n_nodes)

    # --- GENERATION INPUT ---
    input_signal = None
    target_idxs = []

    if params.get("input_active", False):
        input_signal, target_idxs = generate_poisson_input(
            n_nodes=params["number_of_nodes"],
            final_time=int(params["total_time"]),      
            dt=params["dt"],
            coordinates=None,                          
            n_targets=params.get("input_targets", 10),
            rate_hz=params["input_rate"],              
            magnitude=params["input_magnitude"],       
            forced_targets=params.get("forced_targets", None)
        )
        
    

    
    
    input_signal = np.zeros(total_time, dtype=np.float64)    
    traj = evolve_system(model, State_0, total_time, input_signal=input_signal, rk4_step=rk4_step)
    full_data = traj[:, :, :] 

    # 4. Save Logic
    output_folder = params.get("output_folder", "Data_output")

    coupling_val = model.coupling_str
    cr_val = model.c_r
    filename = f"ts_N{n_nodes}_Coup{coupling_val:.3f}_cr{cr_val:.3f}_G-{graph_uuid}.h5"
    save_path = os.path.join(output_folder, filename)

    transitory = int(params.get("transitory_time", total_time * 0.1))

    with np.errstate(over="ignore"):
        data_to_save = full_data[transitory:].astype(np.float32)

    save_simulation_data(save_path, data_to_save, graph_uuid)

    return {}


def run_order_parameter(params):
    """
    Runs simulation WITHOUT input perturbation to measure global dynamics 
    (Synchronization, Metastability, etc.).
    Implementation strictly follows 'time_series' pattern to avoid Numba errors.
    """
    
    # 1. Setup & Topology
    model, graph_uuid = common_worker_setup(params)
    n_nodes = params["number_of_nodes"]
    total_time = int(params["total_time"])
    

    if "preloaded_graph" in params:
        adj = params["preloaded_graph"] 
        res_topo = (adj, None)
    else:
        print("[DEBUG] Fallback to file loading (preloaded_graph not found)")
        res_topo = load_graph_topology(params["graph_file_path"])
        adj = res_topo[0]

    # 2. Simulate (In-Memory)
    rng = default_rng(params.get("seed", None))
    
    State_0 = np.zeros((3, n_nodes), dtype=np.float64)
    State_0[0] = 0.1 + 0.1 * rng.standard_normal(n_nodes)
    State_0[1] = 0.3 + 0.1 * rng.standard_normal(n_nodes)
    State_0[2] = 1.0 + 0.1 * rng.standard_normal(n_nodes)

    # 3. Input Signal 
    input_signal = np.zeros(total_time, dtype=np.float64)

    # 4. Simulation
    traj = evolve_system(model, State_0, total_time, input_signal , rk4_step)

    # 5. Extraction des données
    start = int(params["transitory_time"])
    

    voltage_data = traj[start:, 0, :]

    # 6. Calcul des métriques
    results = {
        "epsilon": model.coupling_str,
        "fhn_eps": model.fhn_eps, 
        "cr": model.c_r,
        "graph_uuid": graph_uuid,
    }

    metrics = params.get("metrics", [])
    
    # Sécurité pour les metrics demandées
    for m in metrics:
        if m in AVAILABLE_METRICS_ORDER_PARAMETER:
            try:
                val = AVAILABLE_METRICS_ORDER_PARAMETER[m](voltage_data, params)
                results[m] = val
            except Exception as e:
                print(f"Error calculating metric {m}: {e}")
                results[m] = np.nan

    return results




def research_alignment_worker(params):
    """
    Simulates FHN, loops over LAGS, computes Stratified TE vs Theory,
    and returns flat research metrics. (RAM Optimized)
    """
    # 1. Setup
    model, graph_uuid = common_worker_setup(params)


    adj, _, _, _ = load_graph_topology(params["graph_file_path"])

    # 2. Simulate (In-Memory)
    total_time = int(params["total_time"])
    rng = default_rng(params.get("seed", None))
    n_nodes = params["number_of_nodes"]
    noise = params.get("noise", 0.0)

    State_0 = np.zeros((3, n_nodes), dtype=np.float64)
    State_0[0] = 0.1 + noise * rng.standard_normal(n_nodes)
    State_0[1] = 0.3 + noise * rng.standard_normal(n_nodes)
    State_0[2] = 1.0 + noise * rng.standard_normal(n_nodes)

    traj = evolve_system(model, State_0, total_time, None, rk4_step)
 

    start = int(params.get("transitory_time", 1000))
    voltage_data = traj[start:, 0, :]

    results = {
        "epsilon": model.coupling_str,
        "cr": model.c_r,
        "fhn_eps": model.fhn_eps,
        "graph_uuid": graph_uuid,
    }

    if not np.isfinite(voltage_data).all():
        print(
            f"!!! CRASH DETECTED: Simulation exploded for eps={model.coupling_str}, cr={model.c_r}"
        )
        return {
            "epsilon": model.coupling_str,
            "cr": model.c_r,
            "error": "Simulation exploded (NaN/Inf values), try to look at rk4 dt",
        }


    var = np.std(voltage_data)
    if var < 1e-9:
        print(
            f">>> SKIPPING TE: Fixed point detected (Std={var:.2e}) for cr={model.c_r}"
        )


        analysis_cfg = params.get("research_analysis", {})
        lags = analysis_cfg.get("te_lags", [1])
        metrics = analysis_cfg.get("metrics", ["kl_divergence"])

        for tau in lags:
            results[f"te_uncertainty_lag{tau}"] = 0.0
            for m in metrics:
                results[f"{m}_lag{tau}"] = (
                    0.0  
                )

        return results

    # 3. Analyze
    analysis_cfg = params.get("research_analysis", {})
    if not analysis_cfg.get("active", False):
        return {"error": "Research analysis inactive in config"}

    # A. Get Pairs
    pairs_dict = get_stratified_pairs(adj, analysis_cfg["stratified_sampling"])

    # B. Loop over Lags (Flattening dimensions)
    lags_to_test = analysis_cfg.get("te_lags", [1])

    n_real = analysis_cfg.get("n_real", 1)

    for tau in lags_to_test:
        L_exp = expm(-model.coupling_op * model.dt * tau)
 
        measured_means = []
        measured_stds = []
        theory_vals = []

        for group, pairs in pairs_dict.items():
            for u, v in pairs:
                x = prepare_data(voltage_data[:, u])
                y = prepare_data(voltage_data[:, v])


                val_means, val_stds = compute_te_over_lags(
                    x,
                    y,
                    stride = tau,
                    n_real=n_real,
                    n_eff=analysis_cfg["n_eff"],
                    lag = [tau],
                    kNN=analysis_cfg["kNN"],
                    verbose=False,
                )

                measured_means.append(val_means[0])
                measured_stds.append(val_stds[0])
                theory_vals.append(L_exp[u, v])

        # Convert to arrays and Compute Metrics
        vec_means = np.array(measured_means)
        vec_stds = np.array(measured_stds)
        vec_theory = np.array(theory_vals)

        # Save uncertainty metric (Good for error bars in plots later)
        results[f"te_uncertainty_lag{tau}"] = np.mean(vec_stds)
        results[f"te_value_lag{tau}"] = np.mean(vec_means)

        # Compute Metrics using the ROBUST MEAN
        requested_metrics = analysis_cfg.get("metrics", ["kl_divergence"])
        for metric_name in requested_metrics:
            if metric_name in RESEARCH_METRICS:
                score = RESEARCH_METRICS[metric_name](vec_means, vec_theory)
                results[f"{metric_name}_lag{tau}"] = score

    return results

def research_alignment_worker_2(params):
    """
    Simulates FHN with POISSON INPUT, loops over LAGS.
    """
    # 1. Setup & Topology
    model, graph_uuid = common_worker_setup(params)
    
    # --- RECUPERATION DES COORDONNEES ---
    # On essaie de récupérer les coords depuis le loader
    # Si load_graph_topology renvoie (adj, coords, ...), c'est gagné.
    res_topo = load_graph_topology(params["graph_file_path"])
    adj = res_topo[0]
    
    coordinates = None
    
    # Tentative 1 : Vérifier si le 2ème retour est utilisable comme coordonnées
    if len(res_topo) > 1 and isinstance(res_topo[1], (np.ndarray, list)):
        possible_coords = np.array(res_topo[1])
        # Vérif simple : shape (N, 2) ou (N, 3)
        if possible_coords.shape[0] == params["number_of_nodes"] and possible_coords.shape[1] in [2, 3]:
            coordinates = possible_coords

    # Tentative 2 : Si pas trouvé, on charge le graphe brute force (Fallback)
    if coordinates is None:
        # On suppose que le fichier est un .gml, .graphml ou .pkl géré par networkx
        try:
            # Charge le graphe (adapte read_gml selon ton format réel)
            G_tmp = nx.read_gml(params["graph_file_path"]) 
            # Extraction des positions (clé 'pos', 'x'/'y', ou 'coordinates')
            # Souvent 'pos' dans les générateurs standards
            pos_dict = nx.get_node_attributes(G_tmp, 'pos')
            if not pos_dict:
                # Si pas d'attribut pos, on génère des coords aléatoires (mieux que rien)
                coordinates = np.random.rand(params["number_of_nodes"], 2)
            else:
                # Conversion dict -> array trié par index
                coordinates = np.array([pos_dict[i] for i in range(len(G_tmp))])
        except Exception as e:
            # Dernier recours : Coordonnées aléatoires pour éviter le crash
            print(f"Warning: Could not load coords ({e}).Using Random.")
            coordinates = np.random.rand(params["number_of_nodes"], 2)

    # 2. Simulate (In-Memory)
    total_time = int(params["total_time"])
    rng = default_rng(params.get("seed", None))
    n_nodes = params["number_of_nodes"]

    State_0 = np.zeros((3, n_nodes), dtype=np.float64)
    State_0[0] = 0.1 + 0.1 * rng.standard_normal(n_nodes)
    State_0[1] = 0.3 + 0.1 * rng.standard_normal(n_nodes)
    State_0[2] = 1.0 + 0.1 * rng.standard_normal(n_nodes)

    # --- GENERATION INPUT ---
    # On lit les params du JSON ou on prend des défauts
    input_signal = None
    if params.get("input_active", False):
        input_signal, target_ind = generate_poisson_input(
            n_nodes=n_nodes,
            final_time=total_time,
            dt=params['dt'],
            coordinates=coordinates, # <-- Les voilà !
            n_targets=params.get("input_targets", 10),
            rate_hz=params.get("input_rate", 5.0),
            magnitude=params.get("input_magnitude", 1.0)
        )
    
    # Note : evolve_system gère "input_signal is None" en interne ou on passe des zéros if not active
    if input_signal is None:
        input_signal = np.zeros((total_time, n_nodes))
        
    traj = evolve_system(model, State_0, total_time, input_signal, rk4_step)

    # ... (La suite reste identique : Extract Voltage, Checks, Analysis)
    # ...
    
    # Extract Voltage (Post-Transitory)
    start_transitory = int(params.get("transitory_time", 1000))
    voltage_data = traj[start_transitory:, 0, :]
    
    results = {
        "epsilon": model.coupling_str,
        "cr": model.c_r,
        "fhn_eps": model.fhn_eps,
        "graph_uuid": graph_uuid,
    }
    
    # Safety Checks
    if not np.isfinite(voltage_data).all():
        return {"error": "Simulation exploded (NaN/Inf values)"}
    if np.std(voltage_data) < 1e-9:
        results["status"] = "fixed_point"
        return results

    # 3. Analyze
    analysis_cfg = params.get("research_analysis", {})
    if not analysis_cfg.get("active", False):
        return {"error": "Research analysis inactive"}

    # A. Get Pairs
    pairs_dict = get_stratified_pairs(adj, analysis_cfg["stratified_sampling"])

    # B. Configuration TE
    base_stride = analysis_cfg.get("base_stride", 40) # Le paramètre crucial
    lags_indices = analysis_cfg.get("te_lags", [1])   # Indices [1, 2, 3...]
    n_real = analysis_cfg.get("n_real", 10)           # Nombre de fenêtres glissantes
    
    # Génération des offsets pour le sliding window
    # Ex: si stride=40 et n_real=10 -> offsets = [0, 4, 8, 12, ...]
    if n_real > base_stride:
        n_real = base_stride # Impossible de faire plus de shifts que le stride
    offsets = np.linspace(0, base_stride - 1, n_real, dtype=int)

    # C. Loop over Lags Indices
    for lag_idx in lags_indices:

        # --- THEORIE ---
        # Temps physique = (index dans le tableau réduit) * (temps entre 2 points réduits)
        tau_physical_seconds = lag_idx * base_stride * model.dt
        L_exp = expm(-model.coupling_op * tau_physical_seconds)

        for group, pairs in pairs_dict.items():
            for u, v in pairs:
                
                # --- STATISTIQUES (Sliding Window) ---
                te_samples = []

                # On teste plusieurs décalages temporels pour la même paire et le même lag
                # Cela remplace le "n_real" interne de la fonction précédente
                for offset in offsets:
                    # 1. Extraction ("Sliding Subsampling")
                    # On prend 1 point tous les 'base_stride', en commençant à 'offset'
                    raw_x_sub = voltage_data[offset::base_stride, u]
                    raw_y_sub = voltage_data[offset::base_stride, v]
                    
                    
                    

                    x_sub = prepare_data(raw_x_sub)
                    y_sub = prepare_data(raw_y_sub)

                    # 2. Calcul TE sur cette réalisation
                    # stride = 1 car le sous-échantillonnage est déjà fait manuellement
                    # lag = [lag_idx] est l'index dans ce tableau réduit
                    val_means, _ = compute_te_over_lags(
                        x_sub,
                        y_sub,         # Important fix
                        lags=[lag_idx],     # Délai en indices du tableau réduit
                        n_real=1,          # 1 seule mesure déterministe par offset
                        n_eff=analysis_cfg.get("n_eff", -1), # -1 utilise tout x_sub
                        kNN=analysis_cfg.get("kNN", 4),
                        embedding=(3,3),
                        verbose=False,
                    )
                    te_samples.append(val_means[0])

                # 3. Aggregation des statistiques
                mean_te = np.mean(te_samples)
                std_te = np.std(te_samples)

                # 4. Stockage
                # On note lag_idx pour référencer l'échelle temporelle
                key = f"lag{lag_idx}_{u}_{v}" 
                results[f"TE_{key}"] = mean_te
                results[f"STD_{key}"] = std_te
                results[f"Theory_{key}"] = L_exp[u, v]

    return results





def research_alignment_worker_TE_tensor(params):
    """
    Simulates FHN and saves a TENSOR structure directly to disk (.npz).
    Optimized to leverage the C-Library for TE calculation over vectorised lags.
    """

    # 1. SETUP & PATHS

    model, graph_uuid = common_worker_setup(params)
    model.coupling_str = float(params['epsilon'])
    model.c_r = float(params['cr'])
    print(f"[Worker] Updated Model: c_r={model.c_r}, coupling_str={model.coupling_str}")


    run_id = params.get("run_id", "debug")
    output_dir = os.path.join(params["output_folder"], f"{run_id}_tensors")
    os.makedirs(output_dir, exist_ok=True)
    
    task_idx = params.get("task_index", np.random.randint(1e9)) 
    filename = f"te_tensor_eps{params['epsilon']:.3f}_cr{params['cr']:.2f}_{task_idx}.npz"
    filepath = os.path.join(output_dir, filename)

    if "preloaded_graph" in params:
        adj = params["preloaded_graph"]
    else:
        res_topo = load_graph_topology(params["graph_file_path"])
        adj = res_topo[0]

   
    # 2. SIMULATION
 
    total_time = int(params["total_time"])
    rng = default_rng(params.get("seed", None))
    n_nodes = params["number_of_nodes"]
    
    # Init state
    State_0 = np.zeros((3, n_nodes), dtype=np.float64)
    State_0[0] = 0.1 + 0.1 * rng.standard_normal(n_nodes)
    State_0[1] = 0.3 + 0.1 * rng.standard_normal(n_nodes)
    State_0[2] = 1.0 + 0.1 * rng.standard_normal(n_nodes)
    
    input_signal = params.get('preloaded_input_signal')
    if input_signal is None:
        raise RuntimeError("CRITICAL: 'preloaded_input_signal' not found in task parameters. Worker initialization must have failed.")
    else:
        print(f"Worker {os.getpid()}: Using preloaded input signal with shape {input_signal.shape}")

    
    # Evolution
    print(f"Worker {os.getpid()}: Evolving system for epsilon={params.get('epsilon')}, cr={params.get('cr')}")
    traj = evolve_system(model, State_0, total_time, input_signal, rk4_step)
    

    voltage_data = traj[:, 0, :] 


    # 3. PRÉPARATION ANALYSE & THÉORIE

    ac = params["research_analysis"]
    te_lags = ac.get("te_lags", [1, 5, 10, 15, 20]) 
    limit_per_layer = 10
    n_distances = 4
    

    theory_propagators = {}
    

    print(f"Pre-computing theoretical propagators for {len(te_lags)} lags...")
    for lag in te_lags:
        tau_seconds = lag * params["dt"]
        theory_propagators[lag] = expm(-model.coupling_op * tau_seconds)

    # Conteneurs
    records_te = [] 
    records_std = []    
    records_theo = []   
    records_meta = []   

    G = nx.from_numpy_array(adj)
    

    # 4. BOUCLE DE CALCUL 

    n_samples_total = voltage_data.shape[0]
    

    for src in params.get("target_indices"):
        lengths = nx.single_source_shortest_path_length(G, src, cutoff=n_distances)
        
        layers = {2: [], 3: [],4: []}
        for node, dist in lengths.items():
            if dist in layers:
                layers[dist].append(node)
        

        for dist in [2, 3, 4]:
            targets = layers[dist]
            if not targets: continue
            

            if len(targets) > limit_per_layer:
                targets = np.random.choice(targets, limit_per_layer, replace=False)
            
            for dst in targets:

                raw_x = input_signal[:, src]
                raw_y = voltage_data[:, dst]
                decimation_factor = 10 
                x_c = prepare_data(raw_x[::decimation_factor]) 
                y_c = prepare_data(raw_y[::decimation_factor])


                te_vals, te_stds = compute_te_over_lags(
                    x_c, y_c, 
                    lags=te_lags, 
                    n_real=3,         
                    n_eff= 2024,
                    verbose=False
                )
                

                for i, lag_val in enumerate(te_lags):
                    
                    val = te_vals[i]
                    std = te_stds[i]
                    theo = theory_propagators[lag_val][src, dst]
                    
                    records_te.append(val)
                    records_std.append(std) 
                    records_theo.append(theo)
                    records_meta.append([lag_val, dist, src, dst])

    mean_max_te = 0.0
    if len(records_te) > 0:
        try:
            all_te = np.array(records_te)
            n_lags = len(te_lags)
            

            te_matrix = all_te.reshape(-1, n_lags)

            max_per_couple = np.max(te_matrix, axis=1) # (n_couples,)

            mean_max_te = np.mean(max_per_couple)

            std_spatial_te = np.std(max_per_couple) 

            n_couples = len(max_per_couple)
            sem_te = std_spatial_te / np.sqrt(n_couples)
            
        except Exception as e:
            print(f"Warning: Calculation metrics failed: {e}")
            mean_max_te = 0.0
            sem_te = 0.0
            std_spatial_te = 0.0
    

    # 5. SAUVEGARDE EN .NPZ

    
    np.savez_compressed(
        filepath,
        te_values=np.array(records_te, dtype=np.float32),
        te_std=np.array(records_std, dtype=np.float32),
        theory_values=np.array(records_theo, dtype=np.float32),
        metadata=np.array(records_meta, dtype=np.int32), 
        parameters=np.array([params['epsilon'], params['cr']])
    )

    return {
        "status": "done",
        "tensor_file": filename,
        "epsilon": params['epsilon'],
        "cr": params['cr'],
        "mean_max_TE": mean_max_te,   
        "sem_max_TE": sem_te,          
        "std_spatial_TE": std_spatial_te 
    }

