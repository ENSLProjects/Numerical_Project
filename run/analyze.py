#!/usr/bin/env python3

# ======================= Libraries


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from scipy.spatial import KDTree
from bnn_package import load_simulation_data, prepare_data, compute_te_over_lags, detect_oscillating_nodes
import sys
import os
import re
import pandas as pd
from scipy.optimize import curve_fit
import glob


# ======================= Functions


def measure_te(file_path):
    Full_Data = load_simulation_data(file_path)
    if Full_Data is None:
        raise ValueError(f"Failed to load data from {file_path}")

    Trajectory = Full_Data["trajectory"]
    Parameters_simu = Full_Data["parameters"]

    print("\n Parameters of the analyzed simulation", Parameters_simu)

    Parameters_model = Parameters_simu["parameters_model"]
    final_time = Parameters_simu["time length simulation"]
    rk4_time = Parameters_model["time_step rk4"]

    real_time = rk4_time * np.arange(final_time)

    fig, ax = plt.subplots()
    ax.plot(real_time, Trajectory[:, 4, 0], label=f"Node {4}")
    ax.plot(real_time, Trajectory[:, 5, 0], label=f"Node {5}")
    ax.plot(real_time, Trajectory[:, 6, 0], label=f"Node {6}")
    ax.plot(real_time, Trajectory[:, 10, 0], label=f"Node {10}")
    ax.plot(real_time, Trajectory[:, 100, 0], label=f"Node {100}")
    ax.plot(real_time, Trajectory[:, 500, 0], label=f"Node {500}")
    ax.plot(real_time, Trajectory[:, 930, 0], label=f"Node {930}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage $V_e$")
    ax.legend()
    ax.grid(True)
    plt.show()

    # ======================= ENTROPY

    # ------------ parameters
    node_source = 4
    node_target = 5
    kNN = 5
    n_embeded = 2
    N_eff = 8192  # number of points on each subset measurement
    N_real = 5  # number of subsets, for 20 the std is already of order 1e-16
    # ------------- data

 

    LAGS = np.arange(1, 10001, 50, dtype=int)

    # --- RUN ---
    try:
        x, y = (
            prepare_data(Trajectory[:, node_source, 0]),
            prepare_data(Trajectory[:, node_target, 0]),
        )
        # for now we only care about the tension of the active nodes

        te_mean, te_std = compute_te_over_lags(
            x,
            y,
            LAGS,
            n_real=N_real,
            n_eff=N_eff,
            kNN=kNN,
            embedding=(n_embeded, n_embeded),
        )

        # --- PLOT (with Error Bars) ---
        plt.figure(figsize=(10, 6))

        # Plot with error bars
        plt.errorbar(
            LAGS,
            te_mean,
            yerr=te_std,
            fmt="-x",
            color="crimson",
            ecolor="gray",
            capsize=3,
            label=f"TE({node_source} $\\to$ {node_target})",
        )

        plt.title(
            f"Transfer Entropy Lag Profile\nNumber of points used = {N_eff}, Number of calculation = {N_real}"
        )
        plt.xlabel("Lag $\\tau$ (time steps)")
        plt.ylabel("Transfer Entropy (nats)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.show()

    except Exception as e:
        print(f"Error: {e}")


def plot_simulation_nodes(file_path, target_nodes):
    """
    Plots the voltage evolution for a given set of nodes.
    Correctly handles shape (Time, Variables, Nodes).
    """

    try:
        data = load_simulation_data(file_path, graph=False)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    trajectory = data["trajectory"]
    params = data["parameters"]
    graph_uuid = data.get("graph_uuid", "unknown")


    shape = trajectory.shape

    if len(shape) == 3:
        if shape[1] == 3 and shape[2] != 3:
            n_steps, n_vars, n_nodes_total = shape

            def get_voltage(traj, node_idx):
                return traj[:, 0, node_idx]

        elif shape[2] == 3 and shape[1] != 3:

            n_steps, n_nodes_total, n_vars = shape

            def get_voltage(traj, node_idx):
                return traj[:, node_idx, 0]

        else:
            print(f"Warning: Ambiguous shape {shape}. Assuming (Time, Vars, Nodes).")
            n_steps, n_vars, n_nodes_total = shape

            def get_voltage(traj, node_idx):
                return traj[:, 0, node_idx]
    else:
        print(f"Error: Unexpected data shape {shape}. Expected 3 dimensions.")
        return


    dt = float(params.get("dt", 0.01))
    time_axis = np.arange(n_steps) * dt

    plt.figure(figsize=(10, 6))

    for node in target_nodes:
        if 0 <= node < n_nodes_total:
            voltage_signal = get_voltage(trajectory, node)
            plt.plot(time_axis, voltage_signal, label=f"Node {node}")
        else:
            print(
                f"Warning: Node {node} is out of bounds (Total nodes: {n_nodes_total})"
            )


    plt.title(f"Time Series Evolution\nGraph UUID: {graph_uuid} | dt: {dt}")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage ($V_e$)")

    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0)

    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.show()


def plot_simulation_graph(file_path):
    """
    Plots the graph topology.
    Accepts either:
      1. A simulation file (.h5) -> Finds and plots the linked graph.
      2. A graph file (.npz) -> Plots the graph directly.
    """
    path_obj = Path(file_path).resolve()
    if not path_obj.exists():
        print(f"Error: File not found at {file_path}")
        return


    graph_data = None
    graph_uuid = "unknown"


    if path_obj.suffix == ".h5":
        try:

            data = load_simulation_data(file_path, graph=True, load_trajectory=False)
            graph_data = data.get("graph")
            graph_uuid = data.get("graph_uuid", "unknown")
        except Exception as e:
            print(f"Error loading .h5: {e}")
            return


    elif path_obj.suffix == ".npz":
        try:

            raw_data = np.load(str(path_obj))
            graph_data = dict(raw_data)  


            if "uuid" in graph_data:
                uuid_val = graph_data["uuid"]

                if isinstance(uuid_val, bytes):
                    graph_uuid = uuid_val.decode("utf-8")
                else:
                    graph_uuid = str(uuid_val)
            else:

                graph_uuid = path_obj.stem.split("_")[-1]

        except Exception as e:
            print(f"Error loading .npz: {e}")
            return

    else:
        print(f"Error: Unsupported file format {path_obj.suffix}. Use .h5 or .npz")
        return

    # Validate Data
    if graph_data is None:
        print("Error: Could not extract graph data.")
        return

    required = ["adjacency", "positions", "passive_counts"]
    if not all(k in graph_data for k in required):
        print(
            f"Error: Graph file missing keys. Found: {list(graph_data.keys())}, Needed: {required}"
        )
        return

    # ================== 2. PROCESS GRAPH ==================
    adj_matrix = graph_data["adjacency"]
    active_pos = graph_data["positions"]
    passive_counts = graph_data["passive_counts"]


    if active_pos.shape[0] == 2 and active_pos.shape[1] > 2:
        active_pos = active_pos.T

    num_active = len(active_pos)


    G = nx.Graph()
    active_indices = list(range(num_active))
    G.add_nodes_from(active_indices, type="active")

    rows, cols = np.where(adj_matrix > 0)
    active_edges = [(i, j) for i, j in zip(rows, cols) if i < j]
    G.add_edges_from(active_edges, type="active_link")

    pos_dict = {i: active_pos[i] for i in range(num_active)}
    passive_indices = []
    passive_edges = []
    current_idx = num_active
    satellite_radius = 0.25

    # Generate Passive Nodes
    for i in range(num_active):
        n_passive = int(passive_counts[i])
        if n_passive > 0:
            angles = np.linspace(0, 2 * np.pi, n_passive, endpoint=False)
            center_x, center_y = active_pos[i]
            for angle in angles:
                px = center_x + satellite_radius * np.cos(angle)
                py = center_y + satellite_radius * np.sin(angle)
                G.add_node(current_idx, type="passive")
                pos_dict[current_idx] = np.array([px, py])
                passive_indices.append(current_idx)
                passive_edges.append((i, current_idx))
                current_idx += 1

    G.add_edges_from(passive_edges, type="passive_link")

    # ==================  PLOTTING ==================
    plt.figure(figsize=(12, 12))
    ax = plt.gca()

    # Draw Connections
    nx.draw_networkx_edges(
        G,
        pos_dict,
        edgelist=active_edges,
        edge_color="gray",
        alpha=0.3,
        width=1.0,
        ax=ax,
    )
    nx.draw_networkx_edges(
        G,
        pos_dict,
        edgelist=passive_edges,
        edge_color="#e74c3c",
        alpha=0.4,
        style="dotted",
        width=0.8,
        ax=ax,
    )

    # Draw Nodes
    nx.draw_networkx_nodes(
        G,
        pos_dict,
        nodelist=active_indices,
        node_color="#3498db",
        node_size=50,
        edgecolors="white",
        linewidths=0.5,
        label="Active Nodes",
        ax=ax,
    )
    if passive_indices:
        nx.draw_networkx_nodes(
            G,
            pos_dict,
            nodelist=passive_indices,
            node_color="#e74c3c",
            node_size=20,
            alpha=0.8,
            label="Passive Nodes",
            ax=ax,
        )

    # Styling
    plt.title(
        f"Graph Topology\nUUID: {graph_uuid} | Active: {num_active} | Passive: {len(passive_indices)}"
    )

    ax.set_aspect("equal")  
    ax.axis("off") 
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def animate_simulation(file_path, fps=30, steps_per_second=2000):
    """
    Optimized animation to reveal waves by matching playback speed
    to physical oscillation frequency.

    Args:
        file_path (str): Path to .h5 file.
        fps (int): Frames per second of the animation playback.
        steps_per_second (int): How many simulation steps to 'burn' per 1s of video.
                               Increase this to make the waves move faster.
    """
    # 1. Load Data
    data = load_simulation_data(file_path, graph=True, load_trajectory=True)
    trajectory = data["trajectory"]  # Shape: (Time, Variables, Nodes)
    graph_data = data["graph"]
    params = data["parameters"]

    # 2. Calculate optimal skipping
    step_skip = int(steps_per_second / fps)
    if step_skip < 1:
        step_skip = 1


    voltages = trajectory[::step_skip, 0, :]
    pos = graph_data["positions"]
    if pos.shape[0] == 2:
        pos = pos.T

    n_frames = voltages.shape[0]
    dt = float(params.get("dt", 0.01))

    # 3. Dynamic Range
    v_min, v_max = np.min(voltages), np.max(voltages)

    # 4. Setup Figure
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        pos[:, 0],
        pos[:, 1],
        c=voltages[0, :],
        cmap="magma",
        s=40,
        edgecolors="black",
        linewidths=0.2,
        vmin=v_min,
        vmax=v_max,
    )

    plt.colorbar(scatter, label="Active Voltage ($V_e$)")
    ax.set_aspect("equal")
    ax.axis("off")
    title = ax.set_title("Initializing Waves...")

    # 5. Update Function
    def update(frame):
        scatter.set_array(voltages[frame, :])
        current_time = frame * step_skip * dt
        current_step = frame * step_skip
        title.set_text(f"Sim Time: {current_time:.2f}s | Step: {current_step}")

        return (scatter,)

    # 6. Execute Animation
    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=int(1000 / fps), blit=False, repeat=True
    )

    plt.show()
    return ani


def animate_with_tracer(file_path, fps=30, steps_per_second=4000):
    """
    Advanced animation with a clickable node tracer.
    """
    # 1. Load Data
    data = load_simulation_data(file_path, graph=True, load_trajectory=True)
    trajectory = data["trajectory"]  
    pos = data["graph"]["positions"]
    params = data["parameters"]

    if pos.shape[0] == 2:
        pos = pos.T

    tree = KDTree(pos)

    # 2. Timing and Slicing
    dt = float(params.get("dt", 0.01))
    step_skip = max(1, int(steps_per_second / fps))
    voltages = trajectory[::step_skip, 0, :]
    full_time_axis = np.arange(trajectory.shape[0]) * dt

    v_min, v_max = np.min(voltages), np.max(voltages)

    # 3. Setup Figure 
    fig = plt.figure(figsize=(15, 7))
    ax_sim = fig.add_subplot(121)
    ax_trace = fig.add_subplot(122)

    # Animation Plot
    scatter = ax_sim.scatter(
        pos[:, 0],
        pos[:, 1],
        c=voltages[0, :],
        cmap="magma",
        s=40,
        edgecolors="black",
        linewidths=0.2,
        vmin=v_min,
        vmax=v_max,
    )
    ax_sim.set_aspect("equal")
    ax_sim.axis("off")
    title = ax_sim.set_title("Click a node to trace")

    # Trace Plot
    current_node = 0
    (line,) = ax_trace.plot(
        full_time_axis, trajectory[:, 0, current_node], color="crimson", lw=1.5
    )
    time_marker = ax_trace.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax_trace.set_title(f"Voltage Trace: Node {current_node}")
    ax_trace.set_xlabel("Time (s)")
    ax_trace.set_ylabel("Voltage ($V_e$)")
    ax_trace.grid(True, alpha=0.3)
    ax_trace.set_ylim(v_min - 0.1, v_max + 0.1)

    # 4. Interactive Click Logic
    def on_click(event):
        nonlocal current_node
        if event.inaxes != ax_sim:
            return


        dists, idx = tree.query([event.xdata, event.ydata])
        current_node = idx

        line.set_ydata(trajectory[:, 0, current_node])
        ax_trace.set_title(f"Voltage Trace: Node {current_node}")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)

    # 5. Animation Update
    def update(frame):
        scatter.set_array(voltages[frame, :])

        current_time = frame * step_skip * dt
        time_marker.set_xdata([current_time])

        title.set_text(f"Time: {current_time:.2f}s | Node: {current_node}")
        return scatter, time_marker, title

    # 6. Run
    ani = animation.FuncAnimation(
        fig, update, frames=voltages.shape[0], interval=int(1000 / fps), blit=False
    )

    plt.tight_layout()
    plt.show()
    return ani


# ======================= 1. CURVE FITTING MODELS =======================


def power_law(x, a, k, c):
    """The Candidate Law: TE = a * (Propagator)^k + c"""
    return a * np.power(x, k) + c


def linear_model(x, a, b):
    """The Linear Channel: TE = a * x + b"""
    return a * x + b


# ======================= 2. ANALYSIS MODES =======================


def analyze_deep_dive(folder_path):
    """
    Scans folder for 'scatter_*.npz' and performs Curve Fitting + Binning.
    """
    print(f"\n>>> [Deep Dive] Scanning folder: {folder_path}")
    files = glob.glob(os.path.join(folder_path, "scatter_*.npz"))

    if not files:
        print(f"Error: No 'scatter_*.npz' files found in {folder_path}")
        return

    print(f"    Found {len(files)} scatter files. Analyzing...")

    for file_path in files:
        try:
            filename = os.path.basename(file_path)
            data = np.load(file_path)

            if "theory" in data and "te" in data:
                x_theory, y_te = data["theory"], data["te"]
            else:
                print(f"    [Skip] {filename} missing keys.")
                continue

            # Filter Noise
            mask = (x_theory > 1e-9) & (y_te > 1e-9)
            x, y = x_theory[mask], y_te[mask]

            if len(x) < 10:
                print("    [Skip] Not enough data points.")
                continue

            # --- 1. CURVE FITTING ---
            try:
                popt_pow, _ = curve_fit(power_law, x, y, p0=[1, 2, 0], maxfev=5000)
                k = popt_pow[1]
            except Exception:
                k, popt_pow = 0.0, [0, 0, 0]

            # --- 2. BINNED STATISTICS (The Noise Killer) ---
            # We split the x-axis (Propagator) into 15 bins and average the TE in each.
            nbins = 15
            bins = np.linspace(np.min(x), np.max(x), nbins + 1)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            bin_means = []
            bin_errs = []

            for i in range(nbins):
                # Find all points in this slice of X
                idx = (x >= bins[i]) & (x < bins[i + 1])
                if np.sum(idx) > 5:  # Only compute if enough samples
                    mean_val = np.mean(y[idx])
                    std_err = np.std(y[idx]) / np.sqrt(np.sum(idx))
                    bin_means.append(mean_val)
                    bin_errs.append(std_err)
                else:
                    bin_means.append(np.nan)
                    bin_errs.append(np.nan)

            bin_means = np.array(bin_means)
            bin_errs = np.array(bin_errs)

            # --- 3. PLOTTING ---
            plt.figure(figsize=(10, 6))

            # A. The Cloud (Raw)
            plt.scatter(x, y, alpha=0.15, s=10, c="gray", label="Raw Pairs (Noise)")

            # B. The Signal (Binned) - THE TRUTH
            valid_bins = ~np.isnan(bin_means)
            plt.errorbar(
                bin_centers[valid_bins],
                bin_means[valid_bins],
                yerr=bin_errs[valid_bins],
                fmt="o-",
                color="green",
                lw=2,
                capsize=4,
                label="Binned Average (Signal)",
            )

            # C. The Fits
            x_sort = np.sort(x)
            if k != 0:
                plt.plot(
                    x_sort,
                    power_law(x_sort, *popt_pow),
                    "r--",
                    lw=2,
                    label=f"Power Law ($k={k:.2f}$)",
                )

            plt.xlabel(r"Theoretical Propagator $(e^{-\mathcal{L}\tau})_{ij}$")
            plt.ylabel("Transfer Entropy (TE)")
            plt.title(
                f"Structure-Function Relationship ({filename})\nExponent k={k:.4f}"
            )
            plt.legend()
            plt.grid(True, alpha=0.3)

            out_name = os.path.join(
                folder_path, filename.replace(".npz", "_binned.png")
            )
            plt.savefig(out_name, dpi=300)
            plt.close()

            print(
                f"    --> Saved Binned Plot: {os.path.basename(out_name)} (k={k:.4f})"
            )

        except Exception as e:
            print(f"    [Error] {filename}: {e}")


# ======================= 3. VISUALIZATION FUNCTIONS (CSV) =======================


def parse_columns(df):
    """Extracts metrics and lags from column names."""
    metrics = {}
    pattern = re.compile(r"(.+)_lag(\d+)$")
    for col in df.columns:
        match = pattern.match(col)
        if match:
            name = match.group(1)
            lag = int(match.group(2))
            if name not in metrics:
                metrics[name] = {}
            metrics[name][lag] = col
    return metrics


def plot_phase_scan(df, metrics_map, output_prefix):
    """
    Plots Metric vs Epsilon (The 'U-Shape' finder).
    """
    print(">>> Detected PHASE SCAN mode (Multiple Epsilons).")

    df = df.sort_values(by="epsilon")
    for metric_name, lag_dict in metrics_map.items():
        plt.figure(figsize=(10, 6))

        sorted_lags = sorted(lag_dict.keys())
        for lag in sorted_lags:
            col = lag_dict[lag]
            plt.plot(df["epsilon"], df[col], marker="o", label=f"Lag {lag}")

        plt.xscale("log")
        plt.xlabel(r"Coupling Strength $\epsilon$")
        plt.ylabel(metric_name.replace("_", " ").title())
        plt.title(f"Phase Scan: {metric_name}")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.savefig(f"{output_prefix}_{metric_name}_scan.png", dpi=300)
        plt.close()


def plot_time_evolution(df, metrics_map, output_prefix):
    """
    Plots Metric vs Lag (The 'Final Proof' time evolution).
    """
    print(">>> Detected TIME EVOLUTION mode (Single/Few Epsilons).")


    for idx, row in df.iterrows():
        eps = row.get("epsilon", "unknown")

        for metric_name, lag_dict in metrics_map.items():
            lags = sorted(lag_dict.keys())
            values = [row[lag_dict[lag]] for lag in lags]

            plt.figure(figsize=(10, 6))


            plt.plot(
                lags,
                values,
                "o-",
                linewidth=2,
                color="crimson",
                label=rf"$\epsilon={eps}$",
            )

            if "cca" in metric_name.lower():
                pass
            elif "kl" in metric_name.lower():
                plt.axhline(0.0, color="black", linestyle="--", label="Zero Divergence")

            plt.xlabel("Lag $\\tau$")
            plt.ylabel(metric_name.replace("_", " ").title())
            plt.title(rf"Dynamics over Time ($\epsilon={eps}$)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(
                f"{output_prefix}_{metric_name}_eps{eps}_evolution.png", dpi=300
            )
            plt.close()


def analyze_csv(csv_path):
    df = pd.read_csv(csv_path)
    metrics_map = parse_columns(df)
    if not metrics_map:
        print("Error: No 'metric_lagX' columns found in CSV.")
        return

    # 2. Determine Plot Mode

    unique_eps = df["epsilon"].nunique() if "epsilon" in df.columns else 0
    output_prefix = os.path.splitext(csv_path)[0]

    if unique_eps > 3:
        plot_phase_scan(df, metrics_map, output_prefix)
    else:
        plot_time_evolution(df, metrics_map, output_prefix)


def animate_with_tracer_FFT(file_path, fps=30, steps_per_second=4000):
    """
    Animation avec :
    1. Graphe du réseau (gauche)
    2. Tracé temporel du nœud cliqué (milieu)
    3. Spectre FFT du nœud sélectionné (droite)
    """
    # 1. Chargement des données
    data = load_simulation_data(file_path, graph=True, load_trajectory=True)
    trajectory = data["trajectory"]  
    pos = data["graph"]["positions"]
    params = data["parameters"]

    if pos.shape[0] == 2:
        pos = pos.T

    # 2. Configuration temporelle
    dt = float(params.get("dt", 0.01))
    step_skip = max(1, int(steps_per_second / fps))
    voltages = trajectory[::step_skip, 0, :]  
    full_time_axis = np.arange(trajectory.shape[0]) * dt
    n_time_full = trajectory.shape[0]

    # 3. Pré-calcul des FFT pour tous les nœuds (optimisation)
    X_centered = trajectory[:, 0, :] - np.mean(trajectory[:, 0, :], axis=0)
    fft_spectrum = np.fft.rfft(X_centered, axis=0)
    power_spectrum = np.abs(fft_spectrum)**2
    freqs = np.fft.rfftfreq(n_time_full, d=dt)

    # 4. Configuration de la figure (3 colonnes)
    fig = plt.figure(figsize=(18, 7))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.8])

    # 4.1. Graphe du réseau (gauche)
    ax_sim = fig.add_subplot(gs[0])
    scatter = ax_sim.scatter(
        pos[:, 0], pos[:, 1],
        c=voltages[0, :],
        cmap="magma",
        s=40, edgecolors="black", linewidths=0.2,
        vmin=np.min(voltages), vmax=np.max(voltages)
    )
    ax_sim.set_aspect("equal")
    ax_sim.axis("off")
    ax_sim.set_title("Réseau - Cliquez sur un nœud")

    # 4.2. Tracé temporel (milieu)
    ax_trace = fig.add_subplot(gs[1])
    current_node = 0
    (line,) = ax_trace.plot([], [], color="crimson", lw=1.5)
    time_marker = ax_trace.axvline(0, color="black", linestyle="--", alpha=0.5)
    ax_trace.set_title(f"Série temporelle : Nœud {current_node}")
    ax_trace.set_xlabel("Temps (s)")
    ax_trace.set_ylabel("Potentiel")
    ax_trace.grid(True, alpha=0.3)

    # 4.3. Spectre FFT (droite)
    ax_fft = fig.add_subplot(gs[2])
    (fft_line,) = ax_fft.plot([], [], color="darkblue", lw=2)
    ax_fft.set_title(f"Spectre FFT : Nœud {current_node}")
    ax_fft.set_xlabel("Fréquence (Hz)")
    ax_fft.set_ylabel("Puissance")
    ax_fft.grid(True, alpha=0.3)
    ax_fft.set_xlim(0, 50)  # Limite à 50Hz (ajustable)

    # 5. Logique de clic interactif
    tree = KDTree(pos)
    def on_click(event):
        nonlocal current_node
        if event.inaxes != ax_sim:
            return
        dists, idx = tree.query([event.xdata, event.ydata])
        current_node = idx
        update_node_data(current_node)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", on_click)

    # 6. Fonction de mise à jour des données
    def update_node_data(node_idx):
        # Mise à jour du tracé temporel
        line.set_data(full_time_axis, trajectory[:, 0, node_idx])
        ax_trace.set_title(f"Série temporelle : Nœud {node_idx}")
        ax_trace.relim()
        ax_trace.autoscale_view()

        # Mise à jour de la FFT
        node_power = power_spectrum[:, node_idx]
        node_freqs = freqs
        fft_line.set_data(node_freqs, node_power)
        ax_fft.set_title(f"Spectre FFT : Nœud {node_idx}")

        # Détection des oscillations
        is_osc, peak_freq = detect_oscillating_nodes(
            trajectory[:, 0, node_idx].reshape(-1, 1),
            params,
            min_power=1e8,  
            f_min=0.00005
        )

        # Annotations sur le spectre
        if is_osc[0]:
            ax_fft.axvline(peak_freq[0], color="red", linestyle="--",
                          label=f"Pic: {peak_freq[0]:.2f} Hz")
            ax_fft.legend()
        else:
            ax_fft.set_title(f"Spectre FFT : Nœud {node_idx} (Non oscillant)")

        ax_fft.relim()
        ax_fft.autoscale_view()

    # 7. Initialisation
    update_node_data(current_node)

    # 8. Animation
    def animate(frame):
        scatter.set_array(voltages[frame, :])
        current_time = frame * step_skip * dt
        time_marker.set_xdata([current_time, current_time])

        return scatter, line, time_marker, fft_line

    # Calcul du nombre de frames
    n_frames = voltages.shape[0]
    ani = animation.FuncAnimation(
        fig, animate, frames=n_frames,
        interval=1000/fps, blit=False
    )

    plt.tight_layout()
    plt.show()
    return ani


if __name__ == "__main__":
    plot_simulation_graph("/Users/constantindeumier/Desktop/Dev_Numerical_project/pass/Numerical_Project/run/Data_output/20260101-164453_simulation_optimal_linear_cr_epsilon/ts_N1000_Coup0.250_cr1.000_G-unknown.h5")
    animate_with_tracer_FFT("/Users/constantindeumier/Desktop/Dev_Numerical_project/pass/Numerical_Project/run/Data_output/20260104-100620_simulation_optimal_linear_cr_epsilon/ts_N1000_Coup0.800_cr1.200_G-unknown.h5")
