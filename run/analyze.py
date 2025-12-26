#!/usr/bin/env python3

# ======================= Libraries


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
from pathlib import Path
from scipy.spatial import KDTree
from bnn_package import load_simulation_data, prepare_data, compute_te_over_lags


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

    # ee.get_sampling(verbosity=1)

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
    # 1. Load Data
    try:
        # Load trajectory (and skip graph structure for speed)
        data = load_simulation_data(file_path, graph=False)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    trajectory = data["trajectory"]
    params = data["parameters"]
    graph_uuid = data.get("graph_uuid", "unknown")

    # 2. Detect & Parse Dimensions
    # Workers.py defines state as (3, n_nodes), so trajectory is (Time, 3, n_nodes)
    shape = trajectory.shape

    if len(shape) == 3:
        # Check which dimension is likely the variables (usually 3)
        if shape[1] == 3 and shape[2] != 3:
            # Shape: (Time, Variables, Nodes) -> The scenario defined in workers.py
            n_steps, n_vars, n_nodes_total = shape

            # Helper to extract voltage (Variable 0) for a specific node
            def get_voltage(traj, node_idx):
                return traj[:, 0, node_idx]

        elif shape[2] == 3 and shape[1] != 3:
            # Shape: (Time, Nodes, Variables) -> The scenario in analyze.py
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

    # 3. Construct Time Axis
    dt = float(params.get("dt", 0.01))
    time_axis = np.arange(n_steps) * dt

    # 4. Plotting
    plt.figure(figsize=(10, 6))

    for node in target_nodes:
        if 0 <= node < n_nodes_total:
            voltage_signal = get_voltage(trajectory, node)
            plt.plot(time_axis, voltage_signal, label=f"Node {node}")
        else:
            print(
                f"Warning: Node {node} is out of bounds (Total nodes: {n_nodes_total})"
            )

    # 5. Styling
    plt.title(f"Time Series Evolution\nGraph UUID: {graph_uuid} | dt: {dt}")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage ($V_e$)")

    # Place legend outside to avoid obscuring data
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

    # ================== 1. LOAD DATA ==================
    graph_data = None
    graph_uuid = "unknown"

    # CASE A: Input is the Simulation (.h5)
    if path_obj.suffix == ".h5":
        try:
            # load_trajectory=False makes it fast
            data = load_simulation_data(file_path, graph=True, load_trajectory=False)
            graph_data = data.get("graph")
            graph_uuid = data.get("graph_uuid", "unknown")
        except Exception as e:
            print(f"Error loading .h5: {e}")
            return

    # CASE B: Input is the Graph itself (.npz)
    elif path_obj.suffix == ".npz":
        try:
            # Load directly
            raw_data = np.load(str(path_obj))
            graph_data = dict(raw_data)  # Convert to dict

            # The .npz usually contains the UUID inside (see generate_config.py)
            if "uuid" in graph_data:
                uuid_val = graph_data["uuid"]
                # Decode if bytes (common in numpy str saving)
                if isinstance(uuid_val, bytes):
                    graph_uuid = uuid_val.decode("utf-8")
                else:
                    graph_uuid = str(uuid_val)
            else:
                # If not inside, grab from filename as fallback
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

    # Fix Shape if (2, N) -> (N, 2)
    if active_pos.shape[0] == 2 and active_pos.shape[1] > 2:
        active_pos = active_pos.T

    num_active = len(active_pos)

    # Build NetworkX Graph
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

    # ================== 3. PLOTTING ==================
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

    ax.set_aspect("equal")  # Correct aspect ratio
    ax.axis("off")  # Hide axis
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
    # To see waves, we need to skip enough steps to move through time quickly
    step_skip = int(steps_per_second / fps)
    if step_skip < 1:
        step_skip = 1

    # Extract Voltage (Variable 0)
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

    # 5. Update Function (Fixed for readability)
    def update(frame):
        scatter.set_array(voltages[frame, :])

        # Calculate real simulation time
        current_time = frame * step_skip * dt
        current_step = frame * step_skip
        title.set_text(f"Sim Time: {current_time:.2f}s | Step: {current_step}")

        return (scatter,)

    # 6. Execute Animation
    # interval = 1000 / fps ensures the video plays at the desired speed
    ani = animation.FuncAnimation(
        fig, update, frames=n_frames, interval=int(1000 / fps), blit=False, repeat=True
    )

    plt.show()
    return ani


def animate_with_tracer(file_path, fps=30, steps_per_second=2000):
    # 1. Load Data (inchangé)
    data = load_simulation_data(file_path, graph=True, load_trajectory=True)
    trajectory = data["trajectory"]
    pos = data["graph"]["positions"]
    params = data["parameters"]

    if pos.shape[0] == 2:
        pos = pos.T

    # 2. Timing and Slicing (inchangé)
    dt = float(params.get("dt", 0.01))
    step_skip = max(1, int(steps_per_second / fps))
    voltages = trajectory[::step_skip, 0, :]
    full_time_axis = np.arange(trajectory.shape[0]) * dt

    v_min, v_max = np.min(voltages), np.max(voltages)

    # 3. Setup Figure avec colorbar partagée
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3)

    ax_sim = fig.add_subplot(gs[0])
    ax_trace = fig.add_subplot(gs[1])

    # Animation Plot avec colormap "plasma"
    scatter = ax_sim.scatter(
        pos[:, 0],
        pos[:, 1],
        c=voltages[0, :],
        cmap='viridis',  # Meilleur pour les contrastes forts
        s=60,
        edgecolors="white",
        linewidths=0.3,
        vmin=v_min,
        vmax=v_max,
    )

    # Ajout de la colorbar pour l'animation
    cbar = fig.colorbar(scatter, ax=ax_sim, shrink=0.8, label="Voltage ($V_e$)")
    cbar.ax.tick_params(labelsize=10)

    ax_sim.set_aspect("equal")
    ax_sim.axis("off")
    title = ax_sim.set_title("Click a Node to Trace", fontsize=12, pad=20)

   
    current_node = 0
    (line,) = ax_trace.plot(
        full_time_axis,
        trajectory[:, 0, current_node],
        color="#1f77b4",  # Bleu clair pour la ligne
        lw=2,
        label=f"Node {current_node}",
    )
    time_marker = ax_trace.axvline(
        0, color="#d62728", linestyle="--", alpha=0.7, label="Current Time"
    )  # Rouge pour le marqueur
    ax_trace.set_title(f"Voltage Trace: Node {current_node}", fontsize=12)
    ax_trace.set_xlabel("Time (s)", fontsize=10)
    ax_trace.set_ylabel("Voltage ($V_e$)", fontsize=10)
    ax_trace.grid(True, alpha=0.3, linestyle="--")
    ax_trace.set_ylim(v_min - 0.1, v_max + 0.1)
    ax_trace.legend(fontsize=9)



    

    # 4. Interactive Click Logic (inchangé)
    tree = KDTree(pos)
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

    # 6. Run Animation
    ani = animation.FuncAnimation(
        fig, update, frames=voltages.shape[0], interval=int(1000 / fps), blit=False
    )

    plt.tight_layout()
    plt.show()
    return ani

def plot_interactive_fft(file_path, min_power=100.0, f_min=0.01):
    """
    Visualisation interactive de la FFT.
    - Gauche: Carte des fréquences du réseau.
    - Droite: Spectre de puissance du noeud sélectionné.
    
    Respecte scrupuleusement la logique de 'detect_oscillating_nodes'.
    """
    
    # --- 1. CHARGEMENT ET PRÉPARATION ---
    data = load_simulation_data(file_path, graph=True, load_trajectory=True)
    trajectory = data["trajectory"]  # shape (n_time, n_vars, n_nodes)
    pos = data["graph"]["positions"]
    params = data["parameters"]
    dt = float(params.get("dt", 0.01))

    # On ne s'intéresse qu'à la première variable (ex: Voltage)
    X = trajectory[:, 0, :] 
    n_time, n_nodes = X.shape

    # Correction dimension positions
    if pos.shape[0] == 2:
        pos = pos.T

    # --- 2. PRE-CALCUL GLOBAL (Pour colorer le graphe) ---
    # On utilise exactement ta logique pour calculer les fréquences globales
    print("Calcul des fréquences globales...")
    X_centered = X - np.mean(X, axis=0)
    fft_vals = np.abs(np.fft.rfft(X_centered, axis=0))**2
    freqs_axis = np.fft.rfftfreq(n_time, d=dt)
    
    # Filtrage f_min
    valid_idx = freqs_axis >= f_min
    restricted_power = fft_vals[valid_idx, :]
    restricted_freqs = freqs_axis[valid_idx]
    
    # Pics
    peak_indices = np.argmax(restricted_power, axis=0)
    max_powers = restricted_power[peak_indices, np.arange(n_nodes)]
    peak_freqs = restricted_freqs[peak_indices]
    
    # Masque oscillant
    is_osc = max_powers > min_power
    
    # Couleurs des noeuds : NaN (Gris) si pas osc, sinon fréquence
    node_colors = np.full(n_nodes, np.nan)
    node_colors[is_osc] = peak_freqs[is_osc]

    # --- 3. MISE EN PLACE DE LA FIGURE ---
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2], wspace=0.2)
    
    ax_graph = fig.add_subplot(gs[0])
    ax_fft = fig.add_subplot(gs[1])

    # --- GRAPHE (PANNEAU DE GAUCHE) ---
    # On utilise une colormap 'plasma' pour les fréquences, et gris pour le bruit
    cmap = plt.cm.plasma
    cmap.set_bad(color='lightgray') # Les non-oscillants en gris
    
    scatter = ax_graph.scatter(
        pos[:, 0], pos[:, 1],
        c=node_colors,
        cmap=cmap,
        s=60, edgecolors='white', linewidth=0.5,
        picker=True # Active le picking natif matplotlib en backup
    )
    plt.colorbar(scatter, ax=ax_graph, label="Fréquence Dominante (Hz)")
    ax_graph.set_title("Carte des Fréquences (Click to inspect)", fontsize=14)
    ax_graph.set_aspect('equal')
    ax_graph.axis('off')

    # --- LOGIQUE D'AFFICHAGE FFT (PANNEAU DE DROITE) ---
    trace_line, = ax_fft.plot([], [], color='#1f77b4', lw=1.5, label='Power Spectrum')
    peak_point, = ax_fft.plot([], [], 'rx', markersize=10, markeredgewidth=2, label='Peak')
    threshold_line = ax_fft.axhline(min_power, color='k', linestyle='--', alpha=0.5, label=f'Min Power ({min_power})')
    
    
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("Power (|FFT|^2)")
    ax_fft.grid(True, which='both', alpha=0.3)
    ax_fft.legend()

    # Arbre KD pour le clic rapide
    tree = KDTree(pos)

    def update_fft_plot(node_idx):
        """Recalcule et affiche la FFT pour un noeud spécifique"""
        # 1. Calcul local (identique au calcul global mais pour 1 noeud)
        signal = X[:, node_idx]
        sig_centered = signal - np.mean(signal)
        
        fft_spectrum = np.fft.rfft(sig_centered)
        power = np.abs(fft_spectrum)**2
        freqs = np.fft.rfftfreq(n_time, d=dt)
        
        # 2. Mise à jour des données du plot
        # On affiche tout le spectre, mais on zoomera sur la partie utile
        trace_line.set_data(freqs, power)
        
        # 3. Trouver le pic pour l'affichage visuel
        valid_mask = freqs >= f_min
        if np.any(valid_mask):
            sub_power = power[valid_mask]
            sub_freqs = freqs[valid_mask]
            
            idx_max = np.argmax(sub_power)
            peak_p = sub_power[idx_max]
            peak_f = sub_freqs[idx_max]
            
            peak_point.set_data([peak_f], [peak_p])
            
            # Titre dynamique
            status = "OSCILLATING" if peak_p > min_power else "NOISE / SILENT"
            color_status = "green" if peak_p > min_power else "gray"
            ax_fft.set_title(f"Node {node_idx} | Peak: {peak_f:.3f} Hz | Power: {peak_p:.1f}\nStatus: {status}", 
                             fontsize=12, color=color_status, fontweight='bold')
            
            # Ajustement des limites des axes pour bien voir le pic
            ax_fft.set_xlim(0, max(peak_f * 2, 0.5)) # On affiche jusqu'à 2x la fréquence du pic
            # Y limite : un peu au dessus du pic, ou au moins au dessus du seuil
            ymax = max(peak_p * 1.1, min_power * 1.5)
            ax_fft.set_ylim(-0.05 * ymax, ymax)
            
        else:
            ax_fft.set_title(f"Node {node_idx} : Signal invalid (Frequency too low)")

        fig.canvas.draw_idle()

    # --- GESTION DU CLIC ---
    def on_click(event):
        if event.inaxes != ax_graph:
            return
        
        # Trouver le noeud le plus proche
        dists, idx = tree.query([event.xdata, event.ydata])
        # Petit seuil de distance pour éviter les clics fantomes loin du graphe
        # (Optionnel, ici on prend toujours le plus proche)
        
        update_fft_plot(idx)

    fig.canvas.mpl_connect('button_press_event', on_click)

    # Initialisation sur le noeud 0
    update_fft_plot(0)
    
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Remplace par un chemin valide
    file = "Data_output/20251225-130416_paper_config_for_CS_Diff/ts_N1000_Coup3.500_cr1.500_G-3c6fab93.h5"
    plot_interactive_fft(file, min_power=1.0e8, f_min=0.01)
    animate_with_tracer(file, fps=30)


