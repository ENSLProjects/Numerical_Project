#!/usr/bin/env/python3
import h5py
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from bnn_package import get_data


def visualize_evolved_brain(file_path="data_simulation/Evolved_Canyon_Brain.h5"):
    """
    Loads the evolved graph and plots the active pathways.
    """
    print(f">>> Loading brain from {file_path}...")

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        return

    # 1. LOAD DATA
    with h5py.File(file_path, "r") as f:
        adj_matrix = get_data(f, "adjacency")
        pos_array = get_data(f, "positions")
        accuracy = f.attrs.get("accuracy", 0.0)
        generations = f.attrs.get("generations", 0)

    # 2. SETUP GRAPH
    N = adj_matrix.shape[0]
    G = nx.Graph()

    # Add nodes with positions
    # pos_array is (2, N), we need a dict {node_id: (x, y)}
    pos_dict = {i: pos_array[:, i] for i in range(N)}

    # Identify Node Types based on X-coordinate (Robust method)
    # X < 1.0 -> Input, X > 9.0 -> Output, Else -> Hidden
    node_colors = []
    for i in range(N):
        x_coord = pos_dict[i][0]
        if x_coord < 1.0:
            node_colors.append("#32CD32")  # Lime Green (Input)
            G.add_node(i, type="Input")
        elif x_coord > 9.0:
            node_colors.append("#FF4500")  # Orange Red (Output)
            G.add_node(i, type="Output")
        else:
            node_colors.append("#1E90FF")  # Dodger Blue (Hidden)
            G.add_node(i, type="Hidden")

    # 3. ADD EDGES (FILTERING)
    # We only draw edges that are strong enough to matter
    threshold = 0.05  # Ignore tiny "ghost" weights

    edges_to_draw = []
    weights_to_draw = []

    rows, cols = np.triu_indices(N, k=1)
    for r, c in zip(rows, cols):
        w = adj_matrix[r, c]
        if w > threshold:
            G.add_edge(r, c, weight=w)
            edges_to_draw.append((r, c))
            weights_to_draw.append(w * 3.0)  # Thickness multiplier

    print(f"    Nodes: {N}")
    print(f"    Active Edges: {len(edges_to_draw)} (Threshold > {threshold})")

    # 4. PLOT
    plt.figure(figsize=(12, 6))

    # Draw Nodes
    nx.draw_networkx_nodes(
        G, pos_dict, node_color=node_colors, node_size=300, edgecolors="black"
    )

    # Draw Edges (Width = Weight)
    nx.draw_networkx_edges(
        G,
        pos_dict,
        edgelist=edges_to_draw,
        width=weights_to_draw,
        alpha=0.6,
        edge_color="gray",
    )

    # Draw Labels (Optional, small font)
    nx.draw_networkx_labels(G, pos_dict, font_size=8, font_color="white")

    # Annotations
    plt.title(
        f"Evolved Structure (Gen {generations}) | Accuracy: {accuracy:.1%}", fontsize=14
    )
    plt.xlabel("Physical Distance (Signal must cross from Left to Right)")

    # Custom Legend
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#32CD32",
            markersize=10,
            label="Input Zone",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#1E90FF",
            markersize=10,
            label="Bridge (Hidden)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#FF4500",
            markersize=10,
            label="Readout Zone",
        ),
        Line2D([0], [0], color="gray", lw=2, label="Synaptic Weight"),
    ]
    plt.legend(handles=legend_elements, loc="lower right")

    plt.axis("equal")  # Keep aspect ratio so the "Canyon" looks real
    plt.grid(True, linestyle=":", alpha=0.3)

    save_path = "brain_visualization.png"
    plt.savefig(save_path, dpi=300)
    print(f">>> [PLOT SAVED] {save_path}")
    plt.show()


if __name__ == "__main__":
    import os

    visualize_evolved_brain()
