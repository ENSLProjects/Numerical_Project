#!/usr/bin/env/python3

# ======================= LIBRARIES

import h5py
import matplotlib.pyplot as plt
import os

# ======================= CONFIGURATION

MY_FOLDER = "data_simulation"
filename = "2025-12-03/FhN_14-47-45_eps0.30_Laplacian_nodes1000.00.h5"  # Adjust based on your 'model' and 'run_name' logic
file_path = os.path.join(MY_FOLDER, filename)

# The specific run name you used inside the script
run_name = "trial"

# ================= Reading the File =================

if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
else:
    with h5py.File(file_path, "r") as f:
        print(f"--- File: {filename} ---")

        if run_name not in f:
            print(
                f"Error: Run '{run_name}' not found. Available runs: {list(f.keys())}"
            )
        else:
            grp = f[run_name]
            print(f"Reading Group: {run_name}")

            trajectory = grp["trajectory"][:]
            adjacency = grp["adjacency"][:]

            # Check for passive nodes dataset (handle naming variations)
            if "passive_nodes" in grp:
                passive_counts = grp["passive_nodes"][:]
            else:
                passive_counts = None

            print("\n[Data Shapes]")
            print(f"  - Trajectory: {trajectory.shape} (Time, Nodes, Dims)")
            print(f"  - Adjacency:  {adjacency.shape}")

            # 3. LOAD METADATA (Attributes)
            print("\n[Simulation Parameters]")
            graph_file_link = None

            for key, val in grp.attrs.items():
                print(f"  - {key}: {val}")

                # Capture the graph link if we need it later
                if key == "associated_graph_file":
                    graph_file_link = val

            # ================= Visualization =================

            # Example: Plot the time series of the first active node
            # Trajectory shape is (T, N, 3) -> We take Node 0, Variable 0 (Voltage)

            plt.figure(figsize=(10, 5))

            # Plot Active Node 0
            plt.plot(trajectory[:, 0, 0], label="Active Node 0 (Voltage)", color="blue")

            # Plot a connected passive node (if exists)
            if passive_counts is not None and passive_counts[0] > 0:
                # Need to find the index of a passive node connected to Node 0.
                # Since we don't have the graph object here easily,
                # we can guess (passive nodes usually start after active ones).
                # But for now, let's just plot the active one to be safe.
                pass

            plt.title(f"Time Evolution (Run: {run_name})")
            plt.xlabel("Time Steps")
            plt.ylabel("Voltage (v_e)")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

            # 4. (Optional) Check the Graph Link
            if graph_file_link:
                full_graph_path = os.path.join(MY_FOLDER, "graph", graph_file_link)
                if os.path.exists(full_graph_path):
                    print(
                        f"\n[SUCCESS] Linked GraphML file found at: {full_graph_path}"
                    )
                else:
                    print("\n[WARNING] GraphML file mentioned in metadata is missing!")
