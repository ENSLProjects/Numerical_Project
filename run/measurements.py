#!/usr/bin/env/python3

# ======================= LIBRARIES

from bnn_package import corrupted_simulation, pull_out_full_data, prepare_data
import os
import numpy as np
import matplotlib.pyplot as plt

# ======================= CONFIGURATION

MY_FOLDER = "data_simulation"
filename = "2025-12-03/FhN_22-15-15_eps0.08_Laplacian_nodes1000.00.h5"  # Adjust based on your 'model' and 'run_name' logic
file_path = os.path.join(MY_FOLDER, filename)

# --- PATCH DE COMPATIBILITÉ (Indispensable pour Numpy récent) ---
if not hasattr(np, "int"):
    setattr(np, "int", int)
if not hasattr(np, "float"):
    setattr(np, "float", float)

# ======================= DIAGNOSIS and LOADING

corrupted_simulation(file_path)

Full_Data = pull_out_full_data(file_path)

if Full_Data is None:
    raise ValueError(
        f"Failed to load data from {file_path}. Check if the file exists and is valid."
    )

Trajectory = Full_Data["time trajectory"]

fig, ax = plt.subplots()
ax.plot(Trajectory[:, 4, 0], label=f"Node {4}")
ax.plot(Trajectory[:, 5, 0], label=f"Node {5}")
ax.plot(Trajectory[:, 6, 0], label=f"Node {6}")
ax.plot(Trajectory[:, 10, 0], label=f"Node {10}")
ax.plot(Trajectory[:, 100, 0], label=f"Node {100}")
ax.plot(Trajectory[:, 500, 0], label=f"Node {500}")
ax.plot(Trajectory[:, 930, 0], label=f"Node {930}")
ax.set_xlabel("Time Step")
ax.set_ylabel("Voltage $V_e$")
ax.legend()
ax.grid(True)
plt.show()

# ======================= ENTROPY

# ------------ parameters
node = 10
lag = 1
kNN = 5
n_embeded = 1
N = 4096
# ------------- data
ve = prepare_data(
    Trajectory[:, node, 0]
)  # for now we only care about the tension of the active nodes
