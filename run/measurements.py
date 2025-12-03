#!/usr/bin/env/python3

# ======================= LIBRARIES

from bnn_package import corrupted_simulation, pull_out_full_data, prepare_data
import os
import numpy as np

# ======================= CONFIGURATION

MY_FOLDER = "data_simulation"
filename = "2025-12-03/FhN_20-49-40_eps0.08_Laplacian_nodes1000.00.h5"  # Adjust based on your 'model' and 'run_name' logic
file_path = os.path.join(MY_FOLDER, filename)

# The specific run name you used inside the script
run_name = "debugging simulation"

# --- PATCH DE COMPATIBILITÉ (Indispensable pour Numpy récent) ---
if not hasattr(np, "int"):
    setattr(np, "int", int)
if not hasattr(np, "float"):
    setattr(np, "float", float)

# ======================= DIAGNOSIS and LOADING

corrupted_simulation(file_path, run_name)

Full_Data = pull_out_full_data(file_path, run_name)

Trajectory = Full_Data["time trajectory"]

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
