#!/usr/bin/env/python3

# ======================= LIBRARIES

from bnn_package import (
    corrupted_simulation,
    pull_out_full_data,
    prepare_data,
    compute_te_over_lags,
)
import os
import numpy as np
import matplotlib.pyplot as plt
import entropy.entropy as ee

# ======================= CONFIGURATION

MY_FOLDER = "data_simulation"
filename = "2025-12-03/FhN_22-43-55_eps0.08_Laplacian_nodes1000.00.h5"  # Adjust based on your 'model' and 'run_name' logic
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
    raise ValueError(f"Failed to load data from {file_path}")
Trajectory = Full_Data["time trajectory"]
# Note: pull_out_full_data returns the dict under the key "parameters"
All_Params = Full_Data["parameters"]
Parameters_model = All_Params["parameters_model"]
rk4_time = Parameters_model["time_step rk4"]
final_time = All_Params["time length simulation"]
real_time = rk4_time * np.arange(Trajectory.shape[0])

fig, ax = plt.subplots()
ax.plot(real_time, Trajectory[:, 4, 0], label=f"Node {4}")
ax.plot(real_time, Trajectory[:, 5, 0], label=f"Node {5}")
ax.plot(real_time, Trajectory[:, 6, 0], label=f"Node {6}")
ax.plot(real_time, Trajectory[:, 10, 0], label=f"Node {10}")
ax.plot(real_time, Trajectory[:, 100, 0], label=f"Node {100}")
ax.plot(real_time, Trajectory[:, 500, 0], label=f"Node {500}")
ax.plot(real_time, Trajectory[:, 930, 0], label=f"Node {930}")
ax.set_xlabel("Time Step")
ax.set_ylabel("Voltage $V_e$")
ax.legend()
ax.grid(True)
plt.show()

# ======================= ENTROPY

# ------------ parameters
node_source = 5
node_target = 500
lag = 1
kNN = 5
n_embeded = 2
N_eff = 4096  # number of points on each subset measurement
N_real = 20  # number of subsets
# ------------- data

ee.get_sampling(verbosity=1)

LAGS = np.arange(1, 10001, 100, dtype=int)

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
        fmt="-o",
        color="crimson",
        ecolor="gray",
        capsize=3,
        label=f"TE({node_source} $\\to$ {node_target})",
    )

    # Calculate Integral (Sum)
    te_integral = np.sum(te_mean)

    plt.title(f"Transfer Entropy Lag Profile\nIntegral = {te_integral:.4f}")
    plt.xlabel("Lag $\\tau$ (time steps)")
    plt.ylabel("Transfer Entropy (nats)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

except Exception as e:
    print(f"Error: {e}")
