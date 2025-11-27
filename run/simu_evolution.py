#!/usr/bin/env/python3

# ======================= Libraries
from bnn_package import (
    pos_nodes_uniform,
    connexion_normal_deterministic,
    get_coupling_operator,
    evolve_system,
    coupling_func,
    step_fhn_rk4,
    step_henon,
    print_simulation_report,
    get_simulation_path,
)

import numpy as np
from numpy.random import default_rng
import time
import h5py
# ======================= Parameters

############# Graph building

rng = default_rng(1234567890)
n = 200
(xmax, ymax) = (10.0, 10.0)
pos = pos_nodes_uniform(n, xmax, ymax, rng)
std = 2.0
transitoire = 1000

############ Time evolution

parameterHenon = [1.1, 0.3]  # a and b in this order
parameterFhN = [0.7, 0.8, 12.5, 0.5, 0.01]  # a, b, tau, Iext, dt in this order

model = "Henon"

if model == "Henon":
    param = parameterHenon
    model_step_func = step_henon
elif model == "FhN":
    param = parameterFhN
    model_step_func = step_fhn_rk4

N_time = 20000
eps = 0.3
ci = 0.5 * np.ones((n, 2))


params_dict = {
    "number of nodes": n,
    "time length simulation": N_time,
    "epsilon": eps,
    "model": model,
    "model parameters": param,
}

MY_FOLDER = "data_simulation"

save_path = get_simulation_path(MY_FOLDER, model, params_dict)
# ====================== Graph

Adjacency = connexion_normal_deterministic(pos, rng, std)
DiffusionOp = get_coupling_operator(Adjacency)

# ====================== Log Graph

print(60 * "=")

print("Number of nodes: N = ", n)
print(
    "Standard deviation of the Gaussian kernel distance: \N{GREEK SMALL LETTER SIGMA} = ",
    std,
)

print(60 * "=")
print_simulation_report(Adjacency, fast_mode=False)

print(20 * "-" + ">" + " READY TO LAUNCH ")

# ====================== Evolution

t_start = time.time()
FullData = evolve_system(
    ci, N_time, param, model_step_func, coupling_func, DiffusionOp, eps
)
t_end = time.time()

print(
    "\n"
    + 20 * "-"
    + ">"
    + f" SIMULATION SUCCESFULLY COMPLETED in {t_end - t_start:.3f}s"
)

Datacuted = FullData[transitoire:, :, :]

with h5py.File(save_path, "a") as f:
    # Save the heavy data with compression
    # 'chunks' allows efficient slicing later
    f.create_dataset(
        "trajectory", data=Datacuted, compression="gzip", compression_opts=4
    )

    # Save Adjacency
    f.create_dataset("adjacency", data=Adjacency, compression="gzip")

    # === THE KEY FEATURE: METADATA ===
    # Store parameters as attributes of the group
    for key, value in params_dict.items():
        f.attrs[key] = value

print(f"\n DATA SUCCESSFULLY SAVED in {MY_FOLDER}")
