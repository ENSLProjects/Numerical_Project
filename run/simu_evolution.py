#!/usr/bin/env/python3

# ======================= Libraries
from bnn_package import (
    pos_nodes_uniform,
    connexion_normal_deterministic,
    get_coupling_operator,
    evolve_system,
    coupling_func,
    step_fhn_rk4,
    print_simulation_report,
    get_simulation_path
)
from numpy.random import default_rng

import h5py
# ======================= Parameters

############# Graph building

rng = default_rng(1234567890)
n = 100
(xmax, ymax) = (10.0, 10.0)
pos = pos_nodes_uniform(n, xmax, ymax, rng)
std = 2.0

############ Time evolution

parameterHenon = []
parameterFhN = []
N_time = 50000
eps = 0.3
ci = 0
model_step_func = step_fhn_rk4

run_name = "trial"

params_dict = {
    "number of nodes": n,
    "time length simulation": N_time,
    "epsilon": eps,
    "model": f"{model_step_func}",
    "model parameters": parameterHenon,
    "run name": run_name
}

MY_FOLDER = "data_simulation"

save_path = get_simulation_path(MY_FOLDER, "Henon", params_dict)
# ====================== Graph

Adjacency = connexion_normal_deterministic(pos, rng, std)
DiffusionOp = get_coupling_operator(Adjacency)

# ====================== Log Graph

print(30 * "=")

print("Number of nodes: N = ", n)
print(
    "Standard deviation of the Gaussian kernel distance: \N{GREEK SMALL LETTER SIGMA} = ",
    std,
)

print(30 * "=")
print_simulation_report(Adjacency, "Sim_001", fast_mode=False)


# ====================== Evolution

FullData = evolve_system(
    ci, N_time, parameterFhN, model_step_func, coupling_func, DiffusionOp, eps
)


with h5py.File(save_path, "a") as f:
    # Create a Group (like a folder)
    grp = f.create_group(run_name)

    # Save the heavy data with compression
    # 'chunks' allows efficient slicing later
    dset = grp.create_dataset(
        "trajectory", data=FullData, compression="gzip", compression_opts=4
    )

    # Save Adjacency
    grp.create_dataset("adjacency", data=Adjacency, compression="gzip")

    # === THE KEY FEATURE: METADATA ===
    # Store parameters as attributes of the group
    for key, value in params_dict.items():
        grp.attrs[key] = value
