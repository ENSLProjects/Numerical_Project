#!/usr/bin/env/python3

# ======================= LIBRARIES

from bnn_package import (
    pos_nodes_uniform,
    connexion_normal_deterministic,
    get_coupling_operator,
    evolve_system,
    step_fhn_rk4,
    add_passive_nodes,
    step_henon,
    print_simulation_report,
    get_simulation_path,
    save_simulation_data,
)
import networkx as nx
import numpy as np
from numpy.random import default_rng
import time
import os

# ======================= PARAMETERS

#!#!#!#!#!# Graph building

rng = default_rng(1234567890)
N_nodes = 1000  # number of nodes
(xmax, ymax) = (10.0, 10.0)
pos = pos_nodes_uniform(N_nodes, xmax, ymax, rng)
std = 0.25
f = 3  # mean of the Poisson law (o average: number of passive nodes for one active node)

#!#!#!#!#!# Time evolution

# ============ FhN

A = 3.0  # 0
alpha = 0.2  # 1
Eps = 0.1  # 2
K = 0.25  # 3
Vrp = 1.5  # 4
dt = 0.01  # 5
C_r = 1.5  # coupling between active and passive nodes?
parameterFhN = [A, alpha, Eps, K, Vrp, dt]

transitoire = int(500 / dt)  # physical transition time in s

# ============ HENON

a = 1.1
b = 0.3
parameterHenon = [a, b]  # a and b in this order

#!#!#!#!#!# Remaining parameters

model = "FhN"
N_time = 300000
type_diff = "Laplacian"

#!#!#!#!#!# Dictionnaries

if model == "Henon":
    param = np.array(parameterHenon, dtype=np.float64)
    model_step_func = step_henon
    parameters = {"coupling": Eps, "a": a, "b": b}

elif model == "FhN":
    param = np.array(parameterFhN, dtype=np.float64)
    model_step_func = step_fhn_rk4
    State_0 = np.zeros((N_nodes, 3))
    State_0[:, 0] = 0.1 + 0.1 * rng.standard_normal(N_nodes)  # v_e
    State_0[:, 1] = 0.3 + 0.1 * rng.standard_normal(N_nodes)  # g
    State_0[:, 2] = 1.0 + 0.1 * rng.standard_normal(N_nodes)
    parameters = {
        "A": A,
        "alpha": alpha,
        "coupling_active": Eps,
        "K": K,
        "Vrp": Vrp,
        "coupling_passive": C_r,
        "time_step rk4": dt,
        "average Poisson": f,
    }

params_dict = {
    "number of nodes": N_nodes,
    "std graph": std,
    "epsilon": Eps,
    "time length simulation": N_time,
    "model": model,
    "how to diffuse": type_diff,
    "parameters_model": parameters,
}

MY_FOLDER = "data_simulation"
GRAPH_FOLDER = "data_simulation/graph"

os.makedirs(MY_FOLDER, exist_ok=True)
os.makedirs(GRAPH_FOLDER, exist_ok=True)

save_path = get_simulation_path(MY_FOLDER, model, params_dict)

# ====================== GRAPH

Adjacency = connexion_normal_deterministic(pos, rng, std)
DiffusionOp = get_coupling_operator(Adjacency, type_diff)

G = nx.from_numpy_array(Adjacency)
nx.set_node_attributes(G, {node: "active" for node in G.nodes()}, name="type")
Graph_passive, N_p = add_passive_nodes(G, f, rng)

# Ensure passive nodes are tagged (if the function didn't do it)
for node in Graph_passive.nodes():
    if "type" not in Graph_passive.nodes[node]:
        Graph_passive.nodes[node]["type"] = "passive"

# ====================== TYPES

DiffusionOp = DiffusionOp.astype(np.float64)
State_0 = State_0.astype(np.float64)

# ====================== LOG GRAPH

print(60 * "=")

print("Number of nodes: N = ", N_nodes)
print(
    "Standard deviation of the Gaussian kernel distance: \N{GREEK SMALL LETTER SIGMA} = ",
    std,
)

print(60 * "=")
print_simulation_report(
    Adjacency, fast_mode=True
)  # comment this line to avoid all topology analysis

print(20 * "-" + ">" + " READY TO LAUNCH ")

# ====================== EVOLUTION

t_start = time.time()
FullData = evolve_system(
    State_0, N_time, param, step_fhn_rk4, N_p, DiffusionOp, C_r, type_diff
)
t_end = time.time()

print("\n" + 20 * "-" + ">" + " SIMULATION SUCCESFULLY COMPLETED")

print(f"\n[System] Simulation completed in {t_end - t_start:.3f}s")
print("=" * 60)

Datacuted = FullData[transitoire:, :, :]

graph_filename = f"graph_{model}_{N_nodes}_{Eps}_Poisson{f}.graphml"
full_graph_path = os.path.join(GRAPH_FOLDER, graph_filename)

nx.write_graphml(Graph_passive, full_graph_path)

print(f"\nGraph saved to: {full_graph_path}")
print(60 * "=" + "\n")


save_simulation_data(save_path, Datacuted, params_dict, full_graph_path)
