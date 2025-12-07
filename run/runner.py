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
    load_config,
    corrupted_simulation,
    prepare_data,
    compute_te_over_lags,
    load_simulation_data,
)
import networkx as nx
import numpy as np
from numpy.random import default_rng
import time
import os
import matplotlib.pyplot as plt

# ======================= CONFIGURATION

# --- PATCH DE COMPATIBILITÉ (Indispensable pour Numpy récent) ---
if not hasattr(np, "int"):
    setattr(np, "int", int)
if not hasattr(np, "float"):
    setattr(np, "float", float)

MY_FOLDER = "data_simulation"
filename = "2025-12-04/FhN_14-55-35_eps0.10_Laplacian_nodes1000.00.h5"  # Adjust based on your 'model' and 'run_name' logic
file_path = os.path.join(MY_FOLDER, filename)

# ======================= DIAGNOSIS and LOADING

corrupted_simulation(file_path)


config_path = "run/config/config_run.yaml"


def main():
    args = load_config(config_path)

    # ======================= PARAMETERS

    #!#!#!#!#!# Graph building

    rng = default_rng(args.seed)
    N_nodes = args.nodes  # number of nodes
    (xmax, ymax) = (args.square, args.square)
    pos = pos_nodes_uniform(N_nodes, xmax, ymax, rng)
    std = args.std
    f = args.poisson  # mean of the Poisson law (o average: number of passive nodes for one active node)

    #!#!#!#!#!# Time evolution

    # ============ FhN

    A = args.A  # 0
    alpha = args.alpha  # 1
    Eps = args.epsilon  # 2
    K = args.K  # 3
    Vrp = args.Vrp  # 4
    dt = args.dt  # 5
    C_r = args.Cr  # coupling between active and passive nodes?
    parameterFhN = [A, alpha, Eps, K, Vrp, dt]

    transitoire = args.transitoire  # physical transition time in s

    # ============ HENON

    a = 1.1
    b = 0.3
    parameterHenon = [a, b]  # a and b in this order

    #!#!#!#!#!# Remaining parameters

    model = "FhN"
    N_time = args.time
    type_diff = args.operator

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
        State_0, N_time, param, model_step_func, N_p, DiffusionOp, C_r, type_diff
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


def measure_te():
    Full_Data = load_simulation_data(file_path, False)
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


def run_simulation(params, order_parameter):
    """
    Executes one simulation based on the 'params' dictionary.
    Returns a dictionary with the results.
    """
    # 1. Unpack Parameters
    eps = params["epsilon"]
    cr = params["cr"]
    N = params["nodes"]

    # 2. Setup Physics
    rng = np.random.default_rng()  # Unique seed per process

    # Physics Params Vector: [A, alpha, epsilon, K, Vrp, dt]
    p_solver = np.array(
        [params["A"], params["alpha"], eps, params["K"], params["Vrp"], params["dt"]],
        dtype=np.float64,
    )

    # 3. Initialization
    State_0 = np.zeros((N, 3))
    State_0[:, 0] = 0.1 + 0.1 * rng.standard_normal(N)
    State_0[:, 1] = 0.3 + 0.1 * rng.standard_normal(N)
    State_0[:, 2] = 1.0 + 0.1 * rng.standard_normal(N)

    # 4. Build Graph
    pos = pos_nodes_uniform(N, 10.0, 10.0, rng)
    Adjacency = connexion_normal_deterministic(pos, rng, params["std"])
    DiffusionOp = get_coupling_operator(Adjacency, "Laplacian")

    # Passive Nodes
    import networkx as nx

    G = nx.from_numpy_array(Adjacency)
    nx.set_node_attributes(G, {n: "active" for n in G.nodes()}, name="type")
    _, N_p = add_passive_nodes(G, params["f"], rng)

    # 5. Run Evolution
    traj = evolve_system(
        State_0,
        params["time_steps"],
        p_solver,
        step_fhn_rk4,
        N_p,
        DiffusionOp,
        cr,
        "Laplacian",
    )

    # 6. Measure (Order Parameter)
    # We discard transient steps
    start = params["transient"]
    # Sample every 10 steps to speed up calculation
    X = traj[start::10, :, 0]
    Y = traj[start::10, :, 1]

    errs = []
    # Note: X.shape[0] is the new time length
    for t in range(X.shape[0]):
        errs.append(order_parameter(X, Y, N, t))

    order_param = np.mean(errs)

    # 7. Return Result
    # We return the sweep parameters + the result
    return {"epsilon": eps, "cr": cr, "order_parameter": order_param}
