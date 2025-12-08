#!/usr/bin/env/python3

# ======================= Libraries


import numpy as np
import matplotlib.pyplot as plt

from bnn_package import load_simulation_data, prepare_data, compute_te_over_lags


# ======================= Functions


def measure_te(file_path):
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
