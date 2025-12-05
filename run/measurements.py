#!/usr/bin/env/python3

# ======================= LIBRARIES

from bnn_package import (
    corrupted_simulation,
    prepare_data,
    compute_te_over_lags,
    load_simulation_data,
    run_simulation_and_measure,
)
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import multiprocessing
import time

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


def phase_diagram():
    # ================= CONFIGURATION =================
    # 1. Define your Phase Space (The Arrays you wanted)
    EPSILON_ARRAY = np.linspace(0.0, 0.4, 20)  # 20 steps
    CR_ARRAY = np.linspace(0.0, 2.0, 20)  # 20 steps

    # 2. Fixed Simulation Settings
    N_NODES = 500
    TIME_STEPS = 50000

    # 3. Output File
    OUTPUT_FILE = "phase_diagram_results.csv"

    # 4. Parallelization
    # Use roughly 80% of available cores to keep the system responsive
    N_CORES = max(1, int(multiprocessing.cpu_count() * 0.8))
    # =================================================

    # Generate the List of all jobs (Combinations of Eps and Cr)
    tasks = []
    for eps in EPSILON_ARRAY:
        for cr in CR_ARRAY:
            # Create a tuple of arguments for the worker
            tasks.append((eps, cr, N_NODES, TIME_STEPS))

    print(">>> Starting Phase Space Scan")
    print(f"    Total Simulations: {len(tasks)}")
    print(f"    Workers (Cores):   {N_CORES}")
    print(f"    Output File:       {OUTPUT_FILE}")

    start_time = time.time()

    # --- RUN LOOP IN PARALLEL ---
    # Pool creates N_CORES python processes.
    # .imap_unordered is efficient and yields results as soon as they finish.
    results = []

    with multiprocessing.Pool(processes=N_CORES) as pool:
        # We use tqdm if available for a nice progress bar, otherwise standard iterator
        try:
            from tqdm import tqdm

            iterator = tqdm(
                pool.imap_unordered(run_simulation_and_measure, tasks), total=len(tasks)
            )
        except ImportError:
            print("Tip: Install 'tqdm' for a progress bar.")
            iterator = pool.imap_unordered(run_simulation_and_measure, tasks)

        for res in iterator:
            results.append(res)
            # Optional: Save intermediate results immediately to disk
            # (Good practice in case the script crashes halfway)
            with open(OUTPUT_FILE, "a") as f:
                # Format: eps, cr, order_param
                f.write(f"{res[0]},{res[1]},{res[2]}\n")

    end_time = time.time()
    print(f"\n>>> Scan Complete in {end_time - start_time:.2f} seconds.")

    # Save clean final CSV with headers
    df = pd.DataFrame(results, columns=["epsilon", "cr", "order_parameter"])
    # Sort for easier reading
    df = df.sort_values(by=["epsilon", "cr"])
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"    Saved sorted results to {OUTPUT_FILE}")


if __name__ == "__main__":
    measure_te()
