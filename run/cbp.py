#!/usr/bin/env/python3

import numpy as np
import pandas as pd
import multiprocessing
import time
from bnn_package import run_simulation_and_measure


def main():
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
    # Standard boilerplate to protect multiprocessing on Windows/macOS
    main()
