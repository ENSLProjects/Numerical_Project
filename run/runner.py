#!/usr/bin/env/python3

# ======================= Libraries


import itertools
import multiprocessing
import pandas as pd
import os
import sys
import time
import shutil

from run.workers import run_order_parameter, time_series
from bnn_package import load_config


# ======================= Functions


def save_result(res, i, output_file, mode):
    """Save clean results in csv.

    Args:
        res (): _description_
        i (int): Loop index.
        output_file (str): Relative file path to save the csv.
        mode (str): What is computes?
    """
    if mode == "time_series":
        return

    if res is None or not res:
        return

    df_temp = pd.DataFrame([res])
    # Write header only on the first run (i==0) AND if file doesn't exist
    header = (i == 0) and (not os.path.exists(output_file))
    df_temp.to_csv(output_file, mode="a", header=header, index=False)


def generate_tasks(config):
    """Read the .yaml configuration file and create a py dictionnary with all the loop from the cartesian product of all parameters in order to scan the phase space.

    Args:
        config (dic): Input dictionnaty from the config.yaml

    Returns:
        dic: All combinaisons of parameters to run over.
    """
    fixed_params = {}
    sweep_keys = []
    sweep_values = []

    for key, val in config.items():
        if isinstance(val, list):
            sweep_keys.append(key)
            sweep_values.append(val)
        else:
            fixed_params[key] = val

    combinations = list(itertools.product(*sweep_values))
    tasks = []
    for combo in combinations:
        task = fixed_params.copy()
        for i, key in enumerate(sweep_keys):
            task[key] = combo[i]
        tasks.append(task)
    return tasks, sweep_keys


def main():
    # --- LOAD CONFIG ---
    config_path = "run/config/config_phase_scan.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found.")
        return

    config = load_config(config_path)

    # 1. Generate a unique name for this run
    experiment_name = os.path.splitext(os.path.basename(config_path))[0]
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # 2. Create the Result Directory
    result_dir = os.path.join("results", f"{timestamp}_{experiment_name}")
    os.makedirs(result_dir, exist_ok=True)

    # 3. Snapshot the Config File (Save a copy)
    shutil.copy(config_path, os.path.join(result_dir, "config_snapshot.yaml"))

    # 4. Redirect output to this new folder
    output_filename = config.get("output_file", "results.csv")
    output_file = os.path.join(result_dir, output_filename)

    print(f"\n{'=' * 60}")
    print(f">>> RUN STARTED: {experiment_name}")
    print(f"    Results Dir: {result_dir}")
    print("    Config Snapshot Saved.")

    # --- SETUP EXECUTION ---
    mode = config.get("mode", "sweep")

    if mode == "time_series":
        target_function = time_series
        print(">>> MODE: Time Series")
        # IMPORTANT: Tell the worker to save HDF5 files inside our new result_dir
        # We inject this path into every task parameters
    else:
        target_function = run_order_parameter
        print(">>> MODE: Phase Scan")

    # Generate Tasks
    tasks, sweep_vars = generate_tasks(config)

    # Inject result_dir into tasks so workers know where to save (if needed)
    for task in tasks:
        task["output_folder"] = result_dir

    # Setup Parallelism
    use_parallel = config.get("parallel", True)
    n_cores = max(1, int(multiprocessing.cpu_count() * config.get("cores_ratio", 0.8)))

    if not use_parallel:
        print(f">>> SEQUENTIAL MODE ({n_cores} threads/process)")
        os.environ["OMP_NUM_THREADS"] = str(n_cores)
    else:
        print(f">>> PARALLEL MODE ({n_cores} processes)")
        os.environ["OMP_NUM_THREADS"] = "1"

    print(f"    Sweeping: {sweep_vars}")
    print(f"    Jobs:     {len(tasks)}")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    # --- RUN LOOP ---
    if use_parallel:
        with multiprocessing.Pool(n_cores) as pool:
            try:
                from tqdm import tqdm

                iterator = tqdm(
                    pool.imap_unordered(target_function, tasks), total=len(tasks)
                )
            except ImportError:
                iterator = pool.imap_unordered(target_function, tasks)

            for i, res in enumerate(iterator):
                save_result(res, i, output_file, mode)
    else:
        from tqdm import tqdm

        for i, task in enumerate(tqdm(tasks)):
            res = target_function(task)
            save_result(res, i, output_file, mode)

    print(f"\n>>> DONE in {time.time() - start_time:.2f} s")
    print(f"    All data saved in: {result_dir}")


if __name__ == "__main__":
    main()
