#!/usr/bin/env/python3

# ======================= Libraries

import itertools
import multiprocessing
import pandas as pd
import os
import sys
import time

from bnn_package import load_config
from run.workers import run_order_parameter, time_series

# ======================= Functions


def save_result(res, i, output_file, mode):
    """Saves a single result row to CSV safely

    Args:
        res (_type_): _description_
        i (int): loop index
        output_file (str): relative path of the result file
        mode (str): what is computed

    Returns:
        csv: a file with saved data (order parameter)
    """
    if mode == "time_series":
        return  # Time series worker saves its own files

    if res is None or not res:
        return

    df_temp = pd.DataFrame([res])
    # Write header only on the first run (i==0) AND if file doesn't exist
    header = (i == 0) and (not os.path.exists(output_file))
    df_temp.to_csv(output_file, mode="a", header=header, index=False)


def generate_tasks(config):
    """Detects lists in config and generates the Cartesian product grid."""
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
    # 1. Load Configuration
    config_path = "run/config/config_phase_scan.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    if not os.path.exists(config_path):
        print(f"Error: Config file '{config_path}' not found.")
        return

    config = load_config(config_path)
    output_file = config.get("output_file", "results.csv")

    # 2. Determine Mode
    mode = config.get("mode", "sweep")

    if mode == "time_series":
        target_function = time_series
        print(">>> MODE: Time Series (Heavy Data Saving)")
    else:
        target_function = run_order_parameter
        print(">>> MODE: Phase Scan")

    # 3. Generate Grid
    tasks, sweep_vars = generate_tasks(config)

    # 4. Setup Parallelism
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

    start_time = time.time()

    # 5. Run Loop
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
        # Sequential
        from tqdm import tqdm

        for i, task in enumerate(tqdm(tasks)):
            res = target_function(task)
            save_result(res, i, output_file, mode)

    print(f"\n>>> DONE in {time.time() - start_time:.2f} s")


if __name__ == "__main__":
    main()
