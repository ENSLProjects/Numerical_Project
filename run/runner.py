#!/usr/bin/env python3

# ======================= Libraries


import itertools
import multiprocessing
import os
import sys
import time
import shutil
import uuid
from tqdm import tqdm

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from bnn_package import save_result, load_config, corrupted_simulation  # noqa: F401
from workers import run_order_parameter, time_series


# ======================= Functions


def archive_graph(config, result_dir):
    """Archiving Graph Logic."""
    source_path = config.get("existing_graph_path")
    if not source_path or not os.path.exists(source_path):
        if config.get("generate_graph", False):
            return None
        print("\n>>> CRITICAL ERROR: Graph file missing!")
        print(f"    Path: {source_path}")
        sys.exit(1)

    print("\n>>> GRAPH PROVENANCE")
    print(f"    Source:   {source_path}")
    filename = os.path.basename(source_path)
    dest_path = os.path.join(result_dir, filename)
    shutil.copy(source_path, dest_path)
    print(f"    Archived: {dest_path}")
    return dest_path


def generate_tasks(config):
    """Grid Search Generator."""
    fixed_params = {}
    sweep_keys = []
    sweep_values = []

    # Parameters that should NOT be split into permutations
    NON_SWEEP = [
        "square_for_graph",
        "metrics",
        "diffusive_operator",
        "existing_graph_path",
    ]

    for key, val in config.items():
        if isinstance(val, list) and key not in NON_SWEEP:
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
    config_path = "run/configs/config_phase_scan.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        return

    config = load_config(config_path)

    # --- SETUP OUTPUT ---
    raw_name = os.path.splitext(os.path.basename(config_path))[0]
    run_uuid = str(uuid.uuid4())[:8]
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    result_dir = os.path.join("Data_output", f"{timestamp}_{raw_name}")
    os.makedirs(result_dir, exist_ok=True)

    shutil.copy(config_path, os.path.join(result_dir, f"config_{run_uuid}.yaml"))

    graph_path = archive_graph(config, result_dir)

    # --- PREPARE TASKS ---
    mode = config.get("mode", "sweep")
    target_function = time_series if mode == "time_series" else run_order_parameter

    tasks, sweep_vars = generate_tasks(config)

    for task in tasks:
        task["output_folder"] = result_dir
        task["run_id"] = run_uuid
        task["graph_file_path"] = graph_path

    # --- EXECUTION ---
    use_parallel = config.get("parallel", False)
    ratio_cpu = config.get("cores_ratio", 0.8)
    n_cores = max(1, int(multiprocessing.cpu_count() * ratio_cpu))

    output_file = os.path.join(result_dir, config.get("output_file", "results.csv"))

    print(f"\n{'=' * 60}")
    print(f">>> RUN ID:   {run_uuid}")
    print(f"    Mode:     {mode}")
    print(f"    Sweeping: {sweep_vars}")
    print(f"    Tasks:    {len(tasks)}")
    print(f"    Parallel: {use_parallel} ({n_cores} cores)")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    if use_parallel:
        print(">>> Starting Multiprocessing Pool...")
        os.environ["OMP_NUM_THREADS"] = "1"
        with multiprocessing.Pool(n_cores) as pool:
            # imap_unordered + tqdm for real-time progress
            results = list(
                tqdm(
                    pool.imap_unordered(target_function, tasks),
                    total=len(tasks),
                    unit="sim",
                    ncols=80,
                )
            )
            for i, res in enumerate(results):
                save_result(res, i, output_file, mode)
    else:
        print(">>> Starting Sequential Loop...")
        os.environ["OMP_NUM_THREADS"] = str(n_cores)
        for i, task in enumerate(tqdm(tasks, unit="sim", ncols=80)):
            res = target_function(task)
            save_result(res, i, output_file, mode)

    print(f"\n>>> COMPLETED in {time.time() - start_time:.2f}s")
    print(f">>> Results in: {result_dir}")


if __name__ == "__main__":
    corrupted_simulation(
        "Data_output/20251224-175714_test_optimal_dt/ts_N1000_Coup4.000_cr0.800_G-60409aeb.h5"
    )
