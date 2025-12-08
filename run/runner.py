#!/usr/bin/env/python3

# ======================= Libraries


import itertools
import multiprocessing
import os
import sys
import time
import shutil
import uuid

from bnn_package import save_result, load_config
from workers import run_order_parameter, time_series


# ======================= Functions


def archive_graph(config, result_dir):
    """
    Manages Graph Provenance.
    Strictly expects the config to point to an existing graph file
    (which is guaranteed if you use generate_config.py).

    It copies that graph into the result directory so the experiment
    is self-contained.
    """
    source_path = config.get("existing_graph_path")

    # Enforce the Factory Workflow
    if not source_path or not os.path.exists(source_path):
        print("\n>>> CRITICAL ERROR: Graph file missing!")
        print(f"    The config points to: {source_path}")
        print("    Please use 'generate_config.py' to create your experiments.")
        print("    It guarantees a valid graph is created and linked.")
        sys.exit(1)

    print("\n>>> GRAPH PROVENANCE")
    print(f"    Source:   {source_path}")

    # Copy the graph file to the result folder (Archiving)
    filename = os.path.basename(source_path)
    dest_path = os.path.join(result_dir, filename)
    shutil.copy(source_path, dest_path)

    print(f"    Archived: {dest_path}")
    return dest_path


def generate_tasks(config):
    """Create the grid of parameters to run over."""
    fixed_params = {}
    sweep_keys = []
    sweep_values = []

    # Define keys to NEVER sweep
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
    config_path = "run/config/config_phase_scan.yaml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    if not os.path.exists(config_path):
        print("Error: Config path not found.")
        return
    config = load_config(config_path)

    # --- RUN PROVENANCE ---
    run_uuid = str(uuid.uuid4())[:8]
    experiment_name = os.path.splitext(os.path.basename(config_path))[0]
    timestamp = time.strftime("%Y%m%d-%H%M%S")

    # Folder: Date_Name_RunID
    result_dir = os.path.join("Data_output", f"{timestamp}_{experiment_name}")
    os.makedirs(result_dir, exist_ok=True)

    # Snapshot Config
    shutil.copy(config_path, os.path.join(result_dir, f"config_{run_uuid}.yaml"))
    output_file = os.path.join(result_dir, config.get("output_file", "results.csv"))

    print(f"\n{'=' * 60}")
    print(f">>> RUN ID:   {run_uuid}")
    print(f"    Folder:   {result_dir}")

    # --- GRAPH MANAGEMENT (Simplified) ---
    # We replaced 'get_or_create' with 'archive_graph'
    graph_path = archive_graph(config, result_dir)

    # --- GENERATE TASKS ---
    mode = config.get("mode", "sweep")
    target_function = time_series if mode == "time_series" else run_order_parameter

    tasks, sweep_vars = generate_tasks(config)

    # INJECT PROVENANCE INTO EVERY TASK
    for task in tasks:
        task["output_folder"] = result_dir
        task["run_id"] = run_uuid
        task["graph_file_path"] = graph_path

    # --- PARALLEL EXECUTION ---
    use_parallel = config.get("parallel", False)
    ratio_cpu = config.get("cores_ratio", 0.8)
    n_cores = max(
        1, int(multiprocessing.cpu_count() * config.get("cores_ratio", ratio_cpu))
    )

    if not use_parallel:
        print(f"\n>>> SEQUENTIAL ({n_cores} threads)")
        os.environ["OMP_NUM_THREADS"] = str(n_cores)
    else:
        print(f"\n>>> PARALLEL ({n_cores} procs)")
        os.environ["OMP_NUM_THREADS"] = "1"

    print(f"    Sweeping: {sweep_vars}")
    print(f"    Jobs:     {len(tasks)}")
    print(f"{'=' * 60}\n")

    start_time = time.time()

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

    end_time = time.time()

    print(f"\n>>> SIMULATION SUCCESSFULLY COMPLETED IN {end_time - start_time:.3f}s")
    print(f"\n>>> DONE: Saved to {result_dir}")


if __name__ == "__main__":
    main()
