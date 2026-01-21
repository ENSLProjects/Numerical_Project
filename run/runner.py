#!/usr/bin/env python3

# ======================= Libraries
import os
import sys
import fcntl
import numpy as np
import traceback 

# ======================= ADAPTIVE ENVIRONMENT CONFIGURATION =======================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["LOKY_MAX_CPU_COUNT"] = "1"
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

# ======================= IMPORTS =======================
import itertools
import multiprocessing
from multiprocessing import shared_memory
import time
import shutil
import uuid
from tqdm import tqdm
from bnn_package import save_result, load_config, generate_poisson_input
from workers import (
    run_order_parameter, time_series, 
    research_alignment_worker_2, 
    research_alignment_worker_TE_tensor, 
    init_worker_globals,
    worker_wrapper
)
from workers import _global_graph, _global_passive_counts, _global_input_signal


# Mappage complet : on couvre les noms de config ET les noms de fonctions possibles
MODE_MAP = {

    "time_series": time_series,
    "sweep": run_order_parameter,
    "research_alignment": research_alignment_worker_2,
    "TE": research_alignment_worker_TE_tensor,
    "run_order_parameter": run_order_parameter,
    "research_alignment_worker_2": research_alignment_worker_2
}

# ======================= Functions (Helpers) =======================
def archive_graph(config, result_dir):
    source_path = config.get("existing_graph_path")
    if not source_path or not os.path.exists(source_path):
        if config.get("generate_graph", False):
            return None
        print("\n>>> CRITICAL ERROR: Graph file missing!")
        sys.exit(1)

    print("\n>>> GRAPH PROVENANCE")
    print(f"    Source:   {source_path}")
    filename = os.path.basename(source_path)
    dest_path = os.path.join(result_dir, filename)
    shutil.copy(source_path, dest_path)
    print(f"    Archived: {dest_path}")
    return dest_path

def generate_tasks(config):
    fixed_params = {}
    sweep_keys = []
    sweep_values = []

    NON_SWEEP = [
        "square_for_graph", "metrics", "diffusive_operator", 
        "existing_graph_path", "forced_targets", "parallel", "cores_ratio", "output_file"
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



# ======================= MAIN =======================
def main():



    # --- ANTI-DOUBLE-LANCEMENT ---
    lock_file = os.path.join("/tmp", f"phase_scan_{os.getpid()}.lock")
    try:
        fp = open(lock_file, "w")
        fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        print(f"⚠️  Un autre run est déjà en cours. Supprime {lock_file}.")
        sys.exit(1)

    # --- LOAD CONFIG ---
    config_path = sys.argv[1] if len(sys.argv) > 1 else "run/configs/config_phase_scan.yaml"
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

    # --- CREATE INPUT LOCK FILE ---
    input_signal = None
    target_idxs = None

    print(">>> Generating input signal...")
    input_signal_generated, target_idxs = generate_poisson_input(
        n_nodes=config["number_of_nodes"],
        final_time=config["total_time"],
        dt=config["dt"],
        n_targets=config.get("input_targets", 1), 
        rate_hz=config["input_rate"],
        magnitude=config["input_magnitude"],
        forced_targets=config.get("forced_targets", None), 
        
    )
    print(f">>> Input signal generated. Shape: {input_signal_generated.shape}")

    # --- PREPARE DATA ---
    graph_path = archive_graph(config, result_dir)
    print(">>> Loading graph into memory (Main Process)...")
    try:
        with np.load(graph_path) as data:
            shared_graph = data["adjacency"]      
            position = data["positions"]      
            n_nodes = data["n_nodes"]          
            passive_counts = data["passive_counts"] if "passive_counts" in data else None
        print(f">>> Graph loaded. Nodes: {n_nodes}, Passive Counts: {'Yes' if passive_counts is not None else 'No'}")
    except Exception as e:
        print(f"\n>>> ERROR LOADING GRAPH: {e}")
        sys.exit(1)


    # --- SHARED MEMORY ---
    shm = shared_memory.SharedMemory(
        create=True,
        size=input_signal_generated.nbytes
    )

    shared_input = np.ndarray(
        input_signal_generated.shape,
        dtype=input_signal_generated.dtype,
        buffer=shm.buf
    )

    shared_input[:] = input_signal_generated[:]

    # --- PREPARE TASKS ---
    tasks, sweep_vars = generate_tasks(config)
    run_mode = config.get("mode", "sweep")
    
    for task in tasks:
        task.update({
            "output_folder": result_dir,
            "run_id": run_uuid,
            "graph_file_path": graph_path,
            "mode": run_mode,
            "forced_targets": target_idxs,   
            "number_of_nodes": config["number_of_nodes"],
            "total_time": config["total_time"],
            "dt": config["dt"],
            "input_rate": config["input_rate"],
            "input_magnitude": config["input_magnitude"],
            "preloaded_input_signal": input_signal_generated,
            "target_indices": target_idxs,
            "passive_counts": passive_counts,
            "preloaded_graph": shared_graph,  
            
        })


    # --- EXECUTION ---
    use_parallel = config.get("parallel", False)
    ratio_cpu = config.get("cores_ratio", 0.5)
    max_cores = int(multiprocessing.cpu_count() * ratio_cpu)
    n_cores = min(10, max(1, max_cores)) 
    
    output_file = os.path.join(result_dir, config.get("output_file", "results.csv"))

    print(f"\n{'=' * 60}")
    print(f">>> RUN ID:   {run_uuid}")
    print(f"    Mode:     {run_mode}")
    print(f"    Sweeping: {sweep_vars}")
    print(f"    Tasks:    {len(tasks)}")
    print(f"    Parallel: {use_parallel} ({n_cores} cores)")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    if use_parallel:
        print(">>> Starting Multiprocessing Pool (Optimized with Wrapper)...")
        try:
            with multiprocessing.Pool(
                n_cores,
                initializer=init_worker_globals,
                initargs=(shared_graph, passive_counts, input_signal_generated)
            ) as pool:

                results_iterator = pool.imap_unordered(worker_wrapper, tasks)
                
                results = []
                
                print(">>> Saving results...")
                valid_count = 0
                for i, res in enumerate(tqdm(results_iterator, total=len(tasks), desc="Running simulations", ncols=80)):
                    if res is not None:
                        save_result(res, i, output_file, run_mode)
                        valid_count += 1
                
                    else:
                        print(f"\n!!! WARNING: Task {i} returned None. Check worker logs for errors. !!!")
                print(f">>> Saved {valid_count}/{len(tasks)} results.")
        except Exception as e:
            print("\n>>> CRITICAL POOL ERROR:")
            traceback.print_exc()
            
    else:
        print(">>> Starting Sequential Loop...")
        target_func = MODE_MAP.get(run_mode, run_order_parameter)

        for i, task in enumerate(tqdm(tasks, unit="sim", ncols=80)):
            task["preloaded_graph"] = shared_graph
            task["passive_counts"] = passive_counts
            task["input_signal"] = input_signal_generated 

            try:
                res = target_func(task)
                save_result(res, i, output_file, run_mode)
            except Exception as e:
                print(f"Error in task {i}: {e}")
                traceback.print_exc()



    print(f"\n>>> COMPLETED in {time.time() - start_time:.2f}s")
    print(f">>> Results in: {result_dir}")

if __name__ == "__main__":
    main()
