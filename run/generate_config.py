#!/usr/bin/env/python3

# ======================= Libraries


import yaml
import os
from datetime import datetime


# ======================= Functions


def create_experiment_config(experiment_name, **kwargs):
    """
    Generates a standardized YAML config file for a simulation.

    Args:
        experiment_name (str): A descriptive name (e.g., 'scan_epsilon_cr')
        **kwargs: Any parameter you want to override from the default.
    """

    # 1. The BASE Template (Default Values)
    template = {
        # EXECUTION PARAMETERS
        "mode": "sweep",  # 'sweep' or 'time_series'
        "output_file": f"results_{experiment_name}.csv",
        "parallel": True,
        "cores_ratio": 0.8,
        # GRAPH DEFAULT PARAMETERS
        "Number of Nodes": 500,
        "Square for the graph": [10, 10],
        "Diffusive Operator": "Laplacian",
        "Std": 0.25,
        "Mean Poisson": 3,
        # PHYSICAL DEFAULT PARAMETERS
        "Total time": 300000,
        "Transitory time": 10000,
        "A": 3.0,
        "Alpha": 0.2,
        "K": 0.25,
        "Vrp": 1.5,
        "dt": 0.01,
        # PARAMETERS SWEEPED (Empty by default, to be filled by kwargs)
        "Epsilon": 0.1,
        "Cr": 1.0,
        "metrics": ["sync_error"],  # add here your customized default order parameters
    }

    # 2. Update Template with your specific custom values
    # This overrides the defaults with whatever you passed to the function
    for key, value in kwargs.items():
        if key not in template:
            print(f"Warning: Adding new key '{key}' not present in default template.")
        template[key] = value

    # 3. Save to File
    # Create a folder for configs if it doesn't exist
    os.makedirs("run/configs", exist_ok=True)

    # Timestamp ensures you never overwrite previous experiments
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"run/configs/{timestamp}_{experiment_name}.yaml"

    with open(filename, "w") as f:
        yaml.dump(template, f, sort_keys=False, default_flow_style=None)

    print(f">>> Generated Config: {filename}")
    return filename


if __name__ == "__main__":
    create_experiment_config(
        "trial_test",
        mode="time_series",
        Epsilon=0.08,  # Fixed value
        Cr=1.5,  # Fixed value
        Total_time=300000,  # Longer run
    )
