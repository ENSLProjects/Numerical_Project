#!/usr/bin/env python3

from .buildgraph import (
    pos_nodes_uniform,
    norm_multi,
    local_connect_gaussian,
    local_connect_lorentz,
    connexion_normal_random_NUMBA,
    connexion_normal_deterministic,
    add_passive_nodes,
)

from .evolution import (
    evolve_system,
    get_coupling_operator,
    rk4_step,  # REPLACES step_fhn_rk4
    euler_step,  # NEW
    map_step,  # REPLACES step_henon
    FitzHughNagumoModel,  # NEW: The Physics Engine
    HenonMapModel,  # NEW: The Map Engine
)

from .data_processing import (
    get_data,
    corrupted_simulation,
    prepare_data,
    load_simulation_data,
    save_simulation_data,
    get_simulation_path,
    load_config,
    save_result,
)

from .measure import (
    MSD,
    find_settling_time,
    print_simulation_report,
    compute_te_over_lags,
    AVAILABLE_METRICS,  # Good to expose this registry
)

__all__ = [
    # Graph Building
    "pos_nodes_uniform",
    "norm_multi",
    "local_connect_gaussian",
    "local_connect_lorentz",
    "connexion_normal_random_NUMBA",
    "connexion_normal_deterministic",
    "add_passive_nodes",
    # Evolution / Physics
    "evolve_system",
    "get_coupling_operator",
    "rk4_step",
    "euler_step",
    "map_step",
    "FitzHughNagumoModel",
    "HenonMapModel",
    # Data Processing
    "get_data",
    "corrupted_simulation",
    "prepare_data",
    "load_simulation_data",
    "save_simulation_data",
    "get_simulation_path",
    "load_config",
    "save_result",
    # Metrics
    "MSD",
    "find_settling_time",
    "print_simulation_report",
    "compute_te_over_lags",
    "AVAILABLE_METRICS",
]
