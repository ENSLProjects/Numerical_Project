
# Numerical_Project
Simulation of a biological neural network with a random graph, compute the transfert entropy with respect to a time lack and nodes indexes in order to detect a phase transition in the graph.

# ⚙️ Installation and Environment Setup

This project, **bnn-package**, requires specific system libraries (GSL, FFTW) and a local compilation step for the `entropy` package to function correctly. Follow these steps to set up your environment on macOS.

## Prerequisites (macOS)

Before proceeding, ensure you have the following installed:

1.  **Xcode Command Line Tools:** Necessary for the `gcc`/`g++` compilers and `make`.
    ```bash
    xcode-select --install
    ```

2.  **Homebrew:** The package manager for macOS.
    * Installation instructions can be found on the [Homebrew website](https://brew.sh/).

3.  **Python 3:** Ensure Homebrew's Python is installed to provide necessary utilities like `python3-config`.
    ```bash
    brew install python3
    ```

---

## Step 0: Dowload the code source from Nicolas Garnier (ENS de Lyon)

**Source code** for the information theory part [entropy library in C/C++](https://github.com/nbgarnier/entropy)

**Documentation:** Full documentation of the source code is available at this [link](https://perso.ens-lyon.fr/nicolas.garnier/files/html/index.html)

## Step 1: Install System Dependencies (GSL & FFTW)

These are the libraries written in C/C++ that the `entropy` package depends on.

```bash
brew install gsl fftw
```

## Step 2: Create the package

```bash
./configure CFLAGS=-I/opt/homebrew/include LDFLAGS=-L/opt/homebrew/lib
```

## Step 3: Compile for python

```bash
make python
```

# 🏗️ Project Architecture

The project follows a strict separation of concerns between the **Core Physics Library** (`src/bnn_package`) and the **Execution Layer** (`run`).

* **`src/bnn_package/`**: Contains the reusable scientific code. It defines the physics engines, graph topology generators, information-theoretic measures, and data handling utilities. It is highly optimized with `numba` for high-performance computing.
* **`run/`**: Contains the orchestration scripts. It handles configuration parsing, parameter sweeping (grid search), and parallel processing execution using `multiprocessing`.

---

## 1. Execution Layer (**`run/`**)

This directory manages the simulations. It reads YAML configuration files to define experiments and distributes the workload across CPU cores.

### `run/runner.py`
**The Orchestrator.**
* **Role:** The main entry point for all simulations.
* **Key Features:**
    * ***Adaptive Environment Setup:*** Automatically detects if the simulation is running in **Parallel** or **Sequential** mode (via config) and sets environment variables (e.g., `OMP_NUM_THREADS`, `LOKY_MAX_CPU_COUNT`) to optimize for either throughput or latency and patch potential warning withs backend packages using their own multiprocessing pool (MKL, BLAS, scikit-learn *etc*)
    * ***Task Generation:*** Reads a YAML configuration file and generates a list of task dictionaries by computing the Cartesian product of all list parameters (Grid Search).
    * ***Parallel Execution:*** Spawns a `multiprocessing.Pool` of workers to execute tasks concurrently, displaying a global progress bar via `tqdm`.

### `run/workers.py`
**The Simulation Logic.**
* **Role:** Defines the specific routines that run inside the worker processes spawned by `runner.py`.
* **Worker Functions:**
    * `time_series(params)`: Runs a single simulation and saves the full raw trajectory (Voltage vs Time) to an HDF5 file. Used for detailed visual and time series analysis.
    * `run_order_parameter(params)`: Runs a simulation but only returns scalar metrics (e.g., Synchronization Error) at the end. Efficient for phase diagram sweeps.
    * `research_alignment_worker(params)`: A specialized worker for Information Theory research. It simulates the system, keep it on RAM then computes Transfer Entropy (TE) across multiple lags, calculates KL Divergence and Canonical Correlation Analysis against theoretical propagators of the graph, and returns statistical metrics without saving the raw heavy data.

### `run/configs/*.yaml`
**Experiment Definitions.**
* ***Role:*** Declarative files that define what to run.
* ***Structure:*** They control physical parameters (`epsilon`, `cr`, `fhn_eps` *etc*), simulation settings (`dt`, `total_time` *etc*), and execution modes. Lists in these files (e.g., `epsilon: [0.01, 0.02]`) are automatically expanded into parameter sweeps by the runner. They also control the multiprocessing with flags `parallel` and `cores_ratio`. Tey control which order parameters are computed in `run_order_parameter` mode, what are the parameters for the Transfert Entropy computation in `research_alignment_worker` mode.  They also allow to use the exact same graph for many simulations with the `existing_graph_path` flag.

---

## 2. Core Library (**`src/bnn_package/`**)

This package contains the scientific core of the project. Most computational heavy lifting is JIT-compiled using `numba`.

### `src/bnn_package/evolution.py`
**The Physics Engine.**
* **Role:** Defines the dynamical systems and numerical integrators.
* **How it works:**
    * ***Models:*** Uses `numba.experimental.jitclass` to define high-performance classes for the **FitzHugh-Nagumo** model and **Henon Maps**. It handles the interplay between active nodes, passive nodes, and Diffusive/Laplacian coupling.
    * ***Integrators:*** Implements `rk4_step` (Runge-Kutta 4) and `euler_step` as JIT-compiled functions that take a model instance and step the state forward in time.

### `src/bnn_package/measure.py`
**The Metric Calculator.**
* **Role:** Computes complex observables and information-theoretic quantities.
* **Key Features:**
    * ***Transfer Entropy (TE):*** Wraps the Kraskov (KSG) estimator. It includes a robust implementation (`compute_te_over_lags`) that handles time-lagged mutual information to measure information flow. It includes safety checks for "flat-line" signals to prevent C-library crashes. Fully based on [entropy library in C/C++](https://github.com/nbgarnier/entropy).
    * ***Structure-Dynamics Matching:*** Implements `compute_kl_divergence` and `compute_cca_score` (Canonical Correlation Analysis) to quantify how well the measured information flow matches theoretical predictions (e.g., the Laplacian propagator).

### `src/bnn_package/buildgraph.py`
**The Topology Generator.**
* **Role:** Generates the network structures on which the physics evolves.
* **How it works:**
    * Creates spatially embedded graphs (nodes have 2D coordinates).
    * Implements connectivity rules like `connexion_normal_deterministic` (Gaussian distance-based connection). Several are implemented: Power laws, Lorentz distribution. The connection algorithm can also be random (*i.e.* each node have a random number of neighbors following a Gaussian distribution).
    * ***Passive Nodes:*** Contains logic (`add_passive_nodes`) to attach non-dynamical "passive" nodes to the active backbone based on Poisson distributions, adding heterogeneity to the network.

### `src/bnn_package/data_processing.py`
**The I/O Handler.**
* **Role:** Manages file input/output to ensure data integrity.
* **How it works:**
    * ***HDF5 Management:*** Functions like `save_simulation_data` handle the efficient binary storage of massive simulation trajectories using compression with shuffle and moving data on Float32.
    * ***Safety Checks:*** Includes `corrupted_simulation` to scan output files for `NaN` or `Inf` values, diagnosing numerical explosions automatically.
    * ***Memory Layout:*** Contains `prepare_data` to ensure numpy arrays are C-contiguous and correctly shaped before being passed to external C libraries.

### `src/bnn_package/learning.py`
**The Learning Module.**
* **Role:** Implements Reservoir Computing and Evolutionary Optimization.
* **How it works:**
    * ***Reservoir Computing:*** Defines an `FhnReservoir` class that uses the physical graph as a computational substrate to solve tasks (e.g., chaotic time-series prediction).
    * ***Evolutionary Strategy:*** Uses `cma` (Covariance Matrix Adaptation) to evolve the edge weights of the graph, optimizing the network topology to maximize task performance (e.g., bridging a "canyon" in the network structure).
