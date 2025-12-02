#!/usr/bin/env/python3

# ======================= Libraries


import h5py

with h5py.File("experiment_2023.h5", "r") as f:
    # Read metadata without loading array
    print(f["run_01"].attrs["epsilon"])

    # Read specific slice (RAM efficient)
    # Only loads the first 10 time steps
    partial_data = f["run_01"]["trajectory"][:10, :, :]
