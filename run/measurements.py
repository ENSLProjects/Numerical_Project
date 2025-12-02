#!/usr/bin/env/python3

# ======================= Libraries
import entropy.entropy as ee
import h5py
# ======================= Parameters

n_embed = 2
kNN = 5
Npoints = 4096

filepath = "data_simulation/2025-11-27/FhN_22-30-48_eps0.30_nodes200.h5"



with h5py.File(filepath, "r") as f:
        
        # A. LOAD PARAMETERS

        eps = f.attrs["epsilon"]
        model = f.attrs["model"]
        
        # B. LOAD DATA
        # We use [:] to load the dataset from disk into RAM

        traj = f["trajectory"][:]  # pyright: ignore[reportIndexIssue]
        adj = f["adjacency"][:] # pyright: ignore[reportIndexIssue]
        
        print(f" > Parameters: eps={eps}, model={model}")
        print(f" > Data Shape: {traj.shape}") # pyright: ignore[reportAttributeAccessIssue]
        

