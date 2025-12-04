#!/usr/bin/env/python3

# ======================= Libraries

import os
import h5py
from typing import cast
import numpy as np

# ======================= Functions


def prepare_data(arr):
    """
    Prépare les données pour la lib C Entropy.

    La sortie respecte ces règles :

    Règle 1 : Format (1, N_samples) obligatoire (donc 1 ligne, N colonnes).
    Règle 2 : Mémoire contiguë (C-contiguous).
    Règle 3 : Type float64 (double).
    """
    # Si c'est un vecteur plat (N,), on le passe en (1, N)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    # Si c'est (N, 1), on transpose en (1, N)
    elif arr.shape[0] > arr.shape[1]:
        arr = arr.T

    return np.ascontiguousarray(arr, dtype=np.float64)


def get_data(container, key):
    """
    Helper to safely extract a dataset and cast it for the linter.
    """
    return cast(h5py.Dataset, container[key])[:]


def corrupted_simulation(file_path):
    """
    Verifies that the tensor of data has no NaN or Inf data.

    Args:
        file_path (str): Relative path to the simulation file.
    """
    filename = os.path.basename(file_path)

    if not os.path.exists(file_path):
        print(f"CRITICAL ERROR: File not found at {file_path}")
        return True  # Return True (Corrupted/Error) so the script knows to stop

    with h5py.File(file_path, "r") as f:
        print(f"--- File found: {filename} ---")

        # 1. LOAD DATA DIRECTLY FROM ROOT
        try:
            trajectory = get_data(f, "trajectory")
        except KeyError:
            print("Error: 'trajectory' dataset not found at file root.")
            return True

        print("\n" + "=" * 30)
        print("   NUMERICAL DIAGNOSTIC")
        print("=" * 30)

        has_nan = np.isnan(trajectory).any()
        has_inf = np.isinf(trajectory).any()
        max_val = np.max(trajectory)
        min_val = np.min(trajectory)

        print(f"Data Shape:    {trajectory.shape}" + "(Time, Node, Physical variable)")
        print(f"Contains NaNs? {has_nan}")
        print(f"Contains Infs? {has_inf}")

        if has_nan or has_inf:
            print(
                "\n[CONCLUSION] ❌ The simulation EXPLODED. Try to change the Runge-Kutta time step."
            )
            return True  # True means it IS corrupted
        elif max_val == 0 and min_val == 0:
            print("\n[CONCLUSION] ⚠️ The simulation is FLAT (All Zeros).")
            return True
        else:
            print("\n[CONCLUSION] ✅ Data looks valid.")
            return False  # False means NOT corrupted (Healthy)


def pull_out_full_data(file_path):
    with h5py.File(file_path, "r") as f:
        trajectory = get_data(f, "trajectory")

        parameters = load_dict_from_hdf5(f)

    return {"time trajectory": trajectory, "parameters": parameters}


def save_dict_to_hdf5(h5_group, dic):
    """
    Recursively saves a dictionary to an HDF5 group.

    Args:
        h5_group: The HDF5 Group or File object to write into.
        dic: The dictionary to save.
    """
    for key, item in dic.items():
        # Case 1: Item is a nested dictionary -> Create a Group and Recurse
        if isinstance(item, dict):
            # Create the subgroup and capture the object
            subgroup = h5_group.create_group(key)
            # Pass THIS subgroup object to the recursive call
            save_dict_to_hdf5(subgroup, item)

        # Case 2: Item is an Array or List -> Create a Dataset
        elif isinstance(item, (np.ndarray, list)):
            h5_group.create_dataset(key, data=item)

        # Case 3: Item is a simple value -> Save as Attribute
        else:
            # Save directly to the attributes of the current group object
            try:
                h5_group.attrs[key] = item
            except TypeError:
                # Fallback for types HDF5 doesn't like (e.g. None)
                h5_group.attrs[key] = str(item)


def load_dict_from_hdf5(h5_group):
    """
    Recursively reconstructs a dictionary from an HDF5 group.

    Args:
        h5_group: The HDF5 Group or File object to read from.
    """
    ans = {}

    # 1. Load Attributes (Metadata)
    for key, val in h5_group.attrs.items():
        # Handle standard string decoding
        if isinstance(val, bytes):
            val = val.decode("utf-8")
        ans[key] = val

    # 2. Load Items (Datasets or Subgroups)
    # .items() iterates over the direct children of the group
    for key, item in h5_group.items():
        if isinstance(item, h5py.Dataset):
            # It's a dataset -> load value
            ans[key] = item[()]
        elif isinstance(item, h5py.Group):
            # It's a group -> recurse using the group object
            ans[key] = load_dict_from_hdf5(item)

    return ans
