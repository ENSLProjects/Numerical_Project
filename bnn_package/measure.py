#!/usr/bin/env/python3

# ======================= Libraries

import numpy as np
import numba
from scipy.spatial.distance import pdist

# ======================= Functions

@numba.jit(nopython=True)
def MSD_xy(G, X, Y):
    """
    Return the MSD for a given epsilon
    """
    n, N = X.shape
    assert n == len(G), "wrong dimension"
    MSD_values = np.zeros(N)
    mean_X = np.mean(X, axis=0)
    mean_Y = np.mean(Y, axis=0)
    for t in range(N):
        MSD_t = np.mean((X[:, t] - mean_X[t]) ** 2 + (Y[:, t] - mean_Y[t]) ** 2)
        MSD_values[t] = MSD_t
    return MSD_values


def MSD_vec_xy(G, X, Y):
    """
    Return the MSD for a given epsilon
    """
    n, N = X.shape
    assert n == len(G), "wrong dimension"
    MSD_values = np.zeros(N)
    mean_X = np.mean(X, axis=0)
    mean_Y = np.mean(Y, axis=0)
    MSD_values = np.mean(
        (X - mean_X[np.newaxis, :]) ** 2 + (Y - mean_Y[np.newaxis, :]) ** 2, axis=1
    )
    return MSD_values


def MSD(G, X, average=True, axe=1):
    """
    Return the MSD in the order:
    -- axe=1: esperance_t(variance_node(t))
    -- axe=0: esperance_node(variance_t(node))
    """
    (n, N) = np.shape(X)
    assert N == len(G), "wrong dimension"
    msd_values = np.std(X, axis=axe)
    if average:
        return np.mean(msd_values)
    else:
        return msd_values


def MSD_inverse(G, X, axe=0):
    """
    Return the MSD in the order:
    -- axe=1: esperance_t(variance_node(t))
    -- axe=0: esperance_node(variance_t(node))
    """
    n, N = X.shape
    assert N == len(G), "wrong dimension"
    MSD_values = np.mean(X, axis=axe)
    return np.std(MSD_values)


@numba.jit(nopython=True)
def Synchronized_error(X, Y, N_nodes, time): #not easily vectorizable 
    assert N_nodes > 1, "Require at least two nodes"
    error = 0 
    for i in range(N_nodes - 1):
        for j in range(i+1, N_nodes):
            error += np.sqrt(
                (X[time, i] - X[time, j]) ** 2 + (Y[time, i] - Y[time, j]) ** 2
            )
    return 2 * error / N_nodes / (N_nodes - 1)

def Synchronized_error_mean():
    return 4
    
