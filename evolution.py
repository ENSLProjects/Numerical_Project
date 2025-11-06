#!/usr/bin/env/python3

# ======================= Libraries

import numpy as np
import numba

# ======================= Functions


@numba.jit(nopython=True)
def count(data, axis: str):
    assert len(np.shape(data)) == 4, "wrong data shape input"
    if axis == "x":
        p = (0, 0)
    elif axis == "y":
        p = (0, 1)
    elif axis == "u":
        p = (1, 0)
    elif axis == "v":
        p = (1, 1)
    else:
        return "axis not existent"
    simu = np.sort(data[:, p[0], p[1], :], axis=0)
    k = len(simu[0])
    nb_points = [1] * k
    for i in range(k):
        start = simu[0, i]
        for j in range(1, len(simu)):
            if simu[j, i] != start:
                nb_points[i] += 1
                start = simu[j, i]
    return nb_points


@numba.jit(nopython=True)
def transfo_coupling_vec(X, Y, Eps, Adjacence):
    m, n = len(X), len(Y)
    assert m == n, "missmatch dimension"
    weights = np.sum(Adjacence, axis=1)
    if (weights == 0).any():
        raise ValueError(
            "some nodes have no edges/neighbors/connections with other nodes"
        )  # comme ça pas de problème de type sur la sortie
    transition = Eps * Adjacence * 1 / weights[:, np.newaxis] + np.identity(m)
    X_new = np.dot(transition, X) - Eps * X
    Y_new = np.dot(transition, Y) - Eps * Y
    return X_new, Y_new


@numba.jit(nopython=True)
def evolution_vec(X_0, Y_0, N, list_ab, Eps, Adjacence):
    """
    Warning: slicing on rows gives contiguous array while on columns it is not.

    Return the evolution with respect to time of the nodes.
    """
    n, m = len(X_0), len(Y_0)
    p = len(Adjacence)
    assert n == m, "wrong dimensions, both initial conditions don't have the same size"
    assert p == n, "missmatch dimensions"
    X = np.zeros((N, n))
    Y = np.zeros((N, n))
    UN = np.ones(p)
    X[0, :] = X_0
    Y[0, :] = Y_0
    X_c, Y_c = transfo_coupling_vec(X[0, :], Y[0, :], Eps, Adjacence)
    for t in range(1, N):
        X[t, :] = -list_ab[0, :] * X_c**2 + UN + Y_c
        Y[t, :] = list_ab[1, :] * X_c
        # This slice (a row) is contiguous by default!
        X_c, Y_c = transfo_coupling_vec(X[t, :], Y[t, :], Eps, Adjacence)
    return X, Y


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


def MSD(G, X, axe=1):
    """
    Return the MSD in the order:
    -- axe=1: esperance_t(variance_node(t))
    -- axe=0: esperance_node(variance_t(node))
    """
    n, N = X.shape
    assert n == len(G), "wrong dimension"
    MSD_values = np.std(X, axis=axe)
    return np.mean(MSD_values)


def MSD_inverse(G, X, axe=0):
    """
    Return the MSD in the order:
    -- axe=1: esperance_t(variance_node(t))
    -- axe=0: esperance_node(variance_t(node))
    """
    n, N = X.shape
    assert n == len(G), "wrong dimension"
    MSD_values = np.mean(X, axis=axe)
    return np.std(MSD_values)
