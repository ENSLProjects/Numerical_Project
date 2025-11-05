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
    n, m = len(X_0), len(Y_0)
    p = len(Adjacence)
    assert n == m, "wrong dimensions, both initial conditions don't have the same size"
    X = np.zeros((n, N))
    Y = np.zeros((n, N))
    UN = np.ones(p)
    X[:, 0] = X_0
    Y[:, 0] = Y_0
    X_c, Y_c = transfo_coupling_vec(X[:, 0], Y[:, 0], Eps, Adjacence)
    for t in range(1, N):
        X[:, t] = -list_ab[0, :] * X_c**2 + UN + Y_c
        Y[:, t] = list_ab[1, :] * X_c
        X_c, Y_c = transfo_coupling_vec(X[:, t], Y[:, t], Eps, Adjacence)
    return X, Y
