#!/usr/bin/env/python3

# ======================= Libraries

import numpy as np
import numba

# ======================= Functions

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
