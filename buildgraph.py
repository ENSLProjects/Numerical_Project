#!/usr/bin/env/python3

# ======================= Libraries
import numpy as np
from numpy.random import Generator
from scipy.spatial.distance import pdist, squareform


# ======================= Functions
def pos_nodes_uniform(N: int, xmax: float, ymax: float, rng: Generator):
    """
    return N points randomly distributed with uniform law in the rectangle of vertex [(0, 0), (0, xmax), (xmax, ymax), (ymax, 0)]
    """
    xpos = rng.uniform(0.0, xmax, N)
    ypos = rng.uniform(0.0, ymax, N)
    return np.column_stack((xpos, ypos)).T


def pos_nodes_normal(N: int, mean: float, std: float, rng: Generator):
    """
    return N points randomly distributed with normal law of mean mean and standard deviation std
    """
    pos = rng.normal(loc=mean, scale=std, size=(2, N))
    return pos


def norm_multi(x, y, p=2):
    """
    x -- matrix of coordinates (2,N)
    y -- vector of one node (2,)
    Compute the p-norm column by column
    """
    (n, m) = np.shape(x)
    reshaped = y[:, np.newaxis]
    assert n == 2, "The entire code assumes the nodes are in a 2D space"
    return np.linalg.norm(x - reshaped, axis=0, ord=p)


def local_connect_gaussian(x, center, std: float):
    """
    This is the gaussian function that return the probabilty to connect two nodes depending their distance.
    This function has to be fixed.
    """
    assert std != 0, "Diving by zero"
    var = std**2
    return np.exp(-(norm_multi(x, center) ** 2) / 2 / var)


def local_connect_lorentz(x, center, gamma: float, order=2):
    """
    This is the long (heavy) tail function that return the probabilty to connect two nodes depending their distance.
    This function has to be fixed with gamma>0.5 in order to have a heavier tail than the gaussian distribution with std=gamma.
    """
    assert gamma != 0, "Dividing by zero"
    return 1 / (1 + (norm_multi(x, center, order) / gamma) ** order)


def connexion_normal_random(
    pos, rng: Generator, std: float, mean: float, std_draw: float
):
    """
    Return the connectivity matrix of the final graph
    """
    (n, m) = np.shape(pos)  # WARNING this code assume n=2
    number_of_neighbours = rng.normal(
        mean, std_draw, m
    )  # the number of neigbours for each node
    if (number_of_neighbours > m).any():
        raise ValueError("You gave to some nodes more neighbours than possible")
    random_draw = rng.uniform(size=(m, m))
    distance_proba = np.zeros((m, m))
    for i in range(m):
        center = pos[:, i]
        k_neighbours = max(
            0, int(number_of_neighbours[i])
        )  # the number of neighbours is always positive
        chosen_index = rng.choice(m, k_neighbours, replace=False)
        neighbours = pos[:, chosen_index]
        distance_proba[chosen_index, i] = local_connect_gaussian(
            neighbours, center, std
        )
    # connectivity = np.sign(distance_proba-random_draw)
    connectivity = (distance_proba > random_draw).astype(int)
    np.fill_diagonal(connectivity, 0.0)
    return connectivity


def connexion_normal_deterministic(pos, rng: Generator, std: float):
    """
    Return the connectivity matrix of the final graph
    """
    (n, m) = np.shape(pos)
    assert n == 2, "The entire code assumes the nodes are in a 2D space"
    assert std != 0, "Diving by zero"
    distances_matrix = squareform(pdist(pos.T, metric="euclidean"))
    var = std**2
    distance_proba = np.exp(-(distances_matrix**2) / (2 * var))
    random_draw = rng.uniform(size=(m, m))
    connectivity = (distance_proba > random_draw).astype(int)
    np.fill_diagonal(connectivity, 0)
    return connectivity


# ======================= Script
if __name__ == "__main__":
    print("Hello world!")
