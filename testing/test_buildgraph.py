import pytest
import numpy as np
from numpy.random import default_rng

from bnn_package import buildgraph

# ======================= Fixtures (Setup)


@pytest.fixture
def graph_params():
    """Provides standard parameters for graph building tests."""
    rng = default_rng(42)  # Fixed seed for reproducibility
    return {"rng": rng, "N": 10, "xmax": 100.0, "ymax": 100.0}


@pytest.fixture
def evo_params():
    """Provides parameters for evolution tests."""
    m = 5  # number of nodes
    return {
        "m": m,
        "n": 5,
        "Eps": 0.1,
        "adj": np.eye(m),  # Self-connected for simplicity
    }


# ======================= Tests for buildgraph.py


def test_pos_nodes_uniform_shape(graph_params):
    """Check if it generates the correct matrix shape (2, N)"""
    p = graph_params
    pos = buildgraph.pos_nodes_uniform(p["N"], p["xmax"], p["ymax"], p["rng"])

    assert pos.shape == (2, p["N"])


def test_pos_nodes_uniform_bounds(graph_params):
    """Check if generated nodes are actually within the rectangle"""
    p = graph_params
    pos = buildgraph.pos_nodes_uniform(p["N"], p["xmax"], p["ymax"], p["rng"])

    # Check x coordinates
    assert np.all(pos[0, :] >= 0)
    assert np.all(pos[0, :] <= p["xmax"])
    # Check y coordinates
    assert np.all(pos[1, :] >= 0)
    assert np.all(pos[1, :] <= p["ymax"])


def test_norm_multi():
    """Test the Euclidean distance calculation logic"""
    x = np.array([[0, 3, 0], [0, 0, 4]])
    y = np.array([0, 0])

    distances = buildgraph.norm_multi(x, y)
    expected = np.array([0.0, 3.0, 4.0])

    np.testing.assert_allclose(distances, expected)


def test_local_connect_gaussian():
    """Test if probability is 1.0 when distance is 0"""
    center = np.array([0, 0])
    x = np.array([[0], [0]])
    std = 1.0
    prob = buildgraph.local_connect_gaussian(x, center, std)
    assert prob[0] == 1.0
