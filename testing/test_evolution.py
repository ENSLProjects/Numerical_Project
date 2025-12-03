import pytest
import numpy as np
from numpy.random import default_rng

from bnn_package import evolution

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


# ======================= Tests for evolution.py


def test_transfo_coupling_vec_dimensions(evo_params):
    """Ensure the coupling transformation maintains input shapes"""
    p = evo_params
    X = np.ones(p["m"])
    Y = np.ones(p["m"])

    X_new, Y_new = evolution.transfo_coupling_vec(X, Y, p["Eps"], p["adj"])

    assert X_new.shape == (p["m"],)
    assert Y_new.shape == (p["m"],)


def test_transfo_coupling_no_change_if_epsilon_zero(evo_params):
    """If Eps is 0, the system should not change based on neighbors"""
    p = evo_params
    X = np.random.rand(p["m"])
    Y = np.random.rand(p["m"])

    X_new, Y_new = evolution.transfo_coupling_vec(X, Y, 0.0, p["adj"])

    np.testing.assert_array_equal(X, X_new)
    np.testing.assert_array_equal(Y, Y_new)


def test_evolution_vec_output_shape(evo_params):
    """Test the time evolution loop output shape"""
    p = evo_params
    N_steps = 10
    X_0 = np.random.rand(p["m"])
    Y_0 = np.random.rand(p["m"])
    list_ab = np.ones((2, p["m"]))

    X_res, Y_res = evolution.evolution_vec(
        X_0, Y_0, N_steps, list_ab, p["Eps"], p["adj"]
    )

    assert X_res.shape == (N_steps, p["m"])
    assert Y_res.shape == (N_steps, p["m"])
