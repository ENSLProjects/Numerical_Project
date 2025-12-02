import pytest
import numpy as np
from numpy.random import default_rng

from bnn_package import measure

# ======================= Fixtures (Setup)

rng = default_rng(42)


@pytest.fixture
def graph_params():
    """Provides standard parameters for graph building tests."""
    return {"rng": rng, "N": 10, "xmax": 10.0, "ymax": 10.0}


@pytest.fixture
def evo_params():
    """Provides parameters for evolution tests."""
    m = 5  # number of nodes
    return {
        "m": m,
        "n": 1000,
        "Eps": 0.1,
        "adj": np.eye(m),  # Self-connected for simplicity
    }


# ======================= Tests for measure.py


def test_prepare_data_reshaping():
    """Test if 1D arrays are correctly reshaped to (1, N)"""
    arr_1d = np.array([1, 2, 3, 4, 5])
    res = measure.prepare_data(arr_1d)

    assert res.shape == (1, 5)
    assert res.flags["C_CONTIGUOUS"]
    assert res.dtype == np.float64


def test_find_settling_time_exact():
    """Test settling time logic on a synthetic signal"""
    # Signal 0 -> 100 at index 10
    signal = np.concatenate([np.zeros(10), np.ones(10) * 100])

    result = measure.find_settling_time(signal, final_n_samples=5, tolerance_percent=1)
    idx = result if isinstance(result, (int, np.integer)) else result[0]

    # It should start "settling" around index 10
    assert 9 <= idx <= 11


def test_MSD_xy_shape():
    """Test Mean Squared Displacement shape"""
    n_nodes = 3
    n_time = 5

    G = np.ones(n_nodes)
    X = np.zeros((n_nodes, n_time))
    Y = np.zeros((n_nodes, n_time))

    msd = measure.MSD_xy(G, X, Y)

    assert msd.shape == (n_time,)
