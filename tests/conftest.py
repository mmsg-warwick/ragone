import matplotlib

matplotlib.use("Agg")
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import numpy as np
import pytest

# Minimal parameter values that satisfy RagoneSimulation.__init__
# without needing a real pybamm.ParameterValues object.
PARAM_VALUES = {
    "Nominal cell capacity [A.h]": 5.0,
    "Upper voltage cut-off [V]": 4.2,
    "Lower voltage cut-off [V]": 2.5,
}


@pytest.fixture(autouse=True)
def close_plots():
    """Close all matplotlib figures after every test to prevent memory leaks."""
    yield
    plt.close("all")


@pytest.fixture
def param_values():
    """Return a fresh copy of the minimal parameter dict."""
    return dict(PARAM_VALUES)


@pytest.fixture
def mock_model(param_values):
    """Return a MagicMock that mimics a pybamm model's essential attributes."""
    model = MagicMock()
    model.default_parameter_values = param_values
    model.default_solver = MagicMock()
    model.default_var_pts = {}
    return model


@pytest.fixture
def power_data():
    return {
        "Power [W]": np.array([1.0, 5.0, 10.0, 50.0]),
        "Energy [W.h]": np.array([20.0, 18.0, 15.0, 8.0]),
        "Time [h]": np.array([20.0, 3.6, 1.5, 0.16]),
    }


@pytest.fixture
def power_solution(power_data):
    from ragone.solution import RagoneSolution

    return RagoneSolution(power_data, "power")


@pytest.fixture
def current_data():
    return {
        "Current [A]": np.array([1.0, 2.0, 5.0, 10.0]),
        "Capacity [A.h]": np.array([5.0, 4.8, 4.2, 3.5]),
        "Time [h]": np.array([5.0, 2.4, 0.84, 0.35]),
    }


@pytest.fixture
def current_solution(current_data):
    from ragone.solution import RagoneSolution

    return RagoneSolution(current_data, "current")
