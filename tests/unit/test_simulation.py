"""Unit tests for RagoneSimulation.

pybamm.Simulation and pybamm.Experiment are mocked so no actual solver runs
occur. Only the constructor (which calls pybamm.step.VoltageTermination) and
the public API surface are exercised here.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pybamm
import pytest

from ragone.simulation import RagoneSimulation
from ragone.solution import RagoneSolution

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_pybamm_sol(time_final: float = 1.0, energy_final: float = 10.0) -> MagicMock:
    """Return a MagicMock that behaves like a pybamm Solution for the keys
    used by RagoneSimulation.solve()."""
    sol = MagicMock()

    def _getitem(key):
        m = MagicMock()
        if key == "Time [h]":
            m.entries = np.array([0.0, time_final])
        elif key == "Discharge energy [W.h]":
            m.entries = np.array([0.0, energy_final])
        else:
            m.entries = np.array([0.0, 1.0])
        return m

    sol.__getitem__.side_effect = _getitem
    return sol


# ---------------------------------------------------------------------------
# __init__ tests
# ---------------------------------------------------------------------------


class TestRagoneSimulationInit:
    """Tests for attribute initialisation — no pybamm solver calls."""

    @pytest.mark.unit
    def test_power_mode_sets_input_output(self, mock_model):
        sim = RagoneSimulation(mock_model, [1.0])
        assert sim.input == "Power [W]"
        assert sim.output == "Energy [W.h]"

    @pytest.mark.unit
    def test_power_mode_assigns_pybamm_step_fun(self, mock_model):
        sim = RagoneSimulation(mock_model, [1.0])
        assert sim.step_fun is pybamm.step.power

    @pytest.mark.unit
    def test_power_mode_ref_value(self, mock_model):
        # ref_value = capacity * (V_upper + V_lower) / 2
        #           = 5.0 * (4.2 + 2.5) / 2 = 16.75
        sim = RagoneSimulation(mock_model, [1.0])
        assert sim.ref_value == pytest.approx(5.0 * (4.2 + 2.5) / 2)

    @pytest.mark.unit
    def test_current_mode_sets_input_output(self, mock_model):
        sim = RagoneSimulation(mock_model, [1.0], mode="current")
        assert sim.input == "Current [A]"
        assert sim.output == "Capacity [A.h]"

    @pytest.mark.unit
    def test_current_mode_assigns_pybamm_step_fun(self, mock_model):
        sim = RagoneSimulation(mock_model, [1.0], mode="current")
        assert sim.step_fun is pybamm.step.current

    @pytest.mark.unit
    def test_current_mode_ref_value_equals_nominal_capacity(self, mock_model):
        sim = RagoneSimulation(mock_model, [1.0], mode="current")
        assert sim.ref_value == pytest.approx(5.0)

    @pytest.mark.unit
    def test_invalid_mode_raises_value_error(self, mock_model):
        with pytest.raises(ValueError, match="mode must be either"):
            RagoneSimulation(mock_model, [1.0], mode="invalid")

    @pytest.mark.unit
    def test_discharge_direction_sets_positive_sign(self, mock_model):
        sim = RagoneSimulation(mock_model, [1.0], direction="discharge")
        assert sim.sign == 1

    @pytest.mark.unit
    def test_charge_direction_sets_negative_sign(self, mock_model):
        sim = RagoneSimulation(mock_model, [1.0], direction="charge")
        assert sim.sign == -1

    @pytest.mark.unit
    def test_invalid_direction_raises_value_error(self, mock_model):
        with pytest.raises(ValueError, match="Invalid `direction`"):
            RagoneSimulation(mock_model, [1.0], direction="sideways")

    @pytest.mark.unit
    def test_falls_back_to_model_parameter_values(self, mock_model):
        sim = RagoneSimulation(mock_model, [1.0])
        assert sim.parameter_values is mock_model.default_parameter_values

    @pytest.mark.unit
    def test_explicit_parameter_values_override_model_defaults(
        self, mock_model, param_values
    ):
        custom = dict(param_values)
        sim = RagoneSimulation(mock_model, [1.0], parameter_values=custom)
        assert sim.parameter_values is custom
        assert sim.parameter_values is not mock_model.default_parameter_values

    @pytest.mark.unit
    def test_falls_back_to_model_solver(self, mock_model):
        sim = RagoneSimulation(mock_model, [1.0])
        assert sim.solver is mock_model.default_solver

    @pytest.mark.unit
    def test_explicit_solver_overrides_model_default(self, mock_model):
        custom_solver = MagicMock()
        sim = RagoneSimulation(mock_model, [1.0], solver=custom_solver)
        assert sim.solver is custom_solver

    @pytest.mark.unit
    def test_falls_back_to_model_var_pts(self, mock_model):
        sim = RagoneSimulation(mock_model, [1.0])
        assert sim.var_pts is mock_model.default_var_pts

    @pytest.mark.unit
    def test_explicit_var_pts_overrides_model_default(self, mock_model):
        custom_var_pts = {"x_n": 20}
        sim = RagoneSimulation(mock_model, [1.0], var_pts=custom_var_pts)
        assert sim.var_pts is custom_var_pts

    @pytest.mark.unit
    def test_value_range_is_stored(self, mock_model):
        value_range = [1.0, 2.0, 5.0]
        sim = RagoneSimulation(mock_model, value_range)
        assert sim.value_range is value_range

    @pytest.mark.unit
    def test_convert_to_watts_defaults_to_false(self, mock_model):
        sim = RagoneSimulation(mock_model, [1.0])
        assert sim.convert_to_watts is False

    @pytest.mark.unit
    def test_convert_to_watts_can_be_enabled(self, mock_model):
        sim = RagoneSimulation(mock_model, [1.0], convert_to_watts=True)
        assert sim.convert_to_watts is True


# ---------------------------------------------------------------------------
# solve() tests
# ---------------------------------------------------------------------------


class TestRagoneSimulationSolve:
    """Tests for solve() with pybamm.Simulation and pybamm.Experiment mocked."""

    @pytest.fixture
    def sim(self, mock_model):
        s = RagoneSimulation(mock_model, [5.0, 10.0])
        s.step_fun = MagicMock(return_value=MagicMock())
        return s

    @pytest.fixture
    def sim_current_watts(self, mock_model):
        s = RagoneSimulation(mock_model, [2.0], mode="current", convert_to_watts=True)
        s.step_fun = MagicMock(return_value=MagicMock())
        return s

    @pytest.mark.unit
    def test_returns_ragone_solution(self, sim):
        mock_sol = _mock_pybamm_sol(time_final=1.0)
        with (
            patch.object(sim, "_compute_theoretical_value", return_value=50.0),
            patch("pybamm.Experiment"),
            patch("pybamm.Simulation") as MockSim,
        ):
            MockSim.return_value.solve.return_value = mock_sol
            result = sim.solve()
        assert isinstance(result, RagoneSolution)

    @pytest.mark.unit
    def test_stores_solution_on_self(self, sim):
        mock_sol = _mock_pybamm_sol(time_final=1.0)
        with (
            patch.object(sim, "_compute_theoretical_value", return_value=50.0),
            patch("pybamm.Experiment"),
            patch("pybamm.Simulation") as MockSim,
        ):
            MockSim.return_value.solve.return_value = mock_sol
            result = sim.solve()
        assert sim.solution is result

    @pytest.mark.unit
    def test_solution_mode_matches_simulation_mode(self, sim):
        mock_sol = _mock_pybamm_sol(time_final=1.0)
        with (
            patch.object(sim, "_compute_theoretical_value", return_value=50.0),
            patch("pybamm.Experiment"),
            patch("pybamm.Simulation") as MockSim,
        ):
            MockSim.return_value.solve.return_value = mock_sol
            result = sim.solve()
        assert result.mode == "power"

    @pytest.mark.unit
    def test_solver_error_produces_nan_entries(self, sim):
        with (
            patch.object(sim, "_compute_theoretical_value", return_value=50.0),
            patch("pybamm.Experiment"),
            patch("pybamm.Simulation") as MockSim,
        ):
            MockSim.return_value.solve.side_effect = pybamm.SolverError("boom")
            result = sim.solve()
        assert np.all(np.isnan(result.data["Power [W]"]))
        assert np.all(np.isnan(result.data["Energy [W.h]"]))

    @pytest.mark.unit
    def test_stops_early_when_output_below_threshold(self, mock_model):
        """solve() breaks out of the loop when output < 10 % of theoretical."""
        # value=5.0, time_final=0.001 → output=0.005 < 0.1*50.0=5.0 → break
        sim = RagoneSimulation(mock_model, [5.0, 10.0, 20.0])
        sim.step_fun = MagicMock(return_value=MagicMock())
        mock_sol = _mock_pybamm_sol(time_final=0.001)
        with (
            patch.object(sim, "_compute_theoretical_value", return_value=50.0),
            patch("pybamm.Experiment"),
            patch("pybamm.Simulation") as MockSim,
        ):
            MockSim.return_value.solve.return_value = mock_sol
            sim.solve()
        assert MockSim.return_value.solve.call_count == 1

    @pytest.mark.unit
    def test_solution_data_has_expected_keys_power_mode(self, sim):
        mock_sol = _mock_pybamm_sol(time_final=1.0)
        with (
            patch.object(sim, "_compute_theoretical_value", return_value=50.0),
            patch("pybamm.Experiment"),
            patch("pybamm.Simulation") as MockSim,
        ):
            MockSim.return_value.solve.return_value = mock_sol
            result = sim.solve()
        assert "Power [W]" in result.data
        assert "Energy [W.h]" in result.data
        assert "Time [h]" in result.data

    @pytest.mark.unit
    def test_current_convert_to_watts_adds_watt_keys(self, sim_current_watts):
        mock_sol = _mock_pybamm_sol(time_final=1.0, energy_final=5.0)
        with (
            patch.object(
                sim_current_watts, "_compute_theoretical_value", return_value=50.0
            ),
            patch("pybamm.Experiment"),
            patch("pybamm.Simulation") as MockSim,
        ):
            MockSim.return_value.solve.return_value = mock_sol
            result = sim_current_watts.solve()
        assert "Power [W]" in result.data
        assert "Energy [W.h]" in result.data
