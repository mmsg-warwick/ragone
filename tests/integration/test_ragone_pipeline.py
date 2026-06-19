"""Integration tests for the Ragone pipeline.

These tests exercise multiple real components working together:
  - RagoneSolution  <-> RagonePlot
  - RagoneSolution.fit_log() <-> RagonePlot(fit=True)
  - RagoneSimulation.solve() -> RagoneSolution -> RagonePlot
  - config helpers <-> RagoneSimulation

pybamm.Simulation is still mocked to keep the tests fast, but all ragone
modules run for real so that cross-module contracts are verified.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from ragone.simulation import RagoneSimulation
from ragone.solution import RagoneSolution
from ragone.plotting import RagonePlot


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_pybamm_sol(time_final: float = 1.0, energy_final: float = 10.0) -> MagicMock:
    """Return a MagicMock that behaves like a pybamm Solution."""
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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def aged_solutions():
    """Three RagoneSolutions simulating progressive capacity fade."""
    solutions = []
    for scale in [1.0, 0.85, 0.70]:
        data = {
            "Power [W]": np.array([1.0, 5.0, 10.0, 50.0]),
            "Energy [W.h]": np.array([20.0, 18.0, 15.0, 8.0]) * scale,
            "Time [h]": np.array([20.0, 3.6, 1.5, 0.16]) * scale,
        }
        solutions.append(RagoneSolution(data, "power"))
    return solutions


@pytest.fixture
def current_watts_solution():
    """A current-mode solution that also carries watt keys (convert_to_watts)."""
    data = {
        "Current [A]": np.array([1.0, 2.0, 5.0, 10.0]),
        "Capacity [A.h]": np.array([5.0, 4.8, 4.2, 3.5]),
        "Time [h]": np.array([5.0, 2.4, 0.84, 0.35]),
        "Power [W]": np.array([3.35, 6.72, 16.8, 32.0]),
        "Energy [W.h]": np.array([16.75, 16.1, 14.1, 11.2]),
    }
    return RagoneSolution(data, "current")


# ---------------------------------------------------------------------------
# Group 1: RagoneSolution <-> RagonePlot
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSolutionToPlot:
    """Solution objects integrate correctly with RagonePlot."""

    def test_power_solution_plot_returns_fig_and_ax(self, power_solution):
        fig, ax = power_solution.plot()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

    def test_current_solution_plot_returns_fig_and_ax(self, current_solution):
        fig, ax = current_solution.plot()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

    def test_plot_x_limits_contain_data_range(self, power_solution):
        fig, ax = power_solution.plot()
        xlim = ax.get_xlim()
        assert xlim[0] <= power_solution.data["Power [W]"].min()
        assert xlim[1] >= power_solution.data["Power [W]"].max()

    def test_plot_y_limits_contain_max_output(self, power_solution):
        fig, ax = power_solution.plot()
        ylim = ax.get_ylim()
        assert ylim[1] >= power_solution.data["Energy [W.h]"].max()

    def test_linear_scale_plot_succeeds(self, power_solution):
        fig, ax = power_solution.plot(scale="linear")
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

    def test_multiple_solutions_produce_multiple_lines(self, aged_solutions):
        plot = RagonePlot(aged_solutions)
        fig, ax = plot.plot()
        assert isinstance(fig, plt.Figure)
        assert len(ax.lines) >= len(aged_solutions)

    def test_multiple_solutions_with_labels_creates_legend(self, aged_solutions):
        labels = ["Cycle 0", "Cycle 100", "Cycle 200"]
        plot = RagonePlot(aged_solutions, labels=labels)
        fig, ax = plot.plot()
        legend = ax.get_legend()
        assert legend is not None
        legend_texts = [t.get_text() for t in legend.get_texts()]
        assert legend_texts == labels

    def test_secondary_axes_created_with_volume(self, power_solution):
        fig, ax = power_solution.plot(volume=0.01)
        # In matplotlib 3.11+ secondary axes are child_axes, not fig.axes entries
        assert len(ax.child_axes) >= 2

    def test_secondary_axes_created_with_mass(self, power_solution):
        fig, ax = power_solution.plot(mass=0.05)
        assert len(ax.child_axes) >= 2

    def test_data_limits_span_all_solutions(self, aged_solutions):
        plot = RagonePlot(aged_solutions)
        # min_input should match the minimum across all solutions
        expected_min = min(sol.data["Power [W]"].min() for sol in aged_solutions)
        expected_max = max(sol.data["Energy [W.h]"].max() for sol in aged_solutions)
        assert plot.min_input == pytest.approx(expected_min)
        assert plot.max_output == pytest.approx(expected_max)

    def test_mixed_mode_solutions_require_watt_keys(
        self, power_solution, current_solution
    ):
        """Mixing power and current solutions without watt keys raises ValueError."""
        with pytest.raises(ValueError, match="convert_to_watts"):
            RagonePlot([power_solution, current_solution])

    def test_mixed_mode_solutions_work_with_watt_keys(
        self, power_solution, current_watts_solution
    ):
        plot = RagonePlot([power_solution, current_watts_solution])
        fig, ax = plot.plot()
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# Group 2: fit_log() <-> RagonePlot
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFitAndPlot:
    """fit_log() and RagonePlot(fit=True) work together correctly."""

    def test_fit_log_then_plot_succeeds(self, power_solution):
        power_solution.fit_log()
        fig, ax = power_solution.plot()
        assert isinstance(fig, plt.Figure)

    def test_metrics_survive_plotting(self, power_solution):
        power_solution.fit_log()
        metrics_before = dict(power_solution.metrics)
        power_solution.plot()
        assert power_solution.metrics == metrics_before

    def test_fit_true_auto_fits_unfitted_solution(self, power_solution):
        assert power_solution._raw_metrics is None
        plot = RagonePlot(power_solution, fit=True)
        plot.plot()
        # RagonePlot should have triggered fit_log()
        assert power_solution._raw_metrics is not None
        assert power_solution.metrics is not None

    def test_fit_true_uses_existing_metrics_when_already_fitted(self, power_solution):
        power_solution.fit_log()
        original_metrics = dict(power_solution.metrics)
        plot = RagonePlot(power_solution, fit=True)
        plot.plot()
        assert power_solution.metrics == original_metrics

    def test_fit_true_adds_extra_line_for_fitted_curve(self, power_solution):
        # Without fit
        plot_no_fit = RagonePlot(power_solution, fit=False)
        _, ax_no_fit = plot_no_fit.plot()
        lines_no_fit = len(ax_no_fit.lines)

        # With fit
        plot_fit = RagonePlot(power_solution, fit=True)
        _, ax_fit = plot_fit.plot()
        lines_fit = len(ax_fit.lines)

        assert lines_fit > lines_no_fit

    def test_fit_log_with_partial_nan_data_still_plots(self):
        data = {
            "Power [W]": np.array([1.0, 5.0, 10.0, 50.0]),
            "Energy [W.h]": np.array([20.0, 18.0, np.nan, 8.0]),
            "Time [h]": np.array([20.0, 3.6, np.nan, 0.16]),
        }
        solution = RagoneSolution(data, "power")
        solution.fit_log()
        fig, ax = solution.plot()
        assert isinstance(fig, plt.Figure)

    def test_fit_metrics_have_required_keys(self, power_solution):
        power_solution.fit_log()
        assert "n" in power_solution.metrics
        assert "Reference energy [W.h]" in power_solution.metrics
        assert "Reference power [W]" in power_solution.metrics

    def test_fit_multiple_aged_solutions(self, aged_solutions):
        for sol in aged_solutions:
            sol.fit_log()

        plot = RagonePlot(aged_solutions, fit=True)
        fig, ax = plot.plot()
        assert isinstance(fig, plt.Figure)
        # Each solution contributes a data line and a fit line
        assert len(ax.lines) >= 2 * len(aged_solutions)


# ---------------------------------------------------------------------------
# Group 3: RagoneSimulation -> RagoneSolution -> RagonePlot
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSimulationPipeline:
    """Full pipeline: RagoneSimulation.solve() -> RagoneSolution -> RagonePlot."""

    @pytest.fixture
    def patched_sim_power(self, mock_model):
        """RagoneSimulation in power mode with pybamm.Simulation mocked."""
        call_count = [0]

        def _solve_side_effect(**kwargs):
            call_count[0] += 1
            # First call is the theoretical value; subsequent calls are the sweep
            times = [10.0, 5.0, 2.0, 0.8]
            idx = min(call_count[0] - 1, len(times) - 1)
            return _mock_pybamm_sol(time_final=times[idx])

        mock_pybamm_sim = MagicMock()
        mock_pybamm_sim.solve.side_effect = _solve_side_effect

        sim = RagoneSimulation(
            mock_model,
            value_range=np.array([1.0, 5.0, 10.0]),
            parameter_values={
                "Nominal cell capacity [A.h]": 5.0,
                "Upper voltage cut-off [V]": 4.2,
                "Lower voltage cut-off [V]": 2.5,
            },
        )
        sim.step_fun = MagicMock(return_value=MagicMock())

        with (
            patch("ragone.simulation.pybamm.Simulation", return_value=mock_pybamm_sim),
            patch("ragone.simulation.pybamm.Experiment"),
        ):
            solution = sim.solve()

        return sim, solution

    @pytest.fixture
    def patched_sim_current_watts(self, mock_model):
        """RagoneSimulation in current+watts mode with pybamm.Simulation mocked."""
        call_count = [0]

        def _solve_side_effect(**kwargs):
            call_count[0] += 1
            times = [10.0, 5.0, 2.0, 0.8]
            idx = min(call_count[0] - 1, len(times) - 1)
            return _mock_pybamm_sol(
                time_final=times[idx], energy_final=times[idx] * 3.35
            )

        mock_pybamm_sim = MagicMock()
        mock_pybamm_sim.solve.side_effect = _solve_side_effect

        sim = RagoneSimulation(
            mock_model,
            value_range=np.array([1.0, 2.0, 5.0]),
            mode="current",
            convert_to_watts=True,
            parameter_values={
                "Nominal cell capacity [A.h]": 5.0,
                "Upper voltage cut-off [V]": 4.2,
                "Lower voltage cut-off [V]": 2.5,
            },
        )
        sim.step_fun = MagicMock(return_value=MagicMock())

        with (
            patch("ragone.simulation.pybamm.Simulation", return_value=mock_pybamm_sim),
            patch("ragone.simulation.pybamm.Experiment"),
        ):
            solution = sim.solve()

        return sim, solution

    def test_solve_returns_ragone_solution_instance(self, patched_sim_power):
        _, solution = patched_sim_power
        assert isinstance(solution, RagoneSolution)

    def test_solve_stores_solution_on_simulation(self, patched_sim_power):
        sim, solution = patched_sim_power
        assert sim.solution is solution

    def test_solution_mode_is_power(self, patched_sim_power):
        _, solution = patched_sim_power
        assert solution.mode == "power"

    def test_solution_data_has_required_keys(self, patched_sim_power):
        _, solution = patched_sim_power
        assert "Power [W]" in solution.data
        assert "Energy [W.h]" in solution.data
        assert "Time [h]" in solution.data

    def test_solution_inputs_match_value_range(self, patched_sim_power):
        _, solution = patched_sim_power
        assert np.array_equal(solution.data["Power [W]"], np.array([1.0, 5.0, 10.0]))

    def test_solution_outputs_are_input_times_time(self, patched_sim_power):
        _, solution = patched_sim_power
        expected = solution.data["Power [W]"] * solution.data["Time [h]"]
        np.testing.assert_allclose(solution.data["Energy [W.h]"], expected)

    def test_solution_can_be_plotted(self, patched_sim_power):
        _, solution = patched_sim_power
        fig, ax = solution.plot()
        assert isinstance(fig, plt.Figure)

    def test_full_pipeline_solve_fit_plot(self, patched_sim_power):
        _, solution = patched_sim_power
        solution.fit_log()
        assert solution.metrics is not None
        fig, ax = solution.plot()
        assert isinstance(fig, plt.Figure)

    def test_current_watts_solution_has_watt_keys(self, patched_sim_current_watts):
        _, solution = patched_sim_current_watts
        assert "Power [W]" in solution.data
        assert "Energy [W.h]" in solution.data

    def test_current_watts_solution_can_be_plotted(self, patched_sim_current_watts):
        _, solution = patched_sim_current_watts
        # current solution with watt keys can be plotted alongside a power solution
        fig, ax = solution.plot()
        assert isinstance(fig, plt.Figure)

    def test_solver_failure_produces_nan_that_survives_to_plot(self, mock_model):
        """A NaN from a solver failure should not crash RagonePlot."""
        mock_pybamm_sim = MagicMock()
        import pybamm

        call_count = [0]

        def _solve_side_effect(**kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return _mock_pybamm_sol(time_final=10.0)
            if call_count[0] == 2:
                raise pybamm.SolverError("mock failure")
            return _mock_pybamm_sol(time_final=2.0)

        mock_pybamm_sim.solve.side_effect = _solve_side_effect

        sim = RagoneSimulation(
            mock_model,
            value_range=np.array([1.0, 5.0, 10.0]),
            parameter_values={
                "Nominal cell capacity [A.h]": 5.0,
                "Upper voltage cut-off [V]": 4.2,
                "Lower voltage cut-off [V]": 2.5,
            },
        )
        sim.step_fun = MagicMock(return_value=MagicMock())

        with (
            patch("ragone.simulation.pybamm.Simulation", return_value=mock_pybamm_sim),
            patch("ragone.simulation.pybamm.Experiment"),
        ):
            solution = sim.solve()

        assert np.any(np.isnan(solution.data["Power [W]"]))
        # Plot must not raise even with NaN entries
        fig, ax = solution.plot()
        assert isinstance(fig, plt.Figure)


# ---------------------------------------------------------------------------
# Group 4: config helpers <-> RagoneSimulation
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestConfigIntegration:
    """Config helpers produce outputs that integrate with RagoneSimulation."""

    def test_get_options_no_degradation_returns_empty_dict_and_empty_tag(self):
        from ragone.config import get_options

        options, tag = get_options()
        assert options == {}
        assert tag == ""

    def test_get_options_sei_only(self):
        from ragone.config import get_options

        options, tag = get_options(SEI=True)
        assert "SEI" in options
        assert "SEI porosity change" in options
        assert tag == "_SEI"

    def test_get_options_plating_only(self):
        from ragone.config import get_options

        options, tag = get_options(plating=True)
        assert "lithium plating" in options
        assert "lithium plating porosity change" in options
        assert tag == "_plating"

    def test_get_options_lam_only(self):
        from ragone.config import get_options

        options, tag = get_options(lam=True)
        assert "particle mechanics" in options
        assert "loss of active material" in options
        assert tag == "_lam"

    def test_get_options_all_mechanisms_tag_order(self):
        from ragone.config import get_options

        _, tag = get_options(SEI=True, plating=True, lam=True)
        assert tag == "_SEI_plating_lam"

    def test_get_options_sei_lam_without_plating(self):
        from ragone.config import get_options

        options, tag = get_options(SEI=True, lam=True)
        assert "SEI" in options
        assert "loss of active material" in options
        assert "lithium plating" not in options
        assert tag == "_SEI_lam"

    def test_get_options_returns_dict_usable_by_ragone_simulation(self, mock_model):
        """Options dict from get_options() can be passed to RagoneSimulation."""
        from ragone.config import get_options

        options, _ = get_options()
        # options is normally used for pybamm model construction, but the key
        # thing is that RagoneSimulation.__init__ does not consume it directly.
        # We verify the whole init path still works with realistic param values.
        sim = RagoneSimulation(
            mock_model,
            value_range=np.array([1.0, 5.0]),
            parameter_values={
                "Nominal cell capacity [A.h]": 5.0,
                "Upper voltage cut-off [V]": 4.2,
                "Lower voltage cut-off [V]": 2.5,
            },
        )
        assert sim.mode == "power"
        assert isinstance(options, dict)
