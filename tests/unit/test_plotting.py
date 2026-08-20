"""Unit tests for RagonePlot."""

import matplotlib.axes
import matplotlib.figure
import numpy as np
import pytest

from ragone.plotting import RagonePlot
from ragone.solution import RagoneSolution

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def power_plot(power_solution):
    return RagonePlot(power_solution)


@pytest.fixture
def current_solution_with_watts(current_data):
    """current solution that also carries Power/Energy keys (convert_to_watts)."""
    data = dict(current_data)
    data["Power [W]"] = np.array([3.7, 7.4, 18.5, 37.0])
    data["Energy [W.h]"] = np.array([18.5, 17.8, 15.6, 13.0])
    return RagoneSolution(data, "current")


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestRagonePlotInit:
    @pytest.mark.unit
    def test_single_solution_wrapped_in_list(self, power_solution):
        plot = RagonePlot(power_solution)
        assert plot.solutions == [power_solution]

    @pytest.mark.unit
    def test_list_of_solutions_stored_unchanged(self, power_solution):
        plot = RagonePlot([power_solution])
        assert plot.solutions == [power_solution]

    @pytest.mark.unit
    def test_mode_set_from_power_solution(self, power_solution):
        assert RagonePlot(power_solution).mode == "power"

    @pytest.mark.unit
    def test_mode_set_from_current_solution(self, current_solution):
        assert RagonePlot(current_solution).mode == "current"

    @pytest.mark.unit
    def test_input_output_labels_for_power(self, power_solution):
        plot = RagonePlot(power_solution)
        assert plot.input == "Power [W]"
        assert plot.output == "Energy [W.h]"

    @pytest.mark.unit
    def test_input_output_labels_for_current(self, current_solution):
        plot = RagonePlot(current_solution)
        assert plot.input == "Current [A]"
        assert plot.output == "Capacity [A.h]"

    @pytest.mark.unit
    def test_invalid_scale_raises_value_error(self, power_solution):
        with pytest.raises(ValueError, match="scale must be either"):
            RagonePlot(power_solution, scale="semilog")

    @pytest.mark.unit
    def test_mixed_modes_without_watt_keys_raises(
        self, power_solution, current_solution
    ):
        with pytest.raises(ValueError, match="same mode or have"):
            RagonePlot([power_solution, current_solution])

    @pytest.mark.unit
    def test_mixed_modes_with_watt_keys_uses_power_mode(
        self, power_solution, current_solution_with_watts
    ):
        plot = RagonePlot([power_solution, current_solution_with_watts])
        assert plot.mode == "power"
        assert plot.input == "Power [W]"

    @pytest.mark.unit
    def test_labels_stored(self, power_solution):
        plot = RagonePlot(power_solution, labels=["test"])
        assert plot.labels == ["test"]


# ---------------------------------------------------------------------------
# _compute_data_limits
# ---------------------------------------------------------------------------


class TestRagonePlotDataLimits:
    @pytest.mark.unit
    def test_limits_from_single_solution(self, power_solution):
        # power_data: Power [1,5,10,50], Energy [20,18,15,8]
        plot = RagonePlot(power_solution)
        assert plot.min_input == pytest.approx(1.0)
        assert plot.max_input == pytest.approx(50.0)
        assert plot.min_output == pytest.approx(8.0)
        assert plot.max_output == pytest.approx(20.0)

    @pytest.mark.unit
    def test_limits_span_all_solutions(self, power_solution, power_data):
        wider_data = {
            "Power [W]": np.array([0.5, 100.0]),
            "Energy [W.h]": np.array([25.0, 5.0]),
            "Time [h]": np.array([50.0, 0.05]),
        }
        wider_sol = RagoneSolution(wider_data, "power")
        plot = RagonePlot([power_solution, wider_sol])
        assert plot.min_input == pytest.approx(0.5)
        assert plot.max_input == pytest.approx(100.0)
        assert plot.min_output == pytest.approx(5.0)
        assert plot.max_output == pytest.approx(25.0)


# ---------------------------------------------------------------------------
# Helper methods
# ---------------------------------------------------------------------------


class TestRagonePlotHelpers:
    @pytest.mark.unit
    def test_get_ticks_range_one_decade(self, power_plot):
        assert power_plot._get_ticks_range(1.0, 10.0) == [1, 2, 5, 10]

    @pytest.mark.unit
    def test_get_ticks_range_two_decades(self, power_plot):
        assert power_plot._get_ticks_range(1.0, 100.0) == [1, 2, 5, 10, 20, 50, 100]

    @pytest.mark.unit
    def test_format_tick_labels_integers(self, power_plot):
        assert power_plot._format_tick_labels([1, 2, 10]) == ["1", "2", "10"]

    @pytest.mark.unit
    def test_format_tick_labels_mixed(self, power_plot):
        # 1.0 → "1" (int-valued float), 0.5 and 2.5 → str of float
        assert power_plot._format_tick_labels([0.5, 1.0, 2.5]) == ["0.5", "1", "2.5"]


# ---------------------------------------------------------------------------
# plot()
# ---------------------------------------------------------------------------


class TestRagonePlotPlot:
    @pytest.mark.unit
    def test_returns_figure_and_axes(self, power_plot):
        fig, ax = power_plot.plot(show_plot=False)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert isinstance(ax, matplotlib.axes.Axes)

    @pytest.mark.unit
    def test_linear_scale_returns_figure(self, power_solution):
        fig, ax = RagonePlot(power_solution, scale="linear").plot(show_plot=False)
        assert isinstance(fig, matplotlib.figure.Figure)

    @pytest.mark.unit
    def test_labels_produce_a_legend(self, power_solution):
        fig, ax = RagonePlot(power_solution, labels=["Series A"]).plot(show_plot=False)
        assert ax.get_legend() is not None

    @pytest.mark.unit
    def test_no_labels_produces_no_legend(self, power_plot):
        # Default (labels=None) → skip_legend=True → no legend drawn
        _, ax = power_plot.plot(show_plot=False)
        assert ax.get_legend() is None

    @pytest.mark.unit
    def test_linear_scale_with_labels_creates_legend(self, power_solution):
        # Covers the `elif self.scale == "linear"` legend branch (line 315-316)
        fig, ax = RagonePlot(power_solution, scale="linear", labels=["Series A"]).plot(
            show_plot=False
        )
        assert ax.get_legend() is not None

    @pytest.mark.unit
    def test_both_volume_and_mass_produces_secondary_axes(self, power_solution):
        # Covers the `if self.volume and self.mass` branch (lines 298-299)
        fig, ax = RagonePlot(power_solution, volume=0.01, mass=0.05).plot(
            show_plot=False
        )
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(ax.child_axes) >= 2

    @pytest.mark.unit
    def test_current_solution_with_volume_secondary_axes(self, current_solution):
        # Covers convert_labels "Capacity" branch (lines 219-220) and
        # "Current" branch (lines 224-226) inside _set_secondary_axes
        fig, ax = RagonePlot(current_solution, volume=0.001).plot(show_plot=False)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert len(ax.child_axes) >= 2
