"""Unit tests for RagoneSolution."""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from ragone.solution import RagoneSolution


# ---------------------------------------------------------------------------
# Synthetic data for fit_log
# ---------------------------------------------------------------------------
# Exact match to the gaussian_log model with E0=1, P0=1, n=1.
# Starting from the default p0=[1,1,1], curve_fit converges immediately.
_P_FIT = np.array([0.1, 0.3, 0.5, 0.7, 1.0])
_LOG_E_FIT = (1.0 - _P_FIT) / np.log(10)  # gaussian_log(log10(P), 1, 1, 1)
_E_FIT = 10**_LOG_E_FIT


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


class TestRagoneSolutionInit:
    @pytest.mark.unit
    def test_power_mode_sets_input_label(self, power_solution):
        assert power_solution.input == "Power [W]"

    @pytest.mark.unit
    def test_power_mode_sets_output_label(self, power_solution):
        assert power_solution.output == "Energy [W.h]"

    @pytest.mark.unit
    def test_current_mode_sets_input_label(self, current_solution):
        assert current_solution.input == "Current [A]"

    @pytest.mark.unit
    def test_current_mode_sets_output_label(self, current_solution):
        assert current_solution.output == "Capacity [A.h]"

    @pytest.mark.unit
    def test_data_is_stored(self, power_data, power_solution):
        assert power_solution.data is power_data

    @pytest.mark.unit
    def test_mode_is_stored(self, power_solution):
        assert power_solution.mode == "power"

    @pytest.mark.unit
    def test_raw_metrics_initialised_to_none(self, power_solution):
        assert power_solution._raw_metrics is None

    @pytest.mark.unit
    def test_metrics_initialised_to_none(self, power_solution):
        assert power_solution._metrics is None


# ---------------------------------------------------------------------------
# Private math functions
# ---------------------------------------------------------------------------


class TestRagoneSolutionMathFunctions:
    @pytest.mark.unit
    def test_gaussian_log_known_value(self, power_solution):
        # x=0, E0=2, P0=1, n=1 → (2 - (10^0 / 1)^1) / ln(10) = 1/ln(10)
        result = power_solution._gaussian_log(0.0, 2.0, 1.0, 1.0)
        assert result == pytest.approx(1.0 / np.log(10))

    @pytest.mark.unit
    def test_gaussian_known_value(self, power_solution):
        # x=2, E0=10, P0=2, n=1 → 10 * exp(-((2/2)^1)) = 10/e
        result = power_solution._gaussian(2.0, 10.0, 2.0, 1.0)
        assert result == pytest.approx(10.0 / np.e)

    @pytest.mark.unit
    def test_gaussian_loglog_identical_to_gaussian_log(self, power_solution):
        x = np.array([0.0, 0.5, 1.0])
        args = (2.0, 1.5, 2.0)
        np.testing.assert_allclose(
            power_solution._gaussian_loglog(x, *args),
            power_solution._gaussian_log(x, *args),
        )

    @pytest.mark.unit
    def test_gaussian_linear_identical_to_gaussian(self, power_solution):
        x = np.array([1.0, 2.0, 5.0])
        args = (10.0, 3.0, 1.5)
        np.testing.assert_allclose(
            power_solution._gaussian_linear(x, *args),
            power_solution._gaussian(x, *args),
        )


# ---------------------------------------------------------------------------
# fit_log
# ---------------------------------------------------------------------------


class TestRagoneSolutionFitLog:
    @pytest.fixture
    def fittable_solution(self):
        data = {
            "Power [W]": _P_FIT,
            "Energy [W.h]": _E_FIT,
            "Time [h]": _E_FIT / _P_FIT,
        }
        return RagoneSolution(data, "power")

    @pytest.mark.unit
    def test_returns_array_of_three_params(self, fittable_solution):
        popt = fittable_solution.fit_log()
        assert len(popt) == 3

    @pytest.mark.unit
    def test_sets_raw_metrics(self, fittable_solution):
        popt = fittable_solution.fit_log()
        np.testing.assert_array_equal(fittable_solution._raw_metrics, popt)

    @pytest.mark.unit
    def test_metrics_contains_n_key(self, fittable_solution):
        fittable_solution.fit_log()
        assert "n" in fittable_solution.metrics

    @pytest.mark.unit
    def test_metrics_contains_energy_reference_key(self, fittable_solution):
        fittable_solution.fit_log()
        keys = list(fittable_solution.metrics)
        assert any("energy" in k.lower() for k in keys)

    @pytest.mark.unit
    def test_metrics_contains_power_reference_key(self, fittable_solution):
        fittable_solution.fit_log()
        keys = list(fittable_solution.metrics)
        assert any("power" in k.lower() for k in keys)

    @pytest.mark.unit
    def test_metrics_n_matches_popt(self, fittable_solution):
        popt = fittable_solution.fit_log()
        assert fittable_solution.metrics["n"] == pytest.approx(popt[2])

    @pytest.mark.unit
    def test_recovers_known_parameters(self, fittable_solution):
        # Data was generated with E0=1, P0=1, n=1 — expect recovery
        popt = fittable_solution.fit_log()
        assert popt[0] == pytest.approx(1.0, rel=1e-3)
        assert popt[1] == pytest.approx(1.0, rel=1e-3)
        assert popt[2] == pytest.approx(1.0, rel=1e-3)


# ---------------------------------------------------------------------------
# plot() delegation
# ---------------------------------------------------------------------------


class TestRagoneSolutionPlot:
    @pytest.mark.unit
    def test_plot_delegates_to_ragone_plot(self, power_solution):
        mock_fig, mock_ax = MagicMock(), MagicMock()
        with patch("ragone.plotting.RagonePlot") as MockPlot:
            MockPlot.return_value.plot.return_value = (mock_fig, mock_ax)
            result = power_solution.plot()
        assert result == (mock_fig, mock_ax)

    @pytest.mark.unit
    def test_plot_passes_kwargs_to_ragone_plot(self, power_solution):
        with patch("ragone.plotting.RagonePlot") as MockPlot:
            MockPlot.return_value.plot.return_value = (MagicMock(), MagicMock())
            power_solution.plot(labels=["A"], volume=0.1, mass=0.2, scale="linear")
        MockPlot.assert_called_once_with(
            power_solution, labels=["A"], volume=0.1, mass=0.2, scale="linear"
        )


# ---------------------------------------------------------------------------
# fit()  (requires setting solution.scale manually — not set by __init__)
# ---------------------------------------------------------------------------

# Synthetic data for the linear fit: _gaussian_linear(x, E0=10, P0=5, n=1)
_P_LINEAR = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
_E_LINEAR = 10.0 * np.exp(-((_P_LINEAR / 5.0) ** 1.0))


class TestRagoneSolutionFit:
    @pytest.fixture
    def loglog_solution(self):
        data = {
            "Power [W]": _P_FIT,
            "Energy [W.h]": _E_FIT,
            "Time [h]": _E_FIT / _P_FIT,
        }
        sol = RagoneSolution(data, "power")
        sol.scale = "loglog"
        return sol

    @pytest.fixture
    def linear_solution(self):
        data = {
            "Power [W]": _P_LINEAR,
            "Energy [W.h]": _E_LINEAR,
            "Time [h]": _E_LINEAR / _P_LINEAR,
        }
        sol = RagoneSolution(data, "power")
        sol.scale = "linear"
        return sol

    @pytest.mark.unit
    def test_fit_loglog_returns_three_params(self, loglog_solution):
        popt = loglog_solution.fit()
        assert len(popt) == 3

    @pytest.mark.unit
    def test_fit_loglog_sets_raw_metrics(self, loglog_solution):
        popt = loglog_solution.fit()
        np.testing.assert_array_equal(loglog_solution._raw_metrics, popt)

    @pytest.mark.unit
    def test_fit_loglog_metrics_has_fitting_scale_key(self, loglog_solution):
        loglog_solution.fit()
        assert loglog_solution.metrics["Fitting scale"] == "loglog"

    @pytest.mark.unit
    def test_fit_loglog_metrics_has_n_and_reference_keys(self, loglog_solution):
        loglog_solution.fit()
        assert "n" in loglog_solution.metrics
        assert "Reference energy [W.h]" in loglog_solution.metrics
        assert "Reference power [W]" in loglog_solution.metrics

    @pytest.mark.unit
    def test_fit_loglog_recovers_known_parameters(self, loglog_solution):
        # Data generated with E0=1, P0=1, n=1
        popt = loglog_solution.fit()
        assert popt[0] == pytest.approx(1.0, rel=1e-3)
        assert popt[1] == pytest.approx(1.0, rel=1e-3)
        assert popt[2] == pytest.approx(1.0, rel=1e-3)

    @pytest.mark.unit
    def test_fit_linear_returns_three_params(self, linear_solution):
        popt = linear_solution.fit()
        assert len(popt) == 3

    @pytest.mark.unit
    def test_fit_linear_sets_raw_metrics(self, linear_solution):
        popt = linear_solution.fit()
        np.testing.assert_array_equal(linear_solution._raw_metrics, popt)

    @pytest.mark.unit
    def test_fit_linear_metrics_has_fitting_scale_key(self, linear_solution):
        linear_solution.fit()
        assert linear_solution.metrics["Fitting scale"] == "linear"

    @pytest.mark.unit
    def test_fit_linear_recovers_known_parameters(self, linear_solution):
        # Data generated with E0=10, P0=5, n=1.
        # _gaussian_linear returns E0 * exp(-((x/P0)^n)), so popt[0] == E0 directly.
        popt = linear_solution.fit()
        assert popt[0] == pytest.approx(10.0, rel=1e-3)
        assert popt[1] == pytest.approx(5.0, rel=1e-3)
        assert popt[2] == pytest.approx(1.0, rel=1e-3)
