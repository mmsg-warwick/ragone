"""Unit tests for ragone/utils.py.

All three public helpers are covered:
  - get_options()          — pure Python, no pybamm dependency
  - get_parameter_values() — pybamm.ParameterValues mocked
  - get_var_pts()          — pure Python, no pybamm dependency
"""

from unittest.mock import MagicMock, patch

import pytest

from ragone.utils import get_options, get_parameter_values, get_var_pts

# ---------------------------------------------------------------------------
# get_options
# ---------------------------------------------------------------------------


class TestGetOptions:
    @pytest.mark.unit
    def test_no_flags_returns_empty_dict_and_empty_tag(self):
        options, tag = get_options()
        assert options == {}
        assert tag == ""

    @pytest.mark.unit
    def test_sei_only_sets_sei_keys(self):
        options, tag = get_options(SEI=True)
        assert "SEI" in options
        assert options["SEI"] == "reaction limited"
        assert "SEI porosity change" in options
        assert options["SEI porosity change"] == "true"

    @pytest.mark.unit
    def test_sei_only_tag(self):
        _, tag = get_options(SEI=True)
        assert tag == "_SEI"

    @pytest.mark.unit
    def test_plating_only_sets_plating_keys(self):
        options, tag = get_options(plating=True)
        assert "lithium plating" in options
        assert options["lithium plating"] == "irreversible"
        assert "lithium plating porosity change" in options
        assert options["lithium plating porosity change"] == "true"

    @pytest.mark.unit
    def test_plating_only_tag(self):
        _, tag = get_options(plating=True)
        assert tag == "_plating"

    @pytest.mark.unit
    def test_lam_only_sets_lam_keys(self):
        options, tag = get_options(lam=True)
        assert "particle mechanics" in options
        assert options["particle mechanics"] == "swelling only"
        assert "loss of active material" in options
        assert options["loss of active material"] == "stress-driven"

    @pytest.mark.unit
    def test_lam_only_tag(self):
        _, tag = get_options(lam=True)
        assert tag == "_lam"

    @pytest.mark.unit
    def test_all_flags_contains_all_keys(self):
        options, _ = get_options(SEI=True, plating=True, lam=True)
        assert "SEI" in options
        assert "lithium plating" in options
        assert "particle mechanics" in options

    @pytest.mark.unit
    def test_all_flags_tag_order(self):
        _, tag = get_options(SEI=True, plating=True, lam=True)
        assert tag == "_SEI_plating_lam"

    @pytest.mark.unit
    def test_sei_lam_tag(self):
        _, tag = get_options(SEI=True, lam=True)
        assert tag == "_SEI_lam"

    @pytest.mark.unit
    def test_plating_lam_tag(self):
        _, tag = get_options(plating=True, lam=True)
        assert tag == "_plating_lam"

    @pytest.mark.unit
    def test_no_flags_does_not_include_sei_keys(self):
        options, _ = get_options()
        assert "SEI" not in options
        assert "lithium plating" not in options
        assert "particle mechanics" not in options

    @pytest.mark.unit
    def test_returns_new_dict_each_call(self):
        options1, _ = get_options(SEI=True)
        options2, _ = get_options(SEI=True)
        assert options1 is not options2


# ---------------------------------------------------------------------------
# get_parameter_values
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pv_instances():
    """Patch pybamm.ParameterValues and return three ordered mock instances."""
    mock_okane = MagicMock(name="OKane2022")
    mock_chen = MagicMock(name="Chen2020")
    mock_oregan = MagicMock(name="ORegan2022")
    with patch(
        "ragone.utils.pybamm.ParameterValues",
        side_effect=[mock_okane, mock_chen, mock_oregan],
    ) as mock_cls:
        yield mock_cls, mock_okane, mock_chen, mock_oregan


@pytest.fixture
def mock_pv_no_ageing():
    """Patch pybamm.ParameterValues for ageing=False calls."""
    mock_okane = MagicMock(name="OKane2022")
    mock_chen = MagicMock(name="Chen2020")
    mock_oregan = MagicMock(name="ORegan2022")
    with patch(
        "ragone.utils.pybamm.ParameterValues",
        side_effect=[mock_okane, mock_chen, mock_oregan],
    ) as mock_cls:
        yield mock_cls, mock_okane, mock_chen, mock_oregan


class TestGetParameterValues:
    @pytest.mark.unit
    def test_creates_three_parameter_sets(self, mock_pv_instances):
        mock_cls, *_ = mock_pv_instances
        get_parameter_values()
        assert mock_cls.call_count == 3

    @pytest.mark.unit
    def test_loads_okane_chen_oregan_by_name(self, mock_pv_instances):
        mock_cls, *_ = mock_pv_instances
        get_parameter_values()
        mock_cls.assert_any_call("OKane2022")
        mock_cls.assert_any_call("Chen2020")
        mock_cls.assert_any_call("ORegan2022")

    @pytest.mark.unit
    def test_returns_okane_instance(self, mock_pv_instances):
        _, mock_okane, mock_chen, mock_oregan = mock_pv_instances
        result = get_parameter_values()
        assert result is mock_okane

    @pytest.mark.unit
    def test_copies_negative_electrode_ocp_from_chen2020(self, mock_pv_instances):
        _, mock_okane, mock_chen, _ = mock_pv_instances
        get_parameter_values()
        expected_value = mock_chen["Negative electrode OCP [V]"]
        mock_okane.__setitem__.assert_any_call(
            "Negative electrode OCP [V]", expected_value
        )

    @pytest.mark.unit
    def test_copies_all_four_transport_params_from_oregan2022(self, mock_pv_instances):
        _, mock_okane, _, mock_oregan = mock_pv_instances
        get_parameter_values()

        transport_params = [
            "Cation transference number",
            "Thermodynamic factor",
            "Electrolyte conductivity [S.m-1]",
            "Electrolyte diffusivity [m2.s-1]",
        ]
        set_keys = [c.args[0] for c in mock_okane.__setitem__.call_args_list]
        for param in transport_params:
            assert param in set_keys

    @pytest.mark.unit
    def test_ageing_true_sets_sei_exchange_current_density(self, mock_pv_instances):
        _, mock_okane, _, _ = mock_pv_instances
        get_parameter_values(ageing=True)
        mock_okane.__setitem__.assert_any_call(
            "SEI reaction exchange current density [A.m-2]", 3.6e-8
        )

    @pytest.mark.unit
    def test_ageing_true_sets_sei_kinetic_rate(self, mock_pv_instances):
        _, mock_okane, _, _ = mock_pv_instances
        get_parameter_values(ageing=True)
        mock_okane.__setitem__.assert_any_call(
            "SEI kinetic rate constant [m.s-1]", 5e-8
        )

    @pytest.mark.unit
    def test_ageing_true_sets_ec_diffusivity(self, mock_pv_instances):
        _, mock_okane, _, _ = mock_pv_instances
        get_parameter_values(ageing=True)
        mock_okane.__setitem__.assert_any_call("EC diffusivity [m2.s-1]", 1e-20)

    @pytest.mark.unit
    def test_ageing_true_sets_plating_kinetic_rate(self, mock_pv_instances):
        _, mock_okane, _, _ = mock_pv_instances
        get_parameter_values(ageing=True)
        mock_okane.__setitem__.assert_any_call(
            "Lithium plating kinetic rate constant [m.s-1]", 4e-12 * 0.8
        )

    @pytest.mark.unit
    def test_ageing_true_sets_plating_exchange_current_density(self, mock_pv_instances):
        _, mock_okane, _, _ = mock_pv_instances
        get_parameter_values(ageing=True)
        mock_okane.__setitem__.assert_any_call(
            "Exchange-current density for plating [A.m-2]", 8e-4
        )

    @pytest.mark.unit
    def test_ageing_true_sets_lam_params(self, mock_pv_instances):
        _, mock_okane, _, _ = mock_pv_instances
        get_parameter_values(ageing=True)
        set_keys = [c.args[0] for c in mock_okane.__setitem__.call_args_list]
        assert "Negative electrode LAM constant proportional term [s-1]" in set_keys
        assert "Positive electrode LAM constant proportional term [s-1]" in set_keys
        assert "Negative electrode LAM constant exponential term" in set_keys
        assert "Positive electrode LAM constant exponential term" in set_keys

    @pytest.mark.unit
    def test_ageing_false_zeros_sei_kinetic_rate(self, mock_pv_no_ageing):
        _, mock_okane, _, _ = mock_pv_no_ageing
        get_parameter_values(ageing=False)
        mock_okane.__setitem__.assert_any_call("SEI kinetic rate constant [m.s-1]", 0)

    @pytest.mark.unit
    def test_ageing_false_zeros_sei_exchange_current_density(self, mock_pv_no_ageing):
        _, mock_okane, _, _ = mock_pv_no_ageing
        get_parameter_values(ageing=False)
        mock_okane.__setitem__.assert_any_call(
            "SEI reaction exchange current density [A.m-2]", 0
        )

    @pytest.mark.unit
    def test_ageing_false_zeros_sei_solvent_diffusivity(self, mock_pv_no_ageing):
        _, mock_okane, _, _ = mock_pv_no_ageing
        get_parameter_values(ageing=False)
        mock_okane.__setitem__.assert_any_call("SEI solvent diffusivity [m2.s-1]", 0)

    @pytest.mark.unit
    def test_ageing_false_zeros_plating_kinetic_rate(self, mock_pv_no_ageing):
        _, mock_okane, _, _ = mock_pv_no_ageing
        get_parameter_values(ageing=False)
        mock_okane.__setitem__.assert_any_call(
            "Lithium plating kinetic rate constant [m.s-1]", 0
        )

    @pytest.mark.unit
    def test_ageing_false_zeros_lam_params(self, mock_pv_no_ageing):
        _, mock_okane, _, _ = mock_pv_no_ageing
        get_parameter_values(ageing=False)
        set_calls = mock_okane.__setitem__.call_args_list
        lam_calls = {c.args[0]: c.args[1] for c in set_calls}
        assert lam_calls["Negative electrode LAM constant proportional term [s-1]"] == 0
        assert lam_calls["Positive electrode LAM constant proportional term [s-1]"] == 0

    @pytest.mark.unit
    def test_ageing_defaults_to_true(self, mock_pv_instances):
        """Calling with no argument should apply ageing parameters."""
        _, mock_okane, _, _ = mock_pv_instances
        get_parameter_values()
        set_keys = [c.args[0] for c in mock_okane.__setitem__.call_args_list]
        # SEI solvent diffusivity is only zeroed in ageing=False path
        assert "SEI solvent diffusivity [m2.s-1]" not in set_keys

    @pytest.mark.unit
    def test_ageing_true_does_not_zero_plating_rate(self, mock_pv_instances):
        """ageing=True path must not set any kinetic rates to 0."""
        _, mock_okane, _, _ = mock_pv_instances
        get_parameter_values(ageing=True)
        zero_calls = [
            c for c in mock_okane.__setitem__.call_args_list if c.args[1] == 0
        ]
        assert zero_calls == []


# ---------------------------------------------------------------------------
# get_var_pts
# ---------------------------------------------------------------------------


class TestGetVarPts:
    @pytest.mark.unit
    def test_returns_a_dict(self):
        assert isinstance(get_var_pts(), dict)

    @pytest.mark.unit
    def test_contains_all_spatial_keys(self):
        var_pts = get_var_pts()
        for key in ("x_n", "x_s", "x_p", "r_n", "r_p"):
            assert key in var_pts

    @pytest.mark.unit
    def test_all_values_are_positive_integers(self):
        for val in get_var_pts().values():
            assert isinstance(val, int)
            assert val > 0

    @pytest.mark.unit
    def test_electrode_x_values(self):
        var_pts = get_var_pts()
        assert var_pts["x_n"] == 100
        assert var_pts["x_s"] == 30
        assert var_pts["x_p"] == 50

    @pytest.mark.unit
    def test_particle_r_values(self):
        var_pts = get_var_pts()
        assert var_pts["r_n"] == 30
        assert var_pts["r_p"] == 30

    @pytest.mark.unit
    def test_returns_new_dict_each_call(self):
        assert get_var_pts() is not get_var_pts()
