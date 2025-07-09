import pybamm
import numpy as np
import gc
from util import compute_ragone, plot_ragone
from ragone import RagoneSimulation, RagonePlot

model = pybamm.lithium_ion.DFN(
    options={
        "SEI": "none",
        # "surface form": "differential",
    }
)

def j_p0(c_e, c_s_surf, c_s_max, T):
    m_ref = pybamm.Parameter("Positive electrode reaction rate")  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
    E_r = 17800
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5

def j_n0(c_e, c_s_surf, c_s_max, T):
    m_ref = pybamm.Parameter("Negative electrode reaction rate")  # (A/m2)(m3/mol)**1.5 - includes ref concentrations
    E_r = 35000
    arrhenius = np.exp(E_r / pybamm.constants.R * (1 / 298.15 - 1 / T))

    return m_ref * arrhenius * c_e**0.5 * c_s_surf**0.5 * (c_s_max - c_s_surf) ** 0.5

parameter_values = pybamm.ParameterValues("OKane2022")
Chen2020 = pybamm.ParameterValues("Chen2020")
parameter_values["Negative electrode OCP [V]"] = Chen2020["Negative electrode OCP [V]"]
parameter_values["Electrolyte diffusivity [m2.s-1]"] = 1.7694e-10
parameter_values["Electrolyte conductivity [S.m-1]"] = 0.9487
parameter_values["Negative electrode exchange-current density [A.m-2]"] = j_n0
parameter_values["Positive electrode exchange-current density [A.m-2]"] = j_p0
parameter_values.update(
    {
        "Negative electrode reaction rate": 6.48e-7,
        "Positive electrode reaction rate": 3.42e-6,
    },
    check_already_exists=False,
)
parameter_values["Negative electrode exchange-current density [A.m-2]"] = j_n0

volume = parameter_values["Cell volume [m3]"] * 1000

var_pts = {
    "x_n": 50,
    "x_s": 20,
    "x_p": 50,
    "r_n": 20,
    "r_p": 20,
}

parameter_sweeps = {
    # "Negative electrode porosity": np.linspace(0.25, 0.05, 5),
    # "Negative electrode active material volume fraction": np.linspace(0.75, 0.35, 5),
    # "Positive electrode porosity": np.linspace(0.335, 0.135, 5),
    # "Positive electrode active material volume fraction": np.linspace(0.665, 0.265, 5),
    "Negative electrode reaction rate": np.logspace(np.log10(6.48e-7), np.log10(6.48e-9), 5),
    # "Positive electrode reaction rate": np.logspace(np.log10(3.42e-6), np.log10(3.42e-8), 5),
}

filename_extension = {
    "Negative electrode porosity": "eps_n",
    "Negative electrode active material volume fraction": "amvf_n",
    "Positive electrode porosity": "eps_p",
    "Positive electrode active material volume fraction": "amvf_p",
    "Negative electrode reaction rate": "j0_n",
    "Positive electrode reaction rate": "j0_p",
}

value_ranges = {
    "power": np.logspace(np.log10(0.5), np.log10(100), 50),
    "current": np.logspace(np.log10(0.1), np.log10(30), 50),
}

solver = pybamm.IDAKLUSolver(
    output_variables=["Voltage [V]"],
    rtol=1e-6,
    atol=1e-9,
    root_tol=1e-9,
    # root_method="lm",
    # options={
    #     "max_error_test_failures": 200,
    #     "max_convergence_failures": 10000,
    #     "max_nonlinear_iterations": 400,
    #     "dt_min": 1e-9,
    # },
)

for mode, value_range in value_ranges.items():
    for parameter_name, parameter_range in parameter_sweeps.items():
        solutions = []
        edited_parameter_values = parameter_values.copy()
        if parameter_range[0] > 0.01:
            labels = [f"{parameter_value:.3f}" for parameter_value in parameter_range]
        else:
            labels = [f"{parameter_value:.3e}" for parameter_value in parameter_range]

        for i, parameter_value in enumerate(parameter_range):
            print(f"Running Ragone plot for solution {i+1} of {len(parameter_range)}")
            edited_parameter_values[parameter_name] = parameter_value
            sim = RagoneSimulation(
                model,
                parameter_values=edited_parameter_values,
                value_range=value_range,
                var_pts=var_pts,
                solver=solver,
                mode=mode,
            )

            sol = sim.solve()

            solutions.append(sol)

            # gc.collect()

        plt = RagonePlot(solutions, labels=labels, volume=volume)
        fig, _ = plt.plot(show_plot=False)
        fig.savefig("./figures/" + f"ragone_{filename_extension[parameter_name]}_{mode}.png", dpi=300)
