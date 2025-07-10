import pybamm
from ragone import get_options
from os import path

pybamm.set_logging_level("NOTICE")

options, tag = get_options(SEI=True, plating=True, lam=True)

model = pybamm.lithium_ion.DFN(
    options=options,
)

parameter_values = pybamm.ParameterValues("OKane2022")
Chen2020 = pybamm.ParameterValues("Chen2020")
parameter_values["Negative electrode OCP [V]"] = Chen2020["Negative electrode OCP [V]"]
# parameter_values["SEI solvent diffusivity [m2.s-1]"] *= 10
# parameter_values["EC diffusivity [m2.s-1]"] = 2e-18
# parameter_values["SEI kinetic rate constant [m.s-1]"] = 2e-13
parameter_values["SEI reaction exchange current density [A.m-2]"] = 1.5e-7 *  0.15
# parameter_values["SEI open-circuit potential [V]"] = 0
parameter_values["Lithium plating kinetic rate constant [m.s-1]"] = 5e-12
parameter_values["Negative electrode LAM constant proportional term [s-1]"] = 2.7778e-07 * 4
parameter_values["Positive electrode LAM constant proportional term [s-1]"] = 2.7778e-07 * 4

output_variables = [
    "Voltage [V]",
    "Current [A]",
    "Time [h]",
    "Discharge capacity [A.h]",
    "Power [W]",
    "Negative electrode porosity",
]

var_pts = {
    "x_n": 40,
    "x_s": 40,
    "x_p": 40,
    "r_n": 40,
    "r_p": 40,
}

solver = pybamm.IDAKLUSolver(
    output_variables=output_variables,
    # rtol=1e-6,
    # atol=1e-8,
    # options={
    #     "max_error_test_failures": 200,
    #     "max_convergence_failures": 10000,
    #     "max_nonlinear_iterations": 400,
    #     "dt_min": 1e-9,
    # },
)

N_cycles = 2000

experiment = pybamm.Experiment(
    [
        (
            "Discharge at 1C until 2.5 V",
            "Rest for 1 hour",
            "Charge at C/3 until 4.2 V",
            "Hold at 4.2 V until 50 mA",
            "Rest for 1 hour",
        ),
    ] * N_cycles,
    termination="50% capacity",
)

# save_at_cycles = [1, 468, 469]
save_at_cycles = [1]

sim = pybamm.Simulation(
    model,
    parameter_values=parameter_values,
    experiment=experiment,
    var_pts=var_pts,
    solver=solver,
)

sol = sim.solve(save_at_cycles=save_at_cycles)

# print(sol["Negative electrode porosity"].entries)

sol.save(path.join("data", f"aged_solution{tag}.pkl"))