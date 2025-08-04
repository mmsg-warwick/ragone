import pybamm
from ragone import get_options
from os import path
import argparse

parser = argparse.ArgumentParser(description="Run battery ageing simulation.")
parser.add_argument("--SEI", action="store_true", help="Enable SEI (default: disabled)")
parser.add_argument("--plating", action="store_true", help="Enable plating (default: disabled)")
parser.add_argument("--lam", action="store_true", help="Enable LAM (default: disabled)")
parser.add_argument("--N_cycles", type=int, default=1000, help="Number of cycles (default: 1000)")
parser.add_argument("--fast", action="store_true", help="Fast charging (default: disabled)")
parser.set_defaults(SEI=False, plating=False, lam=False, fast=False)
args = parser.parse_args()

options, tag = get_options(SEI=args.SEI, plating=args.plating, lam=args.lam)
N_cycles = args.N_cycles
if args.fast:
    tag = "_fast" + tag

pybamm.set_logging_level("NOTICE")

# options, tag = get_options(SEI=False, plating=True, lam=False)

model = pybamm.lithium_ion.DFN(
    options=options,
)

parameter_values = pybamm.ParameterValues("OKane2022")
Chen2020 = pybamm.ParameterValues("Chen2020")
parameter_values["Negative electrode OCP [V]"] = Chen2020["Negative electrode OCP [V]"]
# parameter_values["SEI solvent diffusivity [m2.s-1]"] *= 10
# parameter_values["EC diffusivity [m2.s-1]"] = 2e-18
# parameter_values["SEI kinetic rate constant [m.s-1]"] = 2e-13
parameter_values["SEI reaction exchange current density [A.m-2]"] = 1.5e-7 *  0.15 * 2
# parameter_values["SEI open-circuit potential [V]"] = 0
parameter_values["Lithium plating kinetic rate constant [m.s-1]"] = 5e-12 * 2
parameter_values["Negative electrode LAM constant proportional term [s-1]"] = 2.7778e-07 * 2
parameter_values["Positive electrode LAM constant proportional term [s-1]"] = 2.7778e-07 * 2
# parameter_values["Negative electrode LAM constant proportional term [s-1]"] = 1.82e-6
# parameter_values["Positive electrode LAM constant proportional term [s-1]"] = 2.132e-6
parameter_values["Negative electrode LAM constant exponential term"] = 1.3
parameter_values["Positive electrode LAM constant exponential term"] = 1.3

output_variables = [
    "Voltage [V]",
    "Current [A]",
    "Time [h]",
    "Discharge capacity [A.h]",
    "Power [W]",
    "Negative electrode porosity",
]

var_pts = {
    "x_n": 30,
    "x_s": 30,
    "x_p": 30,
    "r_n": 30,
    "r_p": 30,
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

# N_cycles = 1000

if args.fast:
    charge_step = "Charge at 1C until 4.2 V"
else:
    charge_step = "Charge at C/3 until 4.2 V"

experiment = pybamm.Experiment(
    [
        (
            "Discharge at 1C until 2.5 V",
            "Rest for 1 hour",
            charge_step,
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
# sol.save(path.join("data", f"aged_solution_fast{tag}.pkl"))