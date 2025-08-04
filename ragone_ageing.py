import pybamm
import numpy as np
# from util import compute_ragone, plot_ragone
from ragone import RagoneSimulation, RagonePlot, get_options
from os import path
import argparse

parser = argparse.ArgumentParser(description="Run battery ageing simulation.")
parser.add_argument("--SEI", action="store_true", help="Enable SEI (default: disabled)")
parser.add_argument("--plating", action="store_true", help="Enable plating (default: disabled)")
parser.add_argument("--lam", action="store_true", help="Enable LAM (default: disabled)")
parser.add_argument("--step", type=int, default=100, help="Step size for ageing (default: 100)")
parser.set_defaults(SEI=False, plating=False, lam=False)
args = parser.parse_args()

step = args.step

options, tag = get_options(SEI=args.SEI, plating=args.plating, lam=args.lam)

model = pybamm.lithium_ion.DFN(
    options=options,
    # options={
    #     # "SEI": "ec reaction limited",
    #     "SEI": "reaction limited",
    #     # "SEI": "solvent-diffusion limited",
    #     # "SEI": "constant",
    #     "SEI porosity change": "true",
    # }
)

parameter_values = pybamm.ParameterValues("OKane2022")
Chen2020 = pybamm.ParameterValues("Chen2020")
parameter_values["Negative electrode OCP [V]"] = Chen2020["Negative electrode OCP [V]"]
parameter_values["SEI kinetic rate constant [m.s-1]"] = 0
parameter_values["SEI reaction exchange current density [A.m-2]"] = 0
parameter_values["SEI solvent diffusivity [m2.s-1]"] = 0
parameter_values["Lithium plating kinetic rate constant [m.s-1]"] = 0
parameter_values["Negative electrode LAM constant proportional term [s-1]"] = 0
parameter_values["Positive electrode LAM constant proportional term [s-1]"] = 0


volume = parameter_values["Cell volume [m3]"] * 1000

aged_sol = pybamm.load(path.join("data", f"aged_solution{tag}.pkl"))

# var_pts = {
#     "x_n": 20,
#     "x_s": 20,
#     "x_p": 20,
#     "r_n": 20,
#     "r_p": 20,
# }

var_pts = {
    "x_n": 30,
    "x_s": 30,
    "x_p": 30,
    "r_n": 30,
    "r_p": 30,
}

# solutions = [sol.all_first_states[0], sol.all_first_states[-1]]
ageing_solutions = [aged_sol.all_first_states[0]] + aged_sol.all_first_states[step-1::step]

modes = [
    "power",
    "current",
]
value_ranges = [
    np.logspace(np.log10(0.5), np.log10(100), 50),
    np.logspace(np.log10(0.1), np.log10(30), 50),
]

labels = [f"Cycle {step * i}" for i in range(len(ageing_solutions))]
labels[1:-1] = [None] * (len(labels) - 2)

for mode, value_range in zip(modes, value_ranges):
    solutions = []
    for i, first_state in enumerate(ageing_solutions):
        print(f"Running Ragone plot for solution {i+1} of {len(ageing_solutions)}")
        new_model = model.set_initial_conditions_from(first_state, inplace=False)
        sim = RagoneSimulation(
            new_model,
            parameter_values=parameter_values,
            value_range=value_range,
            solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-10),
            mode=mode,
            var_pts=var_pts,
        )

        sol = sim.solve()

        # fig, _ = sol.plot_fit(show_plot=False)
        # fig.savefig("./figures/" + f"ragone_ageing_fit_log_cycle{step * i}.png", dpi=300)

        # fig, _ = sol.plot_gaussian(show_plot=False)
        # fig.savefig("./figures/" + f"ragone_ageing_fit_gaussian_cycle{step * i}.png", dpi=300)
        
        solutions.append(sol)
    
    plts = RagonePlot(solutions, labels=labels, volume=volume)
    fig, _ = plts.plot(show_plot=False)
    fig.savefig("./figures/" + f"ragone_ageing{tag}_{mode}.png", dpi=300)