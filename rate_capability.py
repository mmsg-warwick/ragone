import pybamm
import numpy as np
import matplotlib.pyplot as plt
from ragone import RagoneSimulation, get_options
from os import path

options, tag = get_options(SEI=True, plating=False, lam=True)

model = pybamm.lithium_ion.DFN(
    options=options,
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

var_pts = {
    "x_n": 20,
    "x_s": 20,
    "x_p": 20,
    "r_n": 20,
    "r_p": 20,
}

# step = 200
# cycles = [1] + list(range(step-1, len(aged_sol.all_first_states), step))

step = 20
cycles = [1] + list(range(step-1, int(len(aged_sol.all_first_states) / 2) + 1, step))

ageing_solutions = [aged_sol.all_first_states[i] for i in cycles]

modes = [
    "power",
    "current",
]
value_ranges = [
    [10, 20, 30, 35, 40, 45, 50],
    [5, 10, 12.5, 15, 17.5, 20]
]

for mode, value_range in zip(modes, value_ranges):
    fig, ax = plt.subplots()
    data = {}
    for value in value_range:
        solutions = []
        for i, first_state in enumerate(ageing_solutions):
            print(f"Running rate capability plot for solution {i+1} of {len(ageing_solutions)}")
            new_model = model.set_initial_conditions_from(first_state, inplace=False)
            sim = RagoneSimulation(
                new_model,
                parameter_values=parameter_values,
                value_range=[value],
                # solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-10),
                solver=pybamm.IDAKLUSolver(),
                mode=mode,
                var_pts=var_pts,
            )

            sol = sim.solve()
            
            solutions.append(sol.data[sol.output][0])
        
        units = "W" if mode == "power" else "A"
        data[value] = solutions
        ax.plot(cycles, solutions, label=f"{value} {units}")

    ax.set_xlabel("Cycle number")
    ax.set_ylabel(sol.output)
    ax.legend()    
    fig.savefig("./figures/" + f"rate_capability_ageing_{mode}.png", dpi=300)
    print("Saved figure for mode:", mode)
