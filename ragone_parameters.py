import pybamm
import numpy as np
from ragone import (
    RagoneSimulation,
    RagonePlot,
    get_parameter_values,
    get_options,
    get_var_pts,
)
from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 14})

options, tag = get_options(SEI=True, plating=True, lam=True)

model = pybamm.lithium_ion.DFN(
    options=options,
)

parameter_values = get_parameter_values(ageing=False)

volume = parameter_values["Cell volume [m3]"] * 1000

aged_sol = pybamm.load(Path("data") / f"aged_solution{tag}.pkl")

var_pts = get_var_pts()

step = 100

cycles = [0] + [
    i * step - 1 for i in range(1, len(aged_sol.all_first_states) // step + 1)
]

labels = [f"Cycle {i + 1}" for i in cycles]
labels[1:-1] = [None] * (len(labels) - 2)

filename_extension = {
    "Negative electrode porosity": "eps_n",
    "Negative electrode active material volume fraction": "amvf_n",
    "Positive electrode porosity": "eps_p",
    "Positive electrode active material volume fraction": "amvf_p",
}

parameter_sweeps = {}

print("Extracting parameter values from aged solution...")
for label in filename_extension.keys():
    values = []
    for i in cycles:
        values.append(
            aged_sol.all_first_states[i][f"X-averaged {label.lower()}"].entries[0]
        )
    parameter_sweeps[label] = values

# Make plots of parameter evolution
fig, ax = plt.subplots()
for label, values in parameter_sweeps.items():
    ax.plot(cycles, values, label=label)

ax.set_xlabel("Cycle number")
ax.set_ylim(0, 1)
ax.legend(
    ["Neg. porosity", "Neg. AMVF", "Pos. porosity", "Pos. AMVF"], loc="upper right"
)
fig.savefig(Path("figures") / "aged_solution_evolution_vf.png", dpi=300)

solver = pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-10)

value_ranges = {
    "power": np.logspace(np.log10(0.5), np.log10(100), 50),
    "current": np.logspace(np.log10(0.1), np.log10(30), 50),
}

for mode, value_range in value_ranges.items():
    print(f"Starting Ragone plots - {mode} mode")
    for parameter_name, parameter_range in parameter_sweeps.items():
        solutions = []
        edited_parameter_values = get_parameter_values(ageing=False)

        print(f"Running Ragone plot for parameter: {parameter_name}")
        for i, parameter_value in enumerate(parameter_range):
            print(f"Running Ragone plot for solution {i + 1} of {len(parameter_range)}")
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

        plt = RagonePlot(solutions, labels=labels, volume=volume, scale="loglog")
        fig, _ = plt.plot(show_plot=False)
        fig.savefig(
            Path("figures")
            / f"ragone_parameters_{mode}_{filename_extension[parameter_name]}_loglog.png",
            dpi=300,
        )

# Now rerun for linear scale
value_ranges = {
    "power": np.linspace(0.5, 100, 50),
    "current": np.linspace(0.1, 30, 50),
}

for mode, value_range in value_ranges.items():
    print(f"Starting Ragone plots - {mode} mode")
    for parameter_name, parameter_range in parameter_sweeps.items():
        solutions = []
        edited_parameter_values = get_parameter_values(ageing=False)

        print(f"Running Ragone plot for parameter: {parameter_name}")
        for i, parameter_value in enumerate(parameter_range):
            print(f"Running Ragone plot for solution {i + 1} of {len(parameter_range)}")
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

        plt = RagonePlot(solutions, labels=labels, volume=volume, scale="linear")
        fig, _ = plt.plot(show_plot=False)
        fig.savefig(
            Path("figures")
            / f"ragone_parameters_{mode}_{filename_extension[parameter_name]}_linear.png",
            dpi=300,
        )
