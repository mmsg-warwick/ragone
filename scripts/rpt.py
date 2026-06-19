import pybamm
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
from ragone import (
    RagoneSimulation,
    ROOT,
    get_options,
    get_parameter_values,
    get_var_pts,
)
import argparse

plt.rcParams.update({"font.size": 14})

parser = argparse.ArgumentParser(description="Run battery rate capability plot.")
parser = argparse.ArgumentParser(description="Run battery ageing simulation.")
parser.add_argument("--SEI", action="store_true", help="Enable SEI (default: disabled)")
parser.add_argument(
    "--plating", action="store_true", help="Enable plating (default: disabled)"
)
parser.add_argument("--lam", action="store_true", help="Enable LAM (default: disabled)")
parser.add_argument(
    "--fast", action="store_true", help="Fast charging (default: disabled)"
)
parser.set_defaults(SEI=False, plating=False, lam=False, fast=False)
args = parser.parse_args()

options, tag = get_options(SEI=args.SEI, plating=args.plating, lam=args.lam)

if args.fast:
    tag = "_fast" + tag

model = pybamm.lithium_ion.DFN(
    options=options,
)

parameter_values = get_parameter_values(ageing=False)

aged_sol = pybamm.load(ROOT / "data" / f"aged_solution{tag}.pkl")

var_pts = get_var_pts()

solver = pybamm.IDAKLUSolver(
    rtol=1e-6,
    atol=1e-8,
    options={
        "max_error_test_failures": 200,
        "max_convergence_failures": 20000,
        "max_nonlinear_iterations": 400,
        # "dt_min": 1e-9,
    },
)

step = 25
cycles = [1] + list(range(step - 1, len(aged_sol.all_first_states), step))

ageing_solutions = [aged_sol.all_first_states[i] for i in cycles]

value_ranges = {
    # "power": [2, 10, 18, 26, 34, 42, 50],
    "power": [2, 6, 10, 14, 18],
    # "current": [0.5, 1.25, 2.5, 3.75, 5],
}

for mode, value_range in value_ranges.items():
    cmap = colormaps["plasma"]
    colors = cmap(np.linspace(0, 0.9, len(value_range)))
    fig, ax = plt.subplots(constrained_layout=True)
    data = {}
    for value, color in zip(value_range, colors):
        print(f"Running Ragone plots for {mode} {value}...")
        solutions = []
        for i, first_state in enumerate(ageing_solutions):
            print(
                f"Running rate capability plot for solution {i + 1} of {len(ageing_solutions)}"
            )
            new_model = model.set_initial_conditions_from(first_state, inplace=False)
            sim = RagoneSimulation(
                new_model,
                parameter_values=parameter_values,
                value_range=[value],
                solver=solver,
                # solver=pybamm.IDAKLUSolver(),
                mode=mode,
                var_pts=var_pts,
            )

            sol = sim.solve()

            solutions.append(sol.data[sol.output][0])

        units = "W" if mode == "power" else "A"
        data[value] = solutions
        ax.plot(cycles, solutions, label=f"{value} {units}", color=color)

    ax.set_xlabel("Cycle number")
    ax.set_ylabel(sol.output)
    ymax = 20 if mode == "power" else 5
    ax.set_ylim(0, ymax * 1.1)
    ax.legend(fontsize=10)
    fig.savefig(ROOT / "figures" / f"rpt_{mode}{tag}.png", dpi=300)
    print("Saved figure for mode:", mode)
