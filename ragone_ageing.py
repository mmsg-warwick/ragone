import pybamm
import numpy as np
import pandas as pd
from ragone import (
    RagoneSimulation,
    RagonePlot,
    get_options,
    get_parameter_values,
    get_var_pts,
)
from pathlib import Path
import argparse

print("Processing command line arguments...")
parser = argparse.ArgumentParser(description="Run battery ageing simulation.")
parser.add_argument("--SEI", action="store_true", help="Enable SEI (default: disabled)")
parser.add_argument(
    "--plating", action="store_true", help="Enable plating (default: disabled)"
)
parser.add_argument("--lam", action="store_true", help="Enable LAM (default: disabled)")
parser.add_argument(
    "--step", type=int, default=100, help="Step size for ageing (default: 100)"
)
parser.add_argument(
    "--fast", action="store_true", help="Fast charging (default: disabled)"
)
parser.add_argument(
    "--linear",
    action="store_true",
    help="Use linear scale for Ragone plot (default: log scale)",
)

parser.set_defaults(SEI=False, plating=False, lam=False, fast=False, linear=False)
args = parser.parse_args()

options, tag = get_options(SEI=args.SEI, plating=args.plating, lam=args.lam)
step = args.step
if args.fast:
    tag = "_fast" + tag

scale = "linear" if args.linear else "loglog"

print("Simulation tag: ", tag + f"_{scale}")

print("Setting up model and parameters...")
model = pybamm.lithium_ion.DFN(
    options=options,
)
parameter_values = get_parameter_values(ageing=False)
volume = parameter_values["Cell volume [m3]"] * 1000
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

print("Loading aged solution...")
aged_sol = pybamm.load(Path("data") / f"aged_solution{tag}.pkl")
print("Aged solution loaded.")

var_pts = get_var_pts()

cycles = [0] + [
    i * step - 1 for i in range(1, len(aged_sol.all_first_states) // step + 1)
]

labels = [f"Cycle {i + 1}" for i in cycles]
labels[1:-1] = [None] * (len(labels) - 2)

print("Extracting relevant cycles")
# solutions = [sol.all_first_states[0], sol.all_first_states[-1]]
ageing_solutions = [aged_sol.all_first_states[0]] + aged_sol.all_first_states[
    step - 1 :: step
]

if scale == "loglog":
    value_ranges = {
        "power": np.logspace(np.log10(0.5), np.log10(100), 50),
        "current": np.logspace(np.log10(0.1), np.log10(30), 50),
    }
elif scale == "linear":
    value_ranges = {
        "power": np.linspace(0.5, 100, 50),
        "current": np.linspace(0.1, 30, 50),
    }

for mode, value_range in value_ranges.items():
    print(f"Running Ragone plots for {mode}...")
    solutions = []
    for i, first_state in enumerate(ageing_solutions):
        print(f"Running Ragone plot for solution {i + 1} of {len(ageing_solutions)}")
        new_model = model.set_initial_conditions_from(first_state, inplace=False)
        sim = RagoneSimulation(
            new_model,
            parameter_values=parameter_values,
            value_range=value_range,
            solver=solver,
            mode=mode,
            var_pts=var_pts,
        )

        sol = sim.solve()

        my_plt = RagonePlot(sol, labels=None, volume=volume, scale=scale, fit=True)
        fig, ax = my_plt.plot(show_plot=False)

        ax.annotate(
            f"$E_0$ = {np.exp(sol._raw_metrics[0]):.2f},\n $P_0$ = {sol._raw_metrics[1]:.2f},\n n = {sol._raw_metrics[2]:.2f}",
            xy=(0.05, 0.05),
            xycoords="axes fraction",
        )

        ax.axhline(
            np.exp(sol._raw_metrics[0]), color="lightgray", linestyle="--", label="E_0"
        )
        ax.axvline(sol._raw_metrics[1], color="lightgray", linestyle="--", label="P_0")

        fig.savefig(
            Path("figures")
            / "fits"
            / f"ragone_ageing_fit_{mode}{tag}_{scale}_cycle_{step * i}.png",
            dpi=300,
        )

        solutions.append(sol)

    plts = RagonePlot(solutions, labels=labels, volume=volume, scale=scale)
    fig, _ = plts.plot(show_plot=False)
    fig.savefig(Path("figures") / f"ragone_ageing_{mode}{tag}_{scale}.png", dpi=300)

    if mode == "power":
        metrics = {"Cycle number": []}
        for cycle, sol in zip(cycles, solutions):
            sol.fit_log()
            metrics["Cycle number"].append(cycle + 1)

            for key in sol.metrics.keys():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(sol.metrics[key])

        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(
            Path("data") / f"ragone_ageing_metrics_{scale}{tag}.csv", index=False
        )
