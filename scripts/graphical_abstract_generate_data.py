import pickle

import numpy as np
import pandas as pd
import pybamm

from ragone import (
    ROOT,
    RagoneSimulation,
    get_options,
    get_parameter_values,
    get_var_pts,
)

options, tag = get_options(SEI=True, plating=True, lam=True)
step = 250
tag = "_fast" + tag
scale = "loglog"

# Generate data for Ragone plots
print("Setting up model and parameters...")
model = pybamm.lithium_ion.DFN(
    options=options,
)
parameter_values = get_parameter_values(ageing=False)
solver = pybamm.IDAKLUSolver(
    rtol=1e-6,
    atol=1e-8,
    options={
        "max_error_test_failures": 200,
        "max_convergence_failures": 20000,
        "max_nonlinear_iterations": 400,
    },
)

print("Loading aged solution...")
aged_sol = pybamm.load(ROOT / "data" / f"aged_solution{tag}.pkl")
print("Aged solution loaded.")

var_pts = get_var_pts()

cycles = [0] + [
    i * step - 1 for i in range(1, len(aged_sol.all_first_states) // step + 1)
]

labels = [f"Cycle {i + 1}" for i in cycles]
labels[1:-1] = [None] * (len(labels) - 2)

print("Extracting relevant cycles")
ageing_solutions = [aged_sol.all_first_states[0]] + aged_sol.all_first_states[
    step - 1 :: step
]
value_range = np.logspace(np.log10(0.5), np.log10(100), 50)

print("Running Ragone plots for power...")
solutions = []

for i, first_state in enumerate(ageing_solutions):
    print(f"Running Ragone plot for solution {i + 1} of {len(ageing_solutions)}")
    new_model = model.set_initial_conditions_from(first_state, inplace=False)
    sim = RagoneSimulation(
        new_model,
        parameter_values=parameter_values,
        value_range=value_range,
        solver=solver,
        mode="power",
        var_pts=var_pts,
    )

    sol = sim.solve()

    solutions.append(sol)

with open(ROOT / "data" / "graphical_abstract_ragone_solutions.pkl", "wb") as file:
    pickle.dump(solutions, file)


metrics = {"Cycle number": []}
for cycle, sol in zip(cycles, solutions):
    sol.fit_log()
    metrics["Cycle number"].append(cycle + 1)

    for key in sol.metrics:
        if key not in metrics:
            metrics[key] = []
        metrics[key].append(sol.metrics[key])

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(
    ROOT / "data" / "graphical_abstract_ageing_metrics.csv",
    index=False,
)

# Generate data for RPT plots
step = 25
ageing_solutions = [aged_sol.all_first_states[0]] + aged_sol.all_first_states[
    step - 1 :: step
]
cycles = [0] + [
    i * step - 1 for i in range(1, len(aged_sol.all_first_states) // step + 1)
]
value_range = np.logspace(np.log10(2), np.log10(20), 4)
data = {"Cycle number": cycles}
for value in value_range:
    print(f"Running RPT for {value} W...")
    solutions = []
    for i, first_state in enumerate(ageing_solutions):
        print(f"Running RPT for solution {i + 1} of {len(ageing_solutions)}")
        new_model = model.set_initial_conditions_from(first_state, inplace=False)
        sim = RagoneSimulation(
            new_model,
            parameter_values=parameter_values,
            value_range=[value],
            solver=solver,
            # solver=pybamm.IDAKLUSolver(),
            mode="power",
            var_pts=var_pts,
        )

        sol = sim.solve()

        solutions.append(sol.data[sol.output][0])

    data[value] = solutions

with open(ROOT / "data" / "graphical_abstract_rpt_data.pkl", "wb") as file:
    pickle.dump(data, file)
