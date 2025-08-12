import pybamm
import numpy as np
from ragone import RagoneSimulation, RagonePlot, get_parameter_values, get_options

# options, tag = get_options(SEI=False, plating=False, lam=False)

model = pybamm.lithium_ion.DFN()

parameter_values = get_parameter_values(ageing=False)

var_pts = {
    "x_n": 30,
    "x_s": 30,
    "x_p": 30,
    "r_n": 20,
    "r_p": 20,
}

solver = pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-10)

# Obtain discharged and charged solutions
experiment_dch = pybamm.Experiment(
    [
        "Discharge at C/10 until 2.5V",
        "Hold at 2.5V until C/50",
    ]
)
sim_dch = pybamm.Simulation(
    model,
    parameter_values=parameter_values,
    experiment=experiment_dch,
    var_pts=var_pts,
    solver=solver,
)
sol_dch = sim_dch.solve()

experiment_ch = pybamm.Experiment(
    [
        "Charge at C/10 until 4.2V",
        "Hold at 4.2V until C/50",
    ]
)
sim_ch = pybamm.Simulation(
    model,
    parameter_values=parameter_values,
    experiment=experiment_ch,
    var_pts=var_pts,
    solver=solver,
)
sol_ch = sim_ch.solve()


value_ranges = {
    "power": np.logspace(np.log10(0.5), np.log10(100), 5),
    "current": np.logspace(np.log10(0.1), np.log10(30), 5),
}

labels = []
solutions = []

for mode, value_range in value_ranges.items():
    print(f"Starting Ragone plots - {mode} mode")
    for direction, sol_init in [("charge", sol_dch), ("discharge", sol_ch)]:
        print(f"Running Ragone plot for direction: {direction}")
        labels.append(f"{mode} - {direction}")

        new_model = model.set_initial_conditions_from(sol_init.last_state, inplace=False)

        sim = RagoneSimulation(
            new_model,
            parameter_values=parameter_values,
            value_range=value_range,
            var_pts=var_pts,
            solver=solver,
            mode=mode,
            direction=direction,
            convert_to_watts=(mode == "current"),
        )

        sol = sim.solve()

        solutions.append(sol)



plt = RagonePlot(solutions, labels=labels)
fig, ax = plt.plot(show_plot=False)

# Set axes manually to match other figures (current plots "overflow")
solution = solutions[1]
ax.set_xlim(solution.min_input, 80)
ax.set_ylim(0.1 * solution.max_output, 1.1 * solution.max_output)

fig.savefig(
    "./figures/" + "ragone_compare_modes_directions.png",
    dpi=300,
)

ax.set_xscale("linear")
ax.set_yscale("linear")

fig.savefig(
    "./figures/" + "ragone_compare_modes_directions_linear.png",
    dpi=300,
)
