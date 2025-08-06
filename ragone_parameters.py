import pybamm
import numpy as np
from ragone import RagoneSimulation, RagonePlot, get_parameter_values, get_options
from pathlib import Path

options, tag = get_options(SEI=True, plating=True, lam=True)

model = pybamm.lithium_ion.DFN(
    options=options,
)

parameter_values = get_parameter_values(ageing=False)

volume = parameter_values["Cell volume [m3]"] * 1000

aged_sol = pybamm.load(Path("data") / f"aged_solution{tag}.pkl")

var_pts = {
    "x_n": 30,
    "x_s": 30,
    "x_p": 30,
    "r_n": 20,
    "r_p": 20,
}

step = 100

ageing_solutions = [aged_sol.all_first_states[0]] + aged_sol.all_first_states[
    step - 1 :: step
]



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
        fig.savefig(
            "./figures/" + f"ragone_{filename_extension[parameter_name]}_{mode}.png",
            dpi=300,
        )
