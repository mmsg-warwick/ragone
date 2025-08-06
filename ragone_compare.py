import pybamm
import numpy as np
import gc
from util import compute_ragone, plot_ragone

model = pybamm.lithium_ion.DFN(
    options={
        "SEI": "none",
        "calculate discharge energy": "true",
    }
)

parameter_values = pybamm.ParameterValues("OKane2022")
Chen2020 = pybamm.ParameterValues("Chen2020")
parameter_values["Negative electrode OCP [V]"] = Chen2020["Negative electrode OCP [V]"]
parameter_values["Electrolyte diffusivity [m2.s-1]"] = 1.7694e-10
parameter_values["Electrolyte conductivity [S.m-1]"] = 0.9487

volume = parameter_values["Cell volume [m3]"] * 1000

solver = pybamm.IDAKLUSolver()

var_pts = {
    "x_n": 50,
    "x_s": 20,
    "x_p": 50,
    "r_n": 20,
    "r_p": 20,
}
experiment_setup = pybamm.Experiment(
    [
        "Discharge at C/10 until 2.5V",
        "Hold at 2.5V until C/50",
    ]
)

aged_sol = pybamm.load("solution.pkl")
model.set_initial_conditions_from(aged_sol.all_first_states[-1], inplace=True)

sim_setup = pybamm.Simulation(
    model,
    parameter_values=parameter_values,
    experiment=experiment_setup,
    solver=solver,
)
sol_setup = sim_setup.solve()

model_discharged = model.set_initial_conditions_from(
    sol_setup.last_state, inplace=False
)

# modes = ["current"]
modes = ["power", "current"]
models = [model, model_discharged]
directions = ["discharge", "charge"]
outputs = []
inputs = []

ref_voltage = (
    parameter_values["Upper voltage cut-off [V]"]
    + parameter_values["Lower voltage cut-off [V]"]
) / 2

power_range = np.logspace(np.log10(0.1), np.log10(150), 50)
value_ranges = [
    power_range,
    [i / ref_voltage for i in power_range],
]

labels = []

i = 0
for value_range, mode in zip(value_ranges, modes):
    for model, direction in zip(models, directions):
        i += 1
        print(
            f"Running Ragone plot for solution {i} of {len(value_ranges) * len(models)}"
        )
        output, input = compute_ragone(
            model,
            parameter_values,
            value_range=value_range,
            mode=mode,
            direction=direction,
            convert_to_watts=True,
        )
        outputs.append(output)
        inputs.append(input)
        labels.append(f"{mode} - {direction}")

fig = plot_ragone(inputs, outputs, mode="power", labels=labels)
fig.savefig("./figures/" + f"ragone_compare_modes_directions.png", dpi=300)
