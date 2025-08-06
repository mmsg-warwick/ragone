import pybamm
import numpy as np
import gc
from util import compute_ragone, plot_ragone

model = pybamm.lithium_ion.DFN()

parameter_values = pybamm.ParameterValues("ORegan2022")

var_pts = {
    "x_n": 50,
    "x_s": 20,
    "x_p": 50,
    "r_n": 20,
    "r_p": 20,
}

T_range = [-5, 5, 15, 25, 35, 45, 55]
T_range = [25]

labels = [f"{T}°C" for T in T_range]

# modes = ["power", "current"]
value_ranges = {
    "power": np.logspace(np.log10(0.1), np.log10(150), 50),
    "current": np.logspace(np.log10(0.05), np.log10(50), 50),
}

solver = pybamm.IDAKLUSolver()

for mode, value_range in value_ranges.items():
    outputs = []
    inputs = []

    for T in T_range:
        print(f"Computing Ragone for {T}°C")
        edited_parameter_values = parameter_values.copy()
        edited_parameter_values["Ambient temperature [K]"] = T + 273.15
        edited_parameter_values["Reference temperature [K]"] = T + 273.15

        output, input = compute_ragone(
            model,
            edited_parameter_values,
            value_range=value_range,
            mode=mode,
            var_pts=var_pts,
            solver=solver,
        )
        outputs.append(output)
        inputs.append(input)

    fig = plot_ragone(inputs, outputs, labels=labels, mode=mode)
    fig.savefig("./figures/" + f"ragone_temperature_{mode}.png", dpi=300)
