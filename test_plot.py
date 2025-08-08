import pybamm
import numpy as np
from ragone import RagoneSimulation, RagonePlot, get_options, get_parameter_values
from os import path

model = pybamm.lithium_ion.DFN()

parameter_values = get_parameter_values(ageing=False)

volume = parameter_values["Cell volume [m3]"] * 1000

var_pts = {
    "x_n": 30,
    "x_s": 30,
    "x_p": 30,
    "r_n": 20,
    "r_p": 20,
}

sim = RagoneSimulation(
    model,
    parameter_values=parameter_values,
    value_range=np.logspace(np.log10(0.5), np.log10(100), 10),
    solver=pybamm.IDAKLUSolver(rtol=1e-8, atol=1e-10),
    mode="power",
    var_pts=var_pts,
)

sol = sim.solve()

plts = RagonePlot(sol, volume=volume)
plts.plot()
