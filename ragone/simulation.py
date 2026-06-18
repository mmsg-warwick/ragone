import numpy as np
import pybamm

from ragone.solution import RagoneSolution


class RagoneSimulation:
    def __init__(
        self,
        model,
        value_range,
        parameter_values=None,
        solver=None,
        var_pts=None,
        mode="power",
        direction="discharge",
        convert_to_watts=False,
    ):
        self.model = model
        self.value_range = value_range
        self.parameter_values = parameter_values or model.default_parameter_values
        self.solver = solver or model.default_solver
        self.var_pts = var_pts or model.default_var_pts

        self.mode = mode
        self.direction = direction
        self.convert_to_watts = convert_to_watts

        if self.mode == "power":
            self.input = "Power [W]"
            self.output = "Energy [W.h]"
            self.step_fun = pybamm.step.power
            self.ref_value = (
                self.parameter_values["Nominal cell capacity [A.h]"]
                * (
                    self.parameter_values["Upper voltage cut-off [V]"]
                    + self.parameter_values["Lower voltage cut-off [V]"]
                )
                / 2
            )
        elif self.mode == "current":
            self.input = "Current [A]"
            self.output = "Capacity [A.h]"
            self.step_fun = pybamm.step.current
            self.ref_value = self.parameter_values["Nominal cell capacity [A.h]"]
        else:
            raise ValueError("mode must be either 'power' or 'current'")

        if self.direction == "discharge":
            self.sign = 1
            self.termination = pybamm.step.VoltageTermination(
                self.parameter_values["Lower voltage cut-off [V]"]
            )
        elif self.direction == "charge":
            self.sign = -1
            self.termination = pybamm.step.VoltageTermination(
                self.parameter_values["Upper voltage cut-off [V]"]
            )
        else:
            raise ValueError(
                f"Invalid `direction`: {self.direction}. It should be `charge` or `discharge`"
            )

    def _compute_theoretical_value(self):
        # Compute low rate solution for theoretical value (energy/capacity)
        step = self.step_fun(
            self.sign * 0.001 * self.ref_value,
            duration=1e7,
            termination=self.termination,
        )
        experiment = pybamm.Experiment([step])
        sim = pybamm.Simulation(
            self.model,
            parameter_values=self.parameter_values,
            experiment=experiment,
            solver=self.solver,
            var_pts=self.var_pts,
        )
        sol = sim.solve(calc_esoh=False)

        return sol["Time [h]"].entries[-1] * 0.001 * self.ref_value

    def solve(self):
        # Compute theoretical capacity/energy
        theoretical_value = self._compute_theoretical_value()

        # Initialize list to store solutions
        solutions = []

        # Loop over value range
        for i, value in enumerate(self.value_range):
            print(f"Running simulation {i + 1} of {len(self.value_range)}")
            duration = 1e4 / value * self.ref_value

            try:
                step = self.step_fun(
                    self.sign * value, duration=duration, termination=self.termination
                )
                experiment = pybamm.Experiment([step])
                sim = pybamm.Simulation(
                    self.model,
                    parameter_values=self.parameter_values,
                    experiment=experiment,
                    solver=self.solver,
                    var_pts=self.var_pts,
                )

                sol = sim.solve(
                    t_interp=np.array([0, 1]),
                    calc_esoh=False,
                )

            except pybamm.SolverError as e:
                print(f"Solver failed: {e}")
                sol = None

            solutions.append(sol)

            if sol is not None:
                output = sol["Time [h]"].entries[-1] * value
                if output < 0.1 * theoretical_value:
                    print(
                        f"{self.output} too low ({output:.2f} Wh < {0.1 * theoretical_value:.2f}). Stopping simulations."
                    )
                    break

        times = []
        outputs = []
        inputs = []

        input_watts = []
        output_watts = []

        for sol, value in zip(solutions, self.value_range):
            if sol is None:
                times.append(np.nan)
                outputs.append(np.nan)
                inputs.append(np.nan)
                if self.mode == "current" and self.convert_to_watts:
                    input_watts.append(np.nan)
                    output_watts.append(np.nan)
            else:
                time = sol["Time [h]"].entries[-1]
                input = value
                output = input * time

                times.append(time)
                outputs.append(output)
                inputs.append(input)

                if self.mode == "current" and self.convert_to_watts:
                    # energy = self.sign * np.trapz(sol["Power [W]"].entries, sol["Time [h]"].entries)
                    energy = self.sign * (
                        sol["Discharge energy [W.h]"].entries[-1]
                        - sol["Discharge energy [W.h]"].entries[0]
                    )
                    input_watts.append(energy / time)
                    output_watts.append(energy)

        data = {
            "Time [h]": np.array(times),
            self.input: np.array(inputs),
            self.output: np.array(outputs),
        }

        if self.mode == "current" and self.convert_to_watts:
            data["Power [W]"] = np.array(input_watts)
            data["Energy [W.h]"] = np.array(output_watts)

        self.solution = RagoneSolution(data, self.mode)
        return self.solution
