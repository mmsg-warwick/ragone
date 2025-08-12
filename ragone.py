import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import colormaps
import numpy as np
import pybamm
from scipy.optimize import curve_fit


def get_options(SEI=False, plating=False, lam=False):
    tag = ""
    options = {}
    if SEI:
        tag += "_SEI"
        options["SEI"] = "reaction limited"
        options["SEI porosity change"] = "true"
    if plating:
        tag += "_plating"
        options["lithium plating"] = "irreversible"
        options["lithium plating porosity change"] = "true"
    if lam:
        tag += "_lam"
        options["particle mechanics"] = "swelling only"
        options["loss of active material"] = "stress-driven"

    return options, tag


def get_parameter_values(ageing=True):
    parameter_values = pybamm.ParameterValues("OKane2022")
    Chen2020 = pybamm.ParameterValues("Chen2020")
    parameter_values["Negative electrode OCP [V]"] = Chen2020[
        "Negative electrode OCP [V]"
    ]
    if ageing:
        parameter_values["SEI reaction exchange current density [A.m-2]"] = (
            1.5e-7 * 0.15 * 1
        )  # was *2
        parameter_values["Lithium plating kinetic rate constant [m.s-1]"] = 5e-12 * 2
        parameter_values["Negative electrode LAM constant proportional term [s-1]"] = (
            2.7778e-07 * 2
        )
        parameter_values["Positive electrode LAM constant proportional term [s-1]"] = (
            2.7778e-07 * 2
        )
        parameter_values["Negative electrode LAM constant exponential term"] = 1.3
        parameter_values["Positive electrode LAM constant exponential term"] = 1.3
    else:
        parameter_values["SEI kinetic rate constant [m.s-1]"] = 0
        parameter_values["SEI reaction exchange current density [A.m-2]"] = 0
        parameter_values["SEI solvent diffusivity [m2.s-1]"] = 0
        parameter_values["Lithium plating kinetic rate constant [m.s-1]"] = 0
        parameter_values["Negative electrode LAM constant proportional term [s-1]"] = 0
        parameter_values["Positive electrode LAM constant proportional term [s-1]"] = 0

    return parameter_values


class RagoneSolution:
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode

        if self.mode == "power":
            self.input = "Power [W]"
            self.output = "Energy [W.h]"
        elif self.mode == "current":
            self.input = "Current [A]"
            self.output = "Capacity [A.h]"

        self.min_input = np.nanmin(self.data[self.input])
        self.max_input = np.nanmax(self.data[self.input])
        self.min_output = np.nanmin(self.data[self.output])
        self.max_output = np.nanmax(self.data[self.output])

        self._raw_metrics = None
        self._metrics = None

    def plot(self, labels=None, volume=None, mass=None):
        plot = RagonePlot(self, labels=labels, volume=volume, mass=mass)
        return plot.plot()

    def _gaussian_log(self, x, E0, P0, n):
        return (E0 - (10**x / P0) ** n) / np.log(10)

    def _gaussian(self, x, E0, P0, n):
        return E0 * np.exp(-((x / P0) ** n))

    def fit_log(self):
        log_input = np.log10(self.data[self.input])
        log_output = np.log10(self.data[self.output])

        popt, _ = curve_fit(
            self._gaussian_log, log_input, log_output, bounds=(0, np.inf)
        )

        self._raw_metrics = popt
        self.metrics = {
            f"Reference {self.output[0].lower() + self.output[1:]}": np.exp(popt[0]),
            f"Reference {self.input[0].lower() + self.input[1:]}": popt[1],
            "n": popt[2],
        }
        return popt

    def fit_gaussian(self):
        popt, _ = curve_fit(
            self._gaussian,
            self.data[self.input],
            self.data[self.output],
            bounds=(0, np.inf),
        )
        return popt

    def plot_log(self):
        print(self.metrics)
        self.plot_fit()

    def plot_gaussian(self, show_plot=True):
        popt = self.fit_gaussian()
        print(popt)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.loglog(self.data[self.input], self.data[self.output], "kx")
        ax.loglog(
            self.data[self.input],
            self._gaussian(
                self.data[self.input],
                popt[0],
                popt[1],
                popt[2],
            ),
        )
        ax.set_xlabel(self.input)
        ax.set_ylabel(self.output)
        ax.axhline(popt[0], color="lightgray", linestyle="--", label="E0")
        ax.axvline(popt[1], color="lightgray", linestyle="--", label="P0")
        ax.annotate(
            f"E0 = {popt[0]:.2f},\n P0 = {popt[1]:.2f},\n n = {popt[2]:.2f}",
            xy=(0.05, 0.05),
            xycoords="axes fraction",
        )

        if show_plot:
            plt.show()

        return fig, ax

    def plot_fit(self, show_plot=True):
        if self._raw_metrics is None:
            self.fit_log()

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.loglog(self.data[self.input], self.data[self.output], "kx")
        ax.loglog(
            self.data[self.input],
            self._gaussian(
                self.data[self.input],
                np.exp(self._raw_metrics[0]),
                self._raw_metrics[1],
                self._raw_metrics[2],
            ),
        )
        ax.annotate(
            f"E0 = {np.exp(self._raw_metrics[0]):.2f},\n P0 = {self._raw_metrics[1]:.2f},\n n = {self._raw_metrics[2]:.2f}",
            xy=(0.05, 0.05),
            xycoords="axes fraction",
        )

        ax.set_xlabel(self.input)
        ax.set_ylabel(self.output)
        ax.axhline(
            np.exp(self._raw_metrics[0]), color="lightgray", linestyle="--", label="E0"
        )
        ax.axvline(self._raw_metrics[1], color="lightgray", linestyle="--", label="P0")

        if show_plot:
            plt.show()

        return fig, ax

    # @property
    # def raw_metrics(self):
    #     if self._raw_metrics is None:
    #         self.fit_log()
    #     return self._raw_metrics

    # @property
    # def metrics(self):
    #     if self._metrics is None:
    #         self.fit_log()
    #     return self._metrics

    # @metrics.setter
    # def metrics(self, value):
    #     self._metrics = value


class RagonePlot:
    def __init__(
        self, solutions, labels=None, volume=None, mass=None, colormap="plasma"
    ):
        self.solutions = solutions if isinstance(solutions, list) else [solutions]

        # TODO: fix for multiple modes
        modes = {sol.mode for sol in self.solutions}
        if len(modes) == 1:
            self.mode = self.solutions[0].mode
            self.input = self.solutions[0].input
            self.output = self.solutions[0].output
        if len(modes) > 1:
            self.mode = "power"
            self.input = "Power [W]"
            self.output = "Energy [W.h]"
            for sol in self.solutions:
                if sol.mode == "current" and (
                    "Power [W]" not in sol.data.keys() or "Energy [W.h]" not in sol.data.keys()
                ):
                    raise ValueError(
                        "All solutions must either have the same mode or have the"
                        " input/output variables converted to watts (use "
                        "`convert_to_watts=True` in the RagoneSimulation)."
                    )

        self._compute_data_limits()

        self.labels = labels

        if volume and mass:
            raise ValueError("Only one of volume or mass can be provided")

        self.volume = volume
        self.mass = mass
        self.scaling = volume or mass
        self.scaling_unit = "l" if volume else "kg" if mass else None

        cmap = colormaps[colormap]
        self.colors = cmap(np.linspace(0, 0.9, len(self.solutions)))

    def _compute_data_limits(self):
        self.min_input = min([sol.min_input for sol in self.solutions])
        self.max_input = max([sol.max_input for sol in self.solutions])
        self.min_output = min([sol.min_output for sol in self.solutions])
        self.max_output = max([sol.max_output for sol in self.solutions])

    def _get_ticks_range(self, tick_min, tick_max):
        decades = np.floor(np.log10(tick_min)), np.ceil(np.log10(tick_max))
        ticks = []
        for exp in range(int(decades[0]), int(decades[1]) + 1):
            for factor in [1, 2, 5]:
                tick = factor * 10**exp
                if tick_min <= tick <= tick_max:
                    ticks.append(tick)
        return ticks

    def _format_tick_labels(self, xs):
        labels = []
        for x in xs:
            if x == int(x):
                labels.append(str(int(x)))
            else:
                labels.append(str(x))
        return labels

    def _set_axes_limits(self):
        self.y_min = max([self.min_output, 0.1 * self.max_output])
        self.ax.set_xlim([self.min_input, self.max_input])
        self.ax.set_ylim([self.y_min, 1.1 * self.max_output])

    def _set_axes_ticks(self):
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x_ticks = self._get_ticks_range(xlim[0], xlim[1])
        y_ticks = self._get_ticks_range(ylim[0], ylim[1])
        self.ax.set_xticks(x_ticks)
        print(self._format_tick_labels(x_ticks))
        self.ax.set_xticklabels(self._format_tick_labels(x_ticks))
        self.ax.set_yticks(y_ticks)
        self.ax.set_yticklabels(self._format_tick_labels(y_ticks))
        # self.ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
        # self.ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
        self.ax.minorticks_off()

    def _draw_isochrones(self):
        # compute max and min isochrones
        min_iso = np.floor(np.log(self.min_output / self.max_input) / np.log(2))
        max_iso = np.ceil(np.log(self.max_output / self.min_input) / np.log(2))

        for iso in np.arange(min_iso, max_iso + 1):
            self.ax.axline(
                (1, 2**iso),
                (2, 2 ** (iso + 1)),
                color="darkgray",
                linestyle=":",
                linewidth=0.5,
            )

    def _annotate_isochrones(self):
        p1 = self.ax.transData.transform_point((1, 1))
        p2 = self.ax.transData.transform_point((2, 2))
        dy = p2[1] - p1[1]
        dx = p2[0] - p1[0]
        rotn = np.degrees(np.arctan2(dy, dx))

        y_lim = self.ax.get_ylim()
        v_post = 0.5
        x0 = y_lim[0] * (y_lim[1] / y_lim[0]) ** v_post  # weighted average in log scale
        t0 = 1  # isochrone that we place at location x0
        label_hshift = 0.9  # shift so label doesn't overlap with line

        for label, scale in zip(["2 h", "1 h", "30 min"], [2, 1, 0.5]):
            # Rationale: choose the position for the reference label and arrange
            # others along line perpendicular to isochrones
            x_label = label_hshift * np.sqrt(t0 * x0**2 / scale)
            y_label = (t0 * x0**2) / x_label
            self.ax.annotate(
                label,
                xy=(x_label, y_label),
                ha="center",
                va="center",
                rotation=rotn,
                color="darkgray",
                fontsize=8,
            )

    def _set_secondary_axes(self):
        def ext2int(x):
            return x / self.scaling

        def int2ext(x):
            return x * self.scaling

        secx = self.ax.secondary_xaxis("top", functions=(ext2int, int2ext))
        secy = self.ax.secondary_yaxis("right", functions=(ext2int, int2ext))

        # Set ticks for secondary axes
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        x_ticks = self._get_ticks_range(ext2int(xlim[0]), ext2int(xlim[1]))
        y_ticks = self._get_ticks_range(ext2int(ylim[0]), ext2int(ylim[1]))
        secx.set_xticks(x_ticks)
        secx.set_xticklabels(self._format_tick_labels(x_ticks))
        secy.set_yticks(y_ticks)
        secy.set_yticklabels(self._format_tick_labels(y_ticks))
        # secx.xaxis.set_major_formatter(ticker.ScalarFormatter())
        # secy.yaxis.set_major_formatter(ticker.ScalarFormatter())
        secx.minorticks_off()
        secy.minorticks_off()

        def convert_labels(label):
            split_label = label.split("]")[0]
            split_label = split_label[0].lower() + split_label[1:]
            new_label = "Specific " + split_label + "." + self.scaling_unit + "$^{-1}$]"
            return new_label

        secx.set_xlabel(convert_labels(self.input))
        secy.set_ylabel(convert_labels(self.output))

    def plot(self, show_plot=True):
        plt.rcParams.update({"font.size": 14})
        self.fig, self.ax = plt.subplots()
        skip_legend = False

        if self.labels is None:
            self.labels = [None] * len(self.solutions)
            skip_legend = True

        # set axes limits now so we don't mess with text rotation later
        self._set_axes_limits()

        # Draw isochrones
        self._draw_isochrones()

        # Draw Ragone plots
        for sol, color, label in zip(self.solutions, self.colors, self.labels):
            self.ax.loglog(
                sol.data[self.input], sol.data[self.output], color=color, label=label
            )

        # Set labels
        self.ax.set_xlabel(self.input)
        self.ax.set_ylabel(self.output)
        self._set_axes_ticks()

        # Produce secondary axes
        if self.scaling:
            self._set_secondary_axes()

        if not skip_legend:
            self.ax.legend()
        self.fig.tight_layout()

        # annotate isochrones (in the end to get the right transformation)
        self._annotate_isochrones()

        if show_plot:  # pragma: no cover
            plt.show()

        return self.fig, self.ax


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
            print(f"Running simulation {i+1} of {len(self.value_range)}")
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
                    energy = self.sign * np.trapz(sol["Power [W]"].entries, sol["Time [h]"].entries)
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
