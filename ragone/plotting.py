import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np


class RagonePlot:
    def __init__(
        self,
        solutions,
        labels=None,
        volume=None,
        mass=None,
        colormap="plasma",
        scale="loglog",
        fit=False,
    ):
        self.solutions = solutions if isinstance(solutions, list) else [solutions]

        modes = {sol.mode for sol in self.solutions}
        if len(modes) == 1:
            self.mode = self.solutions[0].mode
            self.input = self.solutions[0].input
            self.output = self.solutions[0].output
        if len(modes) > 1:
            self.mode = (
                "power"  # plot power by default, TODO: allow to plot current instead
            )
            self.input = "Power [W]"
            self.output = "Energy [W.h]"
            for sol in self.solutions:
                if sol.mode == "current" and (
                    "Power [W]" not in sol.data.keys()
                    or "Energy [W.h]" not in sol.data.keys()
                ):
                    raise ValueError(
                        "All solutions must either have the same mode or have the"
                        " input/output variables converted to watts (use "
                        "`convert_to_watts=True` in the RagoneSimulation)."
                    )

        self._compute_data_limits()

        self.labels = labels

        if scale not in ["loglog", "linear"]:
            raise ValueError("scale must be either 'loglog' or 'linear'")

        self.scale = scale
        self.fit = fit

        self.volume = volume
        self.mass = mass

        cmap = colormaps[colormap]
        self.colors = cmap(np.linspace(0, 0.9, len(self.solutions)))

    def _compute_data_limits(self):
        self.min_input = min(
            [np.nanmin(sol.data[self.input]) for sol in self.solutions]
        )
        self.max_input = max(
            [np.nanmax(sol.data[self.input]) for sol in self.solutions]
        )
        self.min_output = min(
            [np.nanmin(sol.data[self.output]) for sol in self.solutions]
        )
        self.max_output = max(
            [np.nanmax(sol.data[self.output]) for sol in self.solutions]
        )

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
        if self.scale == "loglog":
            self.ax.set_xlim([self.min_input, self.max_input])
            self.ax.set_ylim([self.y_min, 1.1 * self.max_output])
        elif self.scale == "linear":
            self.ax.set_xlim([0, self.max_input])
            self.ax.set_ylim([0, 1.1 * self.max_output])

    def _set_axes_ticks(self):
        if self.scale == "loglog":
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            x_ticks = self._get_ticks_range(xlim[0], xlim[1])
            y_ticks = self._get_ticks_range(ylim[0], ylim[1])
            self.ax.set_xticks(x_ticks)
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
        if self.scale == "loglog":
            p1 = self.ax.transData.transform_point((1, 1))
            p2 = self.ax.transData.transform_point((2, 2))
            dy = p2[1] - p1[1]
            dx = p2[0] - p1[0]
            rotations = [np.degrees(np.arctan2(dy, dx))] * 3
        elif self.scale == "linear":
            rotations = []
            for t in [3, 2, 1.5]:
                p1 = self.ax.transData.transform_point((1, 1))
                p2 = self.ax.transData.transform_point((2, t))
                dy = p2[1] - p1[1]
                dx = p2[0] - p1[0]
                rotations.append(np.degrees(np.arctan2(dy, dx)))

        y_lim = self.ax.get_ylim()
        v_post = 0.5
        x0 = (
            self.y_min * (y_lim[1] / self.y_min) ** v_post
        )  # weighted average in log scale
        t0 = 1  # isochrone that we place at location x0
        label_hshift = 0.9  # shift so label doesn't overlap with line

        for label, scale, rotn in zip(["2 h", "1 h", "30 min"], [2, 1, 0.5], rotations):
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

    def _set_secondary_axes(self, scaling, shift=None, fontsize=None):
        fontsize = fontsize or plt.rcParams["font.size"]
        x_position = shift or "top"
        y_position = shift or "right"

        if scaling == "volume":
            scaling_factor = self.volume
        elif scaling == "mass":
            scaling_factor = self.mass

        def ext2int(x):
            return x / scaling_factor

        def int2ext(x):
            return x * scaling_factor

        secx = self.ax.secondary_xaxis(x_position, functions=(ext2int, int2ext))
        secy = self.ax.secondary_yaxis(y_position, functions=(ext2int, int2ext))

        # Set ticks for secondary axes (loglog only)
        if self.scale == "loglog":
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            x_ticks = self._get_ticks_range(ext2int(xlim[0]), ext2int(xlim[1]))
            y_ticks = self._get_ticks_range(ext2int(ylim[0]), ext2int(ylim[1]))

            # # X secondary axis
            # secx.xaxis.set_major_locator(FixedLocator(x_ticks))
            # secx.xaxis.set_major_formatter(
            #     FixedFormatter(self._format_tick_labels(x_ticks))
            # )
            # secx.minorticks_off()

            # # Y secondary axis
            # secy.yaxis.set_major_locator(FixedLocator(y_ticks))
            # secy.yaxis.set_major_formatter(
            #     FixedFormatter(self._format_tick_labels(y_ticks))
            # )
            # secy.minorticks_off()

            secx.set_xticks(x_ticks, labels=self._format_tick_labels(x_ticks))
            # secx.set_xticklabels(self._format_tick_labels(x_ticks), fontsize=fontsize)
            secx.minorticks_off()

            secy.set_yticks(y_ticks, labels=self._format_tick_labels(y_ticks))
            # secy.set_yticklabels(self._format_tick_labels(y_ticks), fontsize=fontsize)
            secy.minorticks_off()

        def convert_labels(label, scaling):
            if "Energy" in label:
                quantity = "Energy"
                unit = "W.h"
            elif "Capacity" in label:
                quantity = "Capacity"
                unit = "A.h"
            elif "Power" in label:
                quantity = "Power"
                unit = "W"
            elif "Current" in label:
                quantity = "Current"
                unit = "A"

            if scaling == "volume":
                return f"{quantity} density [{unit}.l$^{{-1}}$]"
            elif scaling == "mass":
                return f"Specific {quantity.lower()} [{unit}.kg$^{{-1}}$]"

        secx.tick_params(axis="x", labelsize=fontsize)
        secy.tick_params(axis="y", labelsize=fontsize)
        secx.set_xlabel(convert_labels(self.input, scaling), fontsize=fontsize)
        secy.set_ylabel(convert_labels(self.output, scaling), fontsize=fontsize)

    def plot(self, show_plot=True):
        plt.rcParams.update({"font.size": 14})
        self.fig, self.ax = plt.subplots(constrained_layout=True)
        skip_legend = False

        if self.labels is None:
            self.labels = [None] * len(self.solutions)
            skip_legend = True

        # set axes limits now so we don't mess with text rotation later
        self._set_axes_limits()

        # Draw isochrones
        self._draw_isochrones()

        # Draw Ragone plots
        if self.scale == "loglog":
            plotfun = self.ax.loglog
        elif self.scale == "linear":
            plotfun = self.ax.plot

        for sol, color, label in zip(self.solutions, self.colors, self.labels):
            if self.fit:
                if sol._raw_metrics is None:
                    sol.fit_log()
                marker = "."
                linestyle = "none"

                plotfun(
                    sol.data[self.input],
                    sol._gaussian(
                        sol.data[self.input],
                        np.exp(sol._raw_metrics[0]),
                        sol._raw_metrics[1],
                        sol._raw_metrics[2],
                    ),
                    color="darkgray",
                    label=None,
                    linestyle="-",
                )
            else:
                marker = None
                linestyle = "-"

            plotfun(
                sol.data[self.input],
                sol.data[self.output],
                color=color,
                label=label,
                marker=marker,
                linestyle=linestyle,
            )

        # Set labels
        self.ax.set_xlabel(self.input)
        self.ax.set_ylabel(self.output)
        self._set_axes_ticks()

        # Produce secondary axes
        if self.volume and self.mass:
            self._set_secondary_axes(scaling="volume", fontsize=10)
            self._set_secondary_axes(scaling="mass", shift=1.15, fontsize=10)

            # secondary_axes = [ax for ax in self.fig.axes if ax is not self.ax]

            # for secax in secondary_axes:
            #     for label in secax.get_xticklabels() + secax.get_yticklabels():
            #         label.set_fontsize(10)

        elif self.volume:
            self._set_secondary_axes(scaling="volume")
        elif self.mass:
            self._set_secondary_axes(scaling="mass")

        if not skip_legend:
            if self.scale == "loglog":
                self.ax.legend(loc="lower left", fontsize=10)
            elif self.scale == "linear":
                self.ax.legend(loc="upper right", fontsize=10)
        # self.fig.tight_layout()

        # annotate isochrones (in the end to get the right transformation)
        self._annotate_isochrones()

        if show_plot:  # pragma: no cover
            plt.show()

        return self.fig, self.ax
