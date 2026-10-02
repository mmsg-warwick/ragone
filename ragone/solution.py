import numpy as np
from scipy.optimize import curve_fit


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

        self._raw_metrics = None
        self._metrics = None

    def plot(self, labels=None, volume=None, mass=None, scale="loglog"):
        from ragone.plotting import RagonePlot

        plot = RagonePlot(self, labels=labels, volume=volume, mass=mass, scale=scale)
        return plot.plot()

    def _gaussian_log(self, x, E0, P0, n):
        return (E0 - (10**x / P0) ** n) / np.log(10)

    def _gaussian(self, x, E0, P0, n):
        return E0 * np.exp(-((x / P0) ** n))

    def _gaussian_loglog(self, x, E0, P0, n):
        return (E0 - (10**x / P0) ** n) / np.log(10)

    def _gaussian_linear(self, x, E0, P0, n):
        return E0 * np.exp(-((x / P0) ** n))

    def fit_log(self):
        log_input = np.log10(self.data[self.input])
        log_output = np.log10(self.data[self.output])

        popt, _ = curve_fit(
            self._gaussian_log,
            log_input,
            log_output,
            bounds=(0, np.inf),
            nan_policy="omit",
        )

        self._raw_metrics = popt

        # Compute residuals
        residuals = log_output - self._gaussian_log(log_input, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((log_output - np.mean(log_output)) ** 2)
        r_squared = 1 - ss_res / ss_tot

        self.metrics = {
            f"Reference {self.output[0].lower() + self.output[1:]}": np.exp(popt[0]),
            f"Reference {self.input[0].lower() + self.input[1:]}": popt[1],
            "n": popt[2],
            "R^2": r_squared,
        }

        return popt

    def fit(self):
        if self.scale == "loglog":
            fit_fun = self._gaussian_loglog
            fit_input = np.log10(self.data[self.input])
            fit_output = np.log10(self.data[self.output])
        elif self.scale == "linear":
            fit_fun = self._gaussian_linear
            fit_input = self.data[self.input]
            fit_output = self.data[self.output]

        popt, _ = curve_fit(
            fit_fun, fit_input, fit_output, bounds=(0, np.inf), nan_policy="omit"
        )

        self._raw_metrics = popt
        self.metrics = {
            f"Reference {self.output[0].lower() + self.output[1:]}": np.exp(popt[0]),
            f"Reference {self.input[0].lower() + self.input[1:]}": popt[1],
            "n": popt[2],
            "Fitting scale": self.scale,
        }
        return popt
