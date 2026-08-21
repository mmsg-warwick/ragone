# Ragone plots

[![Tests](https://github.com/mmsg-warwick/ragone/actions/workflows/tests.yml/badge.svg)](https://github.com/mmsg-warwick/ragone/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/mmsg-warwick/ragone/graph/badge.svg?token=uk6ryEFTRn)](https://codecov.io/gh/mmsg-warwick/ragone)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20762920.svg)](https://doi.org/10.5281/zenodo.20762920)

This repository contains the code to generate Ragone plots and reproduce the results of the article:

> F. Brosa Planella, S.J. Cooper There is no knee in Ragone: Clarifying battery energy and power fade using Ragone plots, _Under review_.

## Installation

Clone the repository and install it in editable mode:

```bash
git clone git@github.com:mmsg-warwick/ragone.git
cd ragone
pip install -e .
```

To also install the optional test dependencies:

```bash
pip install -e ".[test]"
```

## Reproducing the results

In order to reproduce the results of the article, you need to run the scripts in the `scripts/` folder. The scripts are designed to be run in a specific order, and they will produce the data files and figures needed for the article:

1. Run `run_ageing.py` to produce the .pkl files in the `data/` folder. These files contain the solutions of the ageing simulations, which are needed to extract the Ragone curves at different states of health.
2. You can now simultaneosly run:
    - `ragone_ageing.py` to produce the Ragone plots for a specific simulation at different cycle numbers (i.e. different states of health). This wil also compute the metrics that will be saved as .csv files in the `data/` folder, which are needed to run `plot_power_energy_fade.py`.
    - `ragone_compare.py` to produce the Ragone plots for all the combinations of cycling mode (power or current) and direction (charge or discharge).
    - `ragone_parameters.py` to produce the Ragone plots showing the effect of a single parameter.
    - `rpt.py` to produce the reference performance test plots (for the article the SEI plot uses a broader range of powers, this can be adjusted by uncommenting the appropriate line in `rpt.py`).
3. Run `plot_power_energy_fade.py` to produce the plots showing the normalised energy and power fade vs cycle number, comparing slow and fast charging.

## Repository structure

- `data/`: Contains the data files produced by the scripts. The .pkl files are the solutions of the ageing simulations, used later to extract the Ragone curves at different states of health, and they are needed to run the other scripts. The .csv files are the summary metrics extracted from the fits of the Ragone curves, and they are needed to run `plot_power_energy_fade.py`.
- `figures/`: Contains all the figures produced by the scripts (a lot!). There are various types of figures, and the naming convention explains what is being shown. Here are the main types of figures:
    - `ragone_ageing_{mode}_{degradation mechanisms}_{scale}.png`: These are Ragone plots for a specific simulations at different cycle numbers (i.e. different states of health). `mode` is how the battery is cycled (either power or current), `degradation mechanisms` is the combination of degradation mechanisms included in the simulation (e.g. SEI growth, lithium plating and/or LAM), and `scale` is whether the plot is in linear or loglog scale. Some plots have the tag `fast`, which means fast charging.
    - `fits/`: This folder contains the fits of the various Ragone curves (i.e. those in the previous point). The naming convention is the same as above, but with the tag `fit`and the cycle number. The fitted parameters are shown on the plot, but they are collected in the .csv files in the `data/` folder.
    - `ragone_compare_modes_directions_{scale}.png`: For a single simulation, it shows the Ragone curves for all the combinations of cycling mode (power or current) and direction (charge or discharge). `scale` is whether the plot is in linear or loglog scale.
    - `ragone_parameters_{mode}_{parameter}_{scale}.png`: Ragone plots showing the effect of a single parameter. The value is taken to be the average value for a given cycle number. The naming convention is the same as above, but with the tag `parameter` describing which parameter is being varied (e.g. either active material volume fraction or porosity, for each electrode).
    - `rpt_{mode}_{degradation mechanisms}.png`: Show the measured energy vs cycle number for various discharge powers (rate performance test).
    - `power_energy_fade_{degradation mechanisms}.png`: Show the normalised energy and power fade vs cycle number, comparing slow and fast charging.
    - `n_{degradation mechanisms}.png`: Show the evolution of the Ragone fit exponent `n` vs cycle number, comparing slow and fast charging.
    - `aged_solution_evolution_vf.png`: Show the evolution of electrode porosity and active material volume fractions with cycle number.
