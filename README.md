# Ragone plots

## Repository structure

- `data/`: Contains the data files produced by the scripts. The .pkl files are the solutions of the ageing simulations, used later to extract the Ragone curves at different states of health. The .csv files are the summary metrics extracted from the fits of the Ragone curves.
- `figures/`: Contains all the figures produced by the scripts (a lot!). There are various types of figures, and the naming convention explains what is being shown. Here are the main types of figures:
    - `ragone_ageing_{mode}_{degradation mechanisms}_{scale}.png`: These are Ragone plots for a specific simulations at different cycle numbers (i.e. different states of health). `mode` is how the battery is cycled (either power or current), `degradation mechanisms` is the combination of degradation mechanisms included in the simulation (e.g. SEI growth, lithium plating and/or LAM), and `scale` is whether the plot is in linear or loglog scale. Some plots have the tag `fast`, which means fast charging.
    - `fits/`: This folder contains the fits of the various Ragone curves (i.e. those in the previous point). The naming convention is the same as above, but with the tag `fit`and the cycle number. The fitted parameters are shown on the plot, but they are collected in the .csv files in the `data/` folder.
    - `ragone_compare_modes_directions_{scale}.png`: For a single simulation, it shows the Ragone curves for all the combinations of cycling mode (power or current) and direction (charge or discharge). `scale` is whether the plot is in linear or loglog scale.
    - `ragone_parameters_{mode}_{parameter}_{scale}.png`: Ragone plots showing the effect of a single parameter. The value is taken to be the average value for a given cycle number. The naming convention is the same as above, but with the tag `parameter` describing which parameter is being varied (e.g. either active material volume fraction or porosity, for each electrode).
    - `rate_capability_{mode}_{degradation mechanisms}.png`: Show the measured energy vs cycle number for various discharge powers.
