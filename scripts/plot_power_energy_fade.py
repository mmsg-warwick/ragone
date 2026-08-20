import argparse

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import colormaps

from ragone import ROOT, get_options

plt.rcParams.update({"font.size": 14})

print("Processing command line arguments...")
parser = argparse.ArgumentParser(description="Run battery ageing simulation.")
parser.add_argument("--SEI", action="store_true", help="Enable SEI (default: disabled)")
parser.add_argument(
    "--plating", action="store_true", help="Enable plating (default: disabled)"
)
parser.add_argument("--lam", action="store_true", help="Enable LAM (default: disabled)")
parser.add_argument(
    "--fast", action="store_true", help="Fast charging (default: disabled)"
)

parser.set_defaults(SEI=False, plating=False, lam=False, fast=False)
args = parser.parse_args()

_, tag = get_options(SEI=args.SEI, plating=args.plating, lam=args.lam)
# step = args.step
# if args.fast:
#     tag = "_fast" + tag

slow_data = pd.read_csv(ROOT / "data" / f"ragone_ageing_metrics_loglog{tag}.csv")
fast_data = pd.read_csv(ROOT / "data" / f"ragone_ageing_metrics_loglog_fast{tag}.csv")

# Plot energy and power
fig, ax = plt.subplots(constrained_layout=True)
cmap = colormaps["plasma"]

# Energy plots (first color)
ax.plot(
    slow_data["Cycle number"],
    slow_data["Reference energy [W.h]"]
    / slow_data["Reference energy [W.h]"].iloc[0]
    * 100,
    label="Energy - slow",
    color=cmap(0),
    linestyle="-",
)
ax.plot(
    fast_data["Cycle number"],
    fast_data["Reference energy [W.h]"]
    / fast_data["Reference energy [W.h]"].iloc[0]
    * 100,
    label="Energy - fast",
    color=cmap(0),
    linestyle="--",
)

# Power plots (second color)
ax.plot(
    slow_data["Cycle number"],
    slow_data["Reference power [W]"] / slow_data["Reference power [W]"].iloc[0] * 100,
    label="Power - slow",
    color=cmap(0.6),
    linestyle="-",
)
ax.plot(
    fast_data["Cycle number"],
    fast_data["Reference power [W]"] / fast_data["Reference power [W]"].iloc[0] * 100,
    label="Power - fast",
    color=cmap(0.6),
    linestyle="--",
)

ax.set_xlabel("Cycle number")
ax.set_ylabel("Normalised energy/power [%]")
ax.set_xlim([0, 1000])
ax.set_ylim([0, 100])
ax.legend(fontsize=10)

fig.savefig(ROOT / "figures" / f"power_energy_fade{tag}.png", dpi=300)

# Plot n
fig, ax = plt.subplots(constrained_layout=True)

# n plots
ax.plot(
    slow_data["Cycle number"],
    slow_data["n"],
    label="slow",
    color=cmap(0),
    linestyle="-",
)
ax.plot(
    fast_data["Cycle number"],
    fast_data["n"],
    label="fast",
    color=cmap(0),
    linestyle="--",
)

ax.set_xlabel("Cycle number")
ax.set_ylabel("n")
ax.set_xlim([0, 1000])
ax.set_ylim([0, 4])
ax.legend(fontsize=10)

fig.savefig(ROOT / "figures" / f"n{tag}.png", dpi=300)
