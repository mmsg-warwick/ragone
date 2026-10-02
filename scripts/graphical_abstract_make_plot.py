import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colormaps

from ragone import ROOT

# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

# ── Load data
with open(ROOT / "data" / "graphical_abstract_rpt_data.pkl", "rb") as file:
    rpt_data = pickle.load(file)

# ── Load CSV data with pandas
ragone_data = pd.read_csv(ROOT / "data" / "graphical_abstract_ageing_metrics.csv")

# ── Power slice positions and style ───────────────────────────────────────────
SLICE_POWERS = [float(k) for k in rpt_data if k != "Cycle number"]
SLICE_LABELS = [f"$P_{{{i}}}$" for i in range(1, len(SLICE_POWERS) + 1)]
# Colorblind-safe palette (Bang Wong)
SLICE_COLORS = ["#4477AA", "#66CCEE", "#CCBB44", "#EE6677"]

# ── Energy fade arrow (vertical, on the low-power plateau) ────────────────────
X_EF = np.sqrt(SLICE_POWERS[0] * SLICE_POWERS[1])  # power position of arrow (W)
Y_EF_TOP = ragone_data["Reference energy [W.h]"].iloc[0]
Y_EF_BOT = ragone_data["Reference energy [W.h]"].iloc[-1]

# ── Power fade arrow (horizontal, near the high-power knee) ──────────────────
Y_PF = 8.0  # fixed energy level for arrow (Wh)
X_PF_R = 90.0  # power at Cycle 1 knee (W)            ← tweak to match your data
X_PF_L = 25.0  # power at Cycle 1000 knee (W)         ← tweak to match your data

# ── Shared arrow/label styles ─────────────────────────────────────────────────
_ARROW_KW = {"arrowstyle": "->", "color": "black", "lw": 1.5}
_LABEL_KW = {
    "fontsize": 5,
    "ha": "center",
    "va": "center",
    "color": "black",
    "bbox": {"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none", "alpha": 0.85},
}
cmap = colormaps["plasma"]


# ══════════════════════════════════════════════════════════════════════════════
# LEFT PANEL — Ragone plot with fade arrows and slice lines
# ══════════════════════════════════════════════════════════════════════════════


def _gaussian(x, E0, P0, n):
    return E0 * np.exp(-((x / P0) ** n))


def plot_ragone(ax):
    """Plot Ragone curves on ax. Replace this block with your custom function."""

    colors = cmap(np.linspace(0, 0.9, len(ragone_data)))
    x_data = np.logspace(np.log10(0.5), np.log10(100), 100)

    for (_, row), color in zip(ragone_data.iterrows(), colors):
        y_data = _gaussian(
            x_data, row["Reference energy [W.h]"], row["Reference power [W]"], row["n"]
        )
        ax.plot(x_data, y_data, color=color)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Power", fontsize=6)
    ax.set_ylabel("Energy", fontsize=6)
    ax.set_xlim(0.5, 100)
    ax.set_ylim(2, 25)

    ax.text(
        0.55, Y_EF_TOP * 1.05, "Fresh cell", fontsize=4, color=colors[0], va="bottom"
    )
    ax.text(0.55, Y_EF_BOT * 0.95, "Aged cell", fontsize=4, color=colors[-1], va="top")


def annotate_ragone(ax):
    """Add energy/power fade arrows and vertical slice lines to the Ragone axes."""

    # 1. Vertical slice lines + labels
    for p, label, color in zip(SLICE_POWERS, SLICE_LABELS, SLICE_COLORS):
        ax.axvline(p, color=color, lw=1.2, ls="--", zorder=5, alpha=0.9)
        ax.annotate(
            label,
            xy=(p, 0.12),
            xycoords=("data", "axes fraction"),
            xytext=(-2, -4),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=5,
            fontweight="bold",
            color=color,
            bbox={
                "boxstyle": "round,pad=0.15",
                "fc": "white",
                "ec": "none",
                "alpha": 0.75,
            },
        )

    # 2. Energy fade — vertical arrow
    ax.annotate(
        "",
        xy=(X_EF, Y_EF_BOT * 0.85),
        xytext=(X_EF, Y_EF_TOP * 1.10),
        arrowprops=_ARROW_KW,
    )
    ax.text(
        X_EF,
        (Y_EF_BOT) * 0.7,  # geometric mean = midpoint on log scale
        "Energy\nfade",
        **_LABEL_KW,
        zorder=10,
    )

    # 3. Power fade — horizontal arrow
    ax.annotate("", xy=(X_PF_L, Y_PF), xytext=(X_PF_R, Y_PF), arrowprops=_ARROW_KW)
    ax.text(
        (X_PF_R * X_PF_L) ** 0.5,
        Y_PF * 1.4,
        "Power fade",
        **_LABEL_KW,
    )


# ══════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — Energy at fixed power vs. cycle number
# ══════════════════════════════════════════════════════════════════════════════


def plot_right_panel(ax):
    """Plot energy-at-fixed-power vs. cycle number for each slice."""

    cycles = rpt_data["Cycle number"]
    for power, label, color in zip(SLICE_POWERS, SLICE_LABELS, SLICE_COLORS):
        ax.plot(cycles[1:], rpt_data[power][1:], color=color, lw=1.2, label=label)

    ax.set_xlabel("Cycle number", fontsize=6)
    ax.set_ylabel("Energy", fontsize=6)
    ax.set_xlim(cycles[1], cycles[-1])
    ax.set_ylim(0, None)
    ax.legend(loc="lower left", framealpha=0.85, fontsize=5)


# ══════════════════════════════════════════════════════════════════════════════
# ASSEMBLE FIGURE
# ══════════════════════════════════════════════════════════════════════════════

fig, (ax_left, ax_right) = plt.subplots(
    1,
    2,
    figsize=(8 / 2.54, 4 / 2.54),
    gridspec_kw={"width_ratios": [3, 2]},
    constrained_layout=True,
)

# ── Left panel ────────────────────────────────────────────────────────────────
plot_ragone(ax_left)  # sets up axes + plots curves
annotate_ragone(ax_left)  # adds arrows and slice lines

# ── Right panel ───────────────────────────────────────────────────────────────
plot_right_panel(ax_right)

# ── Remove ticks ────────────────────────────────────────────────────────────────
for ax in [ax_left, ax_right]:
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_minor_locator(plt.NullLocator())

# ── Panel titles ────────────────────────────────────────────────────
ax_left.set_title("(a) Ragone plot", fontsize=6, fontweight="bold")
ax_right.set_title("(b) Reference Performance Tests", fontsize=6, fontweight="bold")

# ── Export ────────────────────────────────────────────────────────────────────
fig.savefig(
    ROOT / "figures" / "graphical_abstract.tiff",
    dpi=600,  # increase to 600 if your journal requires it for line art
    format="tiff",
    bbox_inches="tight",
    pil_kwargs={"compression": "tiff_lzw"},  # lossless, keeps file size down
)

fig.savefig(
    ROOT / "figures" / "graphical_abstract.png",
    dpi=600,  # increase to 600 if your journal requires it for line art
    format="png",
    bbox_inches="tight",
)

fig.savefig(
    ROOT / "figures" / "graphical_abstract.svg",
    format="svg",
    bbox_inches="tight",
)

fig.savefig(
    ROOT / "figures" / "graphical_abstract.eps",
    format="eps",
    bbox_inches="tight",
)
