"""
Create Figure 1: Deportation Orders Over Time
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration
from deportations_fallout.config import PATHS, setup_matplotlib

paths = PATHS
setup_matplotlib()


# Default
GROUP_NAMES: dict = {0: "Conviction only", 1: "Conviction and deportation order"}
UNIQUE: bool = True # Count unique people?


def figure_1():
    """
    Create Figure 1: Deportation Orders Over Time
    """

    df = pd.read_parquet(paths.temp / "population.parquet")

    df = df.loc[lambda d: d.year.le(2021), ["pnr", "year", "D"]].copy()

    if UNIQUE:
        y = df.groupby(["year", "D"])["pnr"].nunique().unstack(fill_value=0)
    else:
        y = df.groupby(["year", "D"])["pnr"].count().unstack(fill_value=0)

    # Add share
    y["share_1"] = y[1] / y.sum(axis=1)

    # Rename groups
    y = y.rename(columns=GROUP_NAMES)
    y = y.sort_index()

    # Plot
    fig, ax = plt.subplots()
    gnames = list(GROUP_NAMES.values())

    # Stacked bars
    ax.bar(y.index, y[gnames[0]], label=gnames[0], color=".5")
    ax.bar(y.index, y[gnames[1]], label=gnames[1], color=".7", bottom=y[gnames[0]])

    # Line on second y-axis
    ax2 = ax.twinx()
    ax2.plot(
        y.index,
        y["share_1"] * 100,
        linewidth=1,
        color="0",
        label="Pct. of convictions that include deportation order",
    )

    # Edit axes
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of observations")
    ax2.set_ylabel("Percent")

    ax.set_xticks(np.arange(y.index.min(), y.index.max() + 1, 5))
    ax.set_yticks(np.arange(0, 1001, 200))

    ax.tick_params(axis="y", rotation=90)
    ax2.tick_params(axis="y", rotation=90)

    # Legend
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    legend_handles = handles1 + handles2
    legend_labels = labels1 + labels2
    ax.legend(legend_handles, legend_labels, bbox_to_anchor=(0.5, -0.12), ncol=1)

    return fig, ax
