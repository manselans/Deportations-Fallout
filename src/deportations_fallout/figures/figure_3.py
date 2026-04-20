"""
Create Figure 3: Other Labour Market Outcomes: Trends and Estimation Results.
"""

# import string
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from deportations_fallout.results.dynamic_did import raw_rates, dyn_did, coef_plot
from deportations_fallout.config import PATHS, setup_matplotlib, scale_figsize

paths = PATHS
setup_matplotlib()


def figure_3():
    """
    Create Figure 3: Other Labour Market Outcomes: Trends and Estimation Results.
    """

    df = pd.read_parquet(paths.temp/"panel.parquet")

    vl = ["pnr", "D", "event_time", "fulltime", "wages", "transfers"]
    df = df[vl].astype(float).copy()
    df["income"] = df["wages"] + df["transfers"]

    names = [
        "Share of fulltime employment",
        "Pre-tax wage in 1.000 DKK",
        "Total income (wages + transfers) in 1.000 DKK",
    ]

    row_titles = [
        "Share of fulltime employment",
        "Earnings",
        "Total income (incl. transfers)",
    ]

    fig = plt.figure(figsize=scale_figsize(nrows=3, ncols=2))
    subfigs = fig.subfigures(nrows=3, ncols=1)

    ax = []
    for i, subfig in enumerate(subfigs):
        subfig.suptitle(row_titles[i], fontsize=18)
        axs = subfig.subplots(1, 2)
        ax.append(axs)
    ax = np.array(ax)

    for i, y in enumerate(["fulltime", "wages", "income"]):

        # Left panel: Raw rates
        #let = string.ascii_lowercase[i * 2]
        raw_rates(df, var=y, ylabel=names[i - 1], ylim=None, ax=ax[i, 0])
        ax[i, 0].legend(bbox_to_anchor=(0.5, -0.12), ncol=2)
        #ax[i, 0].set_title(f"({let}) Raw means", fontstyle="italic")

        # Right panel: Regression results
        #let = string.ascii_lowercase[i * 2 + 1]
        _, out = dyn_did(df, y=y)
        coef_plot(out, ylim=None, ax=ax[i, 1])
        # ax[i, 1].set_title(
        #     f"({let}) Regression results: Dynamic DiD", fontstyle="italic"
        # )

    return fig, ax
