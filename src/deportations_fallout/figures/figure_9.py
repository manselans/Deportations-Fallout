"""
Create Figure 9: Main Results by Grounds for Residence.
"""


# import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from deportations_fallout.results.dynamic_did import raw_rates, dyn_did, coef_plot
from deportations_fallout.config import PATHS, setup_matplotlib, scale_figsize

paths = PATHS
setup_matplotlib()

# Defaults
USE_GROUNDS: iter = [2, 0, 1, 4] # determines order of appearance too

def figure_9():
    """
    Create Figure 9: Main Results by Grounds for Residence.
    """

    # ---- Set up data

    df = pd.read_parquet(paths.temp/"panel.parquet")

    df = df.loc[
        lambda d: d["grounds"].isin(USE_GROUNDS),
        ["pnr", "D", "event_time", "grounds", "partic"]
    ].dropna().copy()

    grounds_map = {
        0: "Danish",
        1: "EU/EEA",
        2: "Asylum",
        3: "Study/work",
        4: "Family reunified",
        5: "Unspecified",
    }

    fig = plt.figure(figsize=scale_figsize(nrows=len(USE_GROUNDS), ncols=2))
    subfigs = fig.subfigures(nrows=len(USE_GROUNDS), ncols=1)

    ax = []
    for i, subfig in enumerate(subfigs):
        subfig.suptitle(grounds_map[USE_GROUNDS[i]], fontsize=18)
        axs = subfig.subplots(1, 2)
        ax.append(axs)
    ax = np.array(ax)

    for i, g in enumerate(USE_GROUNDS):

        df_plot = df.loc[lambda d, g=g: d["grounds"] == g]

        # Left panel: Raw rates
        # let = string.ascii_lowercase[i * 2]
        raw_rates(df_plot, var="partic", ylabel="Employment", ylim=None, ax=ax[i, 0])
        ax[i, 0].legend(bbox_to_anchor=(0.5, -0.12), ncol=2)
        # ax[i, 0].set_title(f"({let}) Raw means", fontstyle="italic")

        # Right panel: Regression results
        # let = string.ascii_lowercase[i * 2 + 1]
        _, out = dyn_did(df_plot, y="partic")
        coef_plot(out, ylim=None, ax=ax[i, 1])
        # ax[i, 1].set_title(
        #     f"({let}) Regression results: Dynamic DiD", fontstyle="italic"
        # )

    return fig, ax
