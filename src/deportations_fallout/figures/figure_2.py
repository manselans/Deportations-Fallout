"""
Create Figure 2: Spouses' Employment: Trends and Estimation Results
"""


import pandas as pd
import matplotlib.pyplot as plt

from deportations_fallout.results.dynamic_did import raw_rates, dyn_did, coef_plot
from deportations_fallout.config import PATHS, setup_matplotlib, scale_figsize

paths = PATHS
setup_matplotlib()



def figure_2():
    """
    Create Figure 2: Spouses' Employment: Trends and Estimation Results
    """

    df = pd.read_parquet(paths.temp / "panel.parquet")

    df = df[["pnr", "D", "event_time", "partic"]].copy()

    fig, ax = plt.subplots(1, 2, figsize=scale_figsize(nrows=1, ncols=2))
    axes = ax.flatten()

    # Panel A: Raw rates
    raw_rates(df, var="partic", ylabel="Employment", ax=axes[0])
    #axes[0].set_title("(a) Raw means", fontstyle="italic")
    axes[0].legend(bbox_to_anchor=(0.5, -0.12), ncol=2)

    # Panel B: Regression results
    _, out = dyn_did(df, y="partic")
    coef_plot(out, ylim=(-0.1, 0.06), ax=axes[1])
    #axes[1].set_title("(b) Regression results: Dynamic DiD", fontstyle="italic")

    return fig, ax
