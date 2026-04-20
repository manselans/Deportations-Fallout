"""
Create Figure C1: Residency Seniority by Deportation Order.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import statsmodels.api as sm

from deportations_fallout.config import PATHS, setup_matplotlib, scale_figsize

paths = PATHS
setup_matplotlib()

def figure_c1():
    """
    Create Figure C1: Residency Seniority by Deportation Order.
    """

    # ---- Set up data

    df = pd.read_parquet(paths.temp/"panel.parquet")

    df = df[["pnr", "D", "seniority"]].dropna().copy()
    df["seniority"] = np.floor(df["seniority"] / 365)
    df = df.drop_duplicates()

    # Discretion
    df_c = df.loc[df["D"] == 0].copy()
    df_t = df.loc[df["D"] == 1].copy()

    counts = df_c.groupby(["seniority"]).size()
    df_c = df_c[df_c["seniority"].isin(counts.index[counts >= 5])]

    counts = df_t.groupby(["seniority"]).size()
    df_t = df_t[df_t["seniority"].isin(counts.index[counts >= 5])]

    # ---- Plot as histogram

    fig, ax = plt.subplots(1, 2, figsize=scale_figsize(nrows=1, ncols=2))
    axes = ax.flatten()

    axes[0].hist(  # control group
        df_c["seniority"],
        bins=25,
        density=True,
        alpha=0.25,
        color="0",
        label="Conviction only",
    )

    axes[0].hist(  # treatment group
        df_t["seniority"],
        bins=25,
        density=True,
        alpha=0.25,
        color="red",
        label="Conviction and deportation order",
    )

    # axes[0].set_title("(a) Histogram", fontstyle="italic")
    axes[0].set_xlabel("Years of seniority")
    axes[0].set_ylabel("Density")
    axes[0].legend(bbox_to_anchor=(0.5, -0.12), ncol=2)

    axes[0].tick_params(axis="y", rotation=90)
    axes[0].yaxis.set_major_formatter(
        FuncFormatter(
            lambda x, _: (
                "0"
                if abs(x) < 1e-12
                else f"{x:.2f}".replace("0.", ".").replace("-0.", "-.")
            )
        )
    )

    # ---- Plot as QQ-plot

    sm.qqplot_2samples(df_c["seniority"], df_t["seniority"], line="45", ax=axes[1])

    axes[1].set_ylabel("Quantiles of seniority, Conviction only")
    axes[1].set_xlabel("Quantiles of seniority, Conviction and deportation order")
    axes[1].tick_params(axis="y", rotation=90)
    # axes[1].set_title("(b) QQ-plot", fontstyle="italic")

    axes[1].lines[0].set_markerfacecolor("black")
    axes[1].lines[0].set_markeredgecolor("black")
    axes[1].lines[1].set_zorder(1)

    return fig, ax
