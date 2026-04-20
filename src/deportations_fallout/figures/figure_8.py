"""
Create Figure 8: Migration Patterns by Grounds for Residence.
"""

import string

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from deportations_fallout.config import PATHS, setup_matplotlib, scale_figsize

paths = PATHS
setup_matplotlib()


def figure_8():
    """
    Create Figure 8: Migration Patterns by Grounds for Residence.
    """

    # ---- Set up data

    df = pd.read_parquet(paths.temp/"panel.parquet")

    df = df.loc[
        lambda d: d["grounds"].isin([0, 1, 2, 4]),
        ["D", "event_time", "grounds", "in_country"],
    ].copy()

    grounds_names = {
        0: "Danish",
        1: "EU/EEA",
        2: "Asylum",
        3: "Study/work",
        4: "Family reunified",
        5: "Unspecified",
    }

    # Share in country over time by deportation order and grounds
    df = (
        df.groupby(["D", "event_time", "grounds"])["in_country"]
        .agg(["mean", "count"])
        .reset_index()
    )

    df["out_of_country"] = 1 - df["mean"]
    df["grounds"] = df["grounds"].map(grounds_names)

    # ---- Plot

    fig, ax = plt.subplots(1, 2, figsize=scale_figsize(nrows=1, ncols=2), sharey=True)
    axes = ax.flatten()

    colors = ["black", "0.3", "0.6", "0.4"]
    markers = ["o", "s", "^", "v"]
    ls = ["--", "--", "-", "--"]

    group_names = {0: "Conviction only", 1: "Conviction and deportation order"}

    # By treatment status
    for j, t in enumerate([0, 1]):

        df_plot = df.loc[df["D"] == t].copy()

        # For each type of residency grounds
        for i, (g, d) in enumerate(df_plot.groupby("grounds")):

            # Discretion
            too_few = d["out_of_country"] * d["count"] < 3
            too_many = (1 - d["out_of_country"]) * d["count"] < 3
            d = d.mask(too_few | too_many)

            d = d[["event_time", "out_of_country"]].dropna()

            # Plot
            axes[j].plot(
                d["event_time"],
                d["out_of_country"],
                marker=markers[i],
                color=colors[i],
                linestyle=ls[i],
                label=g,
            )

        axes[j].axvline(0, color="red", linestyle="--")
        axes[j].axvline(-7, color="0.4", linestyle="--")

        # Edits
        axes[j].set_ylim((0, 0.5))
        axes[j].tick_params(axis="y", labelrotation=90)
        axes[j].set_xticks(np.arange(-18, 19, 3))
        axes[j].xaxis.set_major_formatter("{x:.0f}")
        axes[j].legend(bbox_to_anchor=(0.5, -0.12), ncol=4)

        axes[j].set_title(
            f"({string.ascii_lowercase[j]}) {group_names[t]}", fontstyle="italic"
        )

    axes[0].set_ylabel("Share out of country")

    return fig, ax
