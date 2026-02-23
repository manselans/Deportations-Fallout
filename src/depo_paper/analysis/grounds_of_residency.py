"""
Creates figures with migration/attrition results.
Content: Figures 8 and 9.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from depo_paper.config import PATHS
from depo_paper.tools.formatting import setup_pyplot
from depo_paper.tools.plots import dyn_did, raw_rates, did_plot


def run():
    """
    Creates:
    - Figure 8: Migration Patterns by Grounds for Residence
    - Figure 9: Main Results by Grounds for Residence
    """

    # expose paths (and create output folders if they do not exist)
    paths = PATHS

    # setup plotting defaults
    setup_pyplot()

    # load data
    panel = pd.read_parquet(paths.temp / "panel.parquet")

    # ---- Figure 8: Migration Patterns by Grounds for Residence

    grounds_map = {
        0: "Danish",
        1: "EU/EEA",
        2: "Asylum",
        3: "Study/work",
        4: "Family reunified",
        5: "Unspecified",
    }

    df = (
        panel.loc[lambda d: d["grounds"].isin([0, 1, 2, 4])]
        .groupby(["D", "event_time", "grounds"])["in_country"]
        .agg(["mean", "count"])
        .reset_index()
        .assign(
            ooc=lambda d: 1 - d["mean"], grounds=lambda d: d["grounds"].map(grounds_map)
        )
    )

    lit = ["A", "B"]
    colors = ["black", "0.3", "0.6", "0.4"]
    markers = ["o", "s", "^", "v"]
    ls = ["--", "--", "-", "--"]

    for ct in [0, 1]:  # by treatment status

        df_ct = df.loc[lambda d, ct=ct: d.D.eq(ct)].copy()

        fig, ax = plt.subplots(figsize=(8, 6))

        ax.axvline(0, color="red", linestyle="--")
        ax.axvline(-7, color="0.4", linestyle="--")

        for i, (g, d) in enumerate(df_ct.groupby("grounds")):

            # discretion
            mask = d["ooc"] * d["count"] >= 3

            ax.plot(
                d.loc[mask, "event_time"],
                d.loc[mask, "ooc"],
                marker=markers[i],
                color=colors[i],
                linestyle=ls[i],
                label=g
            )

        ax.set_ylim((0, 0.5))
        ax.set_ylabel("Share out of country")
        ax.tick_params(axis="y", labelrotation=90)
        ax.set_xticks(np.arange(-18, 19, 3))
        ax.xaxis.set_major_formatter("{x:.0f}")
        ax.legend(bbox_to_anchor=(0.5, -0.12), ncol=4)

        fig.savefig(paths.figures / f"Figure 8{lit[ct]}.png")

    # ---- Figure 9: Main Results by Grounds for Residence

    df = (
        panel.loc[lambda d: d["grounds"].isin([0, 1, 2, 4])]
        .dropna(subset=["partic"])
        .reset_index(drop=True)
        .copy()
    )

    # Make "regressable"
    for c in df.select_dtypes(exclude=[pd.PeriodDtype, "datetime64[ns]"]):
        df[c] = df[c].astype(float)

    df = df.assign(grounds=lambda d: d["grounds"].map(grounds_map))

    litra = ["A", "B", "C", "D", "E", "F", "G", "H"]
    for i, (g, d) in enumerate(df.groupby("grounds"), start=1):

        # Estimate
        _, out = dyn_did(d, y="partic")

        # Plot and save
        fig_a, ax_a = raw_rates(d, var="partic", ylabel="Employment", ylim=None)
        fig_b, _ = did_plot(out, ylim=None)

        ax_a.legend(bbox_to_anchor=(0.5, -0.12), ncol=2)

        fig_a.savefig(paths.figures / f"Figure 9{litra[i*2 - 2]}.png")
        fig_b.savefig(paths.figures / f"Figure 9{litra[i*2 - 1]}.png")


if __name__ == "__main__":
    run()
