"""
Create Figure B1: Deportation Risk by Residence Seniority.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from deportations_fallout.config import PATHS, setup_matplotlib

paths = PATHS
setup_matplotlib()


def figure_b1():
    """
    Create Figure B1: Deportation Risk by Residence Seniority.
    """

    # ---- Set up data

    panel_df = pd.read_parquet(paths.temp/"panel.parquet")
    convicts_df = pd.read_parquet(paths.temp/"population.parquet")

    # Conviction cannot be exempt from seniority rule
    vl = ["pnr", "convict", "D", "seniority"]
    ids = set(convicts_df.loc[lambda d: d["exception"].eq(False), "pnr"])
    df = panel_df.loc[lambda d: d["convict"].isin(ids), vl].copy()

    # Keep only seniority 1 year around discontinuity (5 years)
    df["seniority"] = (
        df["seniority"] / 30 - 12 * 5 # seniority in months relative to 5 years
    ).apply(lambda v: np.floor(v) if v < 0 else np.ceil(v))

    df = df.loc[
        lambda d: d["seniority"].between(-12, 12) & d["seniority"].ne(0)
    ].dropna().drop_duplicates()

    # ---- Linear fits and discretion

    df_t = (
        df.assign(  # bin seniority
            seniority=lambda d: np.where(
                d.seniority < 0, (d.seniority // 2) * 2, ((d.seniority + 1) // 2) * 2
            )
        )
        .groupby("seniority")["D"]
        .agg(["mean", "count"])
        .reset_index()
        .loc[lambda d: d["count"] >= 3]  # discretion
    )

    left = df[df["seniority"] < 0]
    right = df[df["seniority"] > 0]

    b_left = np.polyfit(left["seniority"], left["D"], 1)
    b_right = np.polyfit(right["seniority"], right["D"], 1)

    x_left = [-12, -1]
    y_left = np.polyval(b_left, x_left)

    x_right = [1, 12]
    y_right = np.polyval(b_right, x_right)

    # ---- Plot

    fig, ax = plt.subplots()

    ax.axvline(0, color="red", ls="--")
    ax.scatter(df_t["seniority"], df_t["mean"], color=".4")
    ax.plot(x_left, y_left, lw=2, color=".6")
    ax.plot(x_right, y_right, lw=2, color=".6")

    # Edits
    ax.set_xticks(np.arange(-12, 13, 2))
    ax.tick_params(axis="y", rotation=90)
    ax.set_ylim(-0.01, 0.25)

    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda x, _: (
                "0"
                if abs(x) < 1e-12
                else f"{x:.2f}".replace("0.", ".").replace("-0.", "-.")
            )
        )
    )

    ax.set_ylabel("Share with deportation order")
    ax.set_xlabel("Conviction month relative to 5 years of seniority")

    return fig, ax
