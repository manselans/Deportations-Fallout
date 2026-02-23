"""
Creates tables and figures for the descriptive part of the analysis.
Content: Figure 1 and Tables 1 and 3.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from depo_paper.config import PATHS
from depo_paper.tools.tables import table_by_two
from depo_paper.tools.formatting import setup_pyplot


def run():
    """
    Creates:
    - Figure 1: Deportation Orders Over Time
    - Table 1: Descriptives of spouses by Deportation Order
    - Table 3: Descriptives of Spouses by Deportation Order and Migration
    """

    # expose paths (and create output folders if they do not exist)
    paths = PATHS

    # setup plotting defaults
    setup_pyplot()

    # ---- Load data

    population = pd.read_parquet(paths.temp / "population.parquet")
    panel = pd.read_parquet(paths.temp / "panel.parquet")

    # ---- Figure 1: Deportation orders over time

    df = (
        population
        # 2000-2021
        .assign(year=lambda d: d["conviction_date"].dt.year)
        .loc[lambda d: d["year"].le(2021), ["pnr", "year", "D"]]
        # Collapse by year and deportation status
        .groupby(["year", "D"])["pnr"]
        .nunique()
        .unstack(fill_value=0)
        .rename(columns={0: "Conviction only", 1: "Conviction and deportation order"})
        .sort_index()
        .assign(
            share_deported=lambda d: d["Conviction and deportation order"]
            / d.sum(axis=1)
        )
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    gnames = ("Conviction only", "Conviction and deportation order")

    # Stacked bars
    ax.bar(df.index, df[gnames[0]], label=gnames[0], color=".5")
    ax.bar(df.index, df[gnames[1]], label=gnames[1], color=".7", bottom=df[gnames[0]])

    # Line on second y-axis
    ax2 = ax.twinx()
    ax2.plot(
        df.index,
        df["share_deported"] * 100,
        linewidth=1,
        color="0",
        label="Pct. of convictions that include deportation order",
    )

    # Edit axes
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of observations")
    ax2.set_ylabel("Percent")

    ax.set_xticks(np.arange(df.index.min(), df.index.max() + 1, 5))
    ax.set_yticks(np.arange(0, 1001, 200))

    ax.tick_params(axis="y", rotation=90)
    ax2.tick_params(axis="y", rotation=90)

    # Legend
    handles1, labels1 = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    legend_handles = handles1 + handles2
    legend_labels = labels1 + labels2
    ax.legend(legend_handles, legend_labels, bbox_to_anchor=(0.5, -0.12), ncol=1)

    fig.savefig(paths.figures / "Figure 1.png")

    # ---- Table 1: Descriptives of spouses by deportation order

    # Setup data
    ext = ["female", "age", "children", "D", "convicted", "charged", "incarcerated"]
    ins = ["partic", "fulltime", "wages", "transfers", "assets"]
    cols = ext + ins

    for c in cols:  # Make variables numeric
        panel[c] = panel[c].astype(float)

    df = (
        panel.loc[
            lambda d: d["event_time"].lt(0)  # pre-conviction
            & d["month"].dt.year.lt(2022)  # Observation window
            & d["in_country"]  # In Denmark at time
        ]
        .assign(
            has_children=lambda d: d["children"].ge(1).where(d["children"].notna()),
        )
        # Collapse to individual level
        .groupby("pnr", as_index=False)
        .agg(
            {
                **{col: "max" for col in ext + ["has_children", "mover"]},
                **{col: "mean" for col in ins},
            }
        )
    )

    # Set up table
    group_names = {
        "0.0": "Conviction only",
        "1.0": "Conviction and deportation order",
        "p": "p-value",
    }
    stat_names = {"count": "N", "mean": "Mean", "std": "SD"}
    row_names = {
        "age": "Age",
        "female": "Female",
        "children": "Number of children",
        "has_children": "Has children",
        "fulltime": "Av. percent of fulltime employment",
        "wages": "Av. prior earnings",
        "transfers": "Av. prior transfer income",
        "partic": "Av. prior employment rate",
        "assets": "Net assets",
        "convicted": "Convicted",
        "charged": "Charged",
        "incarcerated": "Incarcerated",
    }

    tbl = table_by_two(
        df.drop(columns=["pnr", "mover"]),
        group="D",
        group_names=group_names,
        stat_names=stat_names,
        row_names=row_names,
    )

    # Discretion
    for g in ["Conviction only", "Conviction and deportation order"]:

        _n = (g, "N")
        mean = (g, "Mean")
        sd = (g, "SD")

        tbl[_n] = tbl[_n].round(0).astype("Int64")
        breach = tbl[mean].between(0, 1) & (tbl[mean] * tbl[_n] < 3)
        tbl.loc[breach, _n] = np.nan
        tbl.loc[breach, mean] = np.nan
        tbl.loc[breach, sd] = np.nan

    # Save
    tbl.to_html(paths.tables / "Table 1.html", escape=False)

    # ---- Table 3: Descriptives of Spouses by Deportation Order and Migration

    # Conviction only
    tbl_a = table_by_two(
        df.loc[df["D"].eq(0)].drop(columns=["pnr", "D"]),
        group="mover",
        group_names={"True": "Mover", "False": "Stayer", "p": "p-value"},
        stat_names=stat_names,
        row_names=row_names,
    )

    # Conviction and deportation order
    tbl_b = table_by_two(
        df.loc[df["D"].eq(1)].drop(columns=["pnr", "D"]),
        group="mover",
        group_names={"True": "Mover", "False": "Stayer", "p": "p-value"},
        stat_names=stat_names,
        row_names=row_names,
    )

    tbl = pd.concat(
        {"Conviction only": tbl_a, "Conviction and deportation order": tbl_b}, axis=1
    )

    # Discretion
    for g in ["Conviction only", "Conviction and deportation order"]:
        for m in ["Mover", "Stayer"]:

            _n = (g, m, "N")
            mean = (g, m, "Mean")
            sd = (g, m, "SD")

            tbl[_n] = tbl[_n].round(0).astype("Int64")
            breach = tbl[mean].between(0, 1) & (tbl[mean] * tbl[_n] < 3)
            tbl.loc[breach, _n] = np.nan
            tbl.loc[breach, mean] = np.nan
            tbl.loc[breach, sd] = np.nan

    # Save
    tbl.to_html(paths.tables / "Table 3.html", escape=False)


if __name__ == "__main__":
    run()
