"""
Creates figures and tables for Appendix A.
Content: Figure A1 and Table A1
"""

from pathlib import Path

import numpy as np
import pandas as pd

from tools.paths import init
from tools.formatting import setup_pyplot
from tools.plots import dyn_did, raw_rates, did_plot
from tools.tables import table_by_two


def run():
    """
    Creates:
    - Table A1: Descriptives of Female Spouses and Female Foreign-Born Population
    - Figure A1: Main Results for Unconditional Prison Sentences Only
    """

    # expose paths (and create output folders if they do not exist)
    paths = init(Path.cwd())

    # setup plotting defaults
    setup_pyplot()

    # load data
    panel = pd.read_parquet(paths.data / "panel.parquet")
    spouses = pd.read_parquet(paths.data / "spouses.parquet")
    population = pd.read_parquet(paths.data / "population.parquet")

    # ---- Table A1: Descriptives of Female Spouses and Female Foreign-Born Population

    vl = [
        "pnr",
        "age",
        "children",
        "born_in_dk",
        "has_children",
        "assets",
        "spouse",
        "grounds",
    ]

    # Female spouses of deportees
    df1 = spouses.loc[lambda d: d["female"].eq(1) & d["D"].eq(1)].assign(
        born_in_dk=lambda d: d["ie_type"].ne(2),
        has_children=lambda d: d["children"].ge(1),
        spouse=1,
    )[vl]

    # Add covariates from panel (12 months prior to conviction)
    vl1 = ["convicted", "charged", "incarcerated"]
    vl2 = ["partic", "fulltime", "wages", "transfers"]

    df1 = df1.merge(
        panel.loc[lambda d: d.event_time.between(-12, -1)]
        .groupby("pnr")
        .agg({**{c: "max" for c in vl1}, **{c: "mean" for c in vl2}}),
        on="pnr",
        how="left",
        validate="1:1",
    )

    # Female immigrants/descendants w. registered spouse or partner (2010)
    df2 = pd.read_parquet(paths.data / "female_foreign_born.parquet")

    df = (
        pd.concat([df1, df2], ignore_index=True)
        .assign(
            danish=lambda d: d["grounds"].eq(0),
            eu=lambda d: d["grounds"].eq(1),
            asylum=lambda d: d["grounds"].eq(2),
            family=lambda d: d["grounds"].eq(4),
        )
        .drop(columns=["grounds"])
    )

    for c in df.columns:
        df[c] = df[c].astype(float)

    # Create table
    group_names = {
        "0.0": "Female foreign-born population and their daughters",
        "1.0": "Conviction and deportation order (women in treatment group)",
        "p": "p-value",
    }
    stat_names = {"count": "N", "mean": "Mean", "std": "SD"}
    row_names = {
        "age": "Age",
        "has_children": "Has children",
        "children": "Number of children",
        "born_in_dk": "Born in Denmark",
        "convicted": "Convicted",
        "charged": "Charged",
        "incarcerated": "Incarcerated",
        "partic": "Av. prior employment rate",
        "fulltime": "Av. percent of fulltime employment",
        "wages": "Av. prior earnings",
        "transfers": "Av. prior transfer income",
        "assets": "Net assets",
        "danish": "Danish",
        "eu": "EU/EEA",
        "asylum": "Asylum",
        "family": "Family reunification",
    }

    # Order variables for table
    df = df[["spouse"] + [c for c in row_names]]

    # Table
    tbl = table_by_two(
        df.sort_values("spouse"),
        group="spouse",
        group_names=group_names,
        stat_names=stat_names,
        row_names=row_names,
    )

    # discretion
    for g in [
        "Female foreign-born population and their daughters",
        "Conviction and deportation order (women in treatment group)",
    ]:

        _n = (g, "N")
        mean = (g, "Mean")
        sd = (g, "SD")

        tbl[_n] = tbl[_n].round(0).astype("Int64")
        breach = tbl[mean].between(0, 1) & (tbl[mean] * tbl[_n] < 3)
        # discretion can be ignored in the large sample do to decimal rounding
        if g != "Female foreign-born population and their daughters":
            tbl.loc[breach, _n] = np.nan
            tbl.loc[breach, mean] = np.nan
            tbl.loc[breach, sd] = np.nan

    # Save
    tbl.to_html(paths.tables / "Table A1.html", escape=False)

    # ---- Figure A1: Main Results for Unconditional Prison Sentences Only

    mask = (
        population.loc[
            lambda d: d["D"].eq(1) | d["prison_sentence"].gt(0), ["pnr", "conviction_date"]
        ]
        .rename(columns={"pnr": "convict"})
        .assign(conviction_date=lambda d: d["conviction_date"].dt.to_period("M"))
        .drop_duplicates()
    )

    df = (
        panel.merge(
            mask, on=["convict", "conviction_date"], how="inner", validate="m:1"
        )
        .reset_index(drop=True)
        .copy()
    )

    # Make "regressable"
    for c in df.select_dtypes(exclude=[pd.PeriodDtype, "datetime64[ns]"]):
        df[c] = df[c].astype(float)

    name = [
        "Employment",
        "Share of fulltime employment",
        "Pre-tax wage in 1.000 DKK",
        "Total income (wages + transfers) in 1.000 DKK",
    ]
    lit = ["A", "B", "C", "D", "E", "F", "G", "H"]

    df["income"] = df["wages"] + df["transfers"]

    for i, y in enumerate(["partic", "fulltime", "wages", "income"], start=1):

        # Estimate
        _, out = dyn_did(df, y=y)

        # Plot and save
        fig_a, ax_a = raw_rates(df, var=y, ylabel=name[i - 1], ylim=None)
        fig_b, _ = did_plot(out, ylim=None)

        ax_a.legend(bbox_to_anchor=(0.5, -0.12), ncol=2)

        fig_a.savefig(paths.figures / f"Figure A1{lit[i*2 - 2]}.png")
        fig_b.savefig(paths.figures / f"Figure A1{lit[i*2 - 1]}.png")


if __name__ == "__main__":
    run()
