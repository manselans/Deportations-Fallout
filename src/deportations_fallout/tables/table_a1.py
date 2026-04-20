"""
Create Table A1: Descriptives of Female Spouses and Female Foreign-Born Population
"""

import pandas as pd

from deportations_fallout.results.tabulate import table_by_two
from deportations_fallout.config import PATHS

paths = PATHS


# Defaults
GROUP_NAMES = {
    "0.0": "Female foreign-born population and their daughters",
    "1.0": "Conviction and deportation order (women in treatment group)",
    "p": "p-value",
}

STAT_NAMES = {"count": "N", "mean": "Mean", "std": "SD"}

VAR_NAMES = {
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


def table_a1():
    """
    Create Table A1: Descriptives of Female Spouses and Female Foreign-Born Population
    """

    spouse_df = pd.read_parquet(paths.temp/"spouses.parquet")
    foreign_df = pd.read_parquet(paths.temp/"female_foreignborn.parquet")
    panel_df = pd.read_parquet(paths.temp/"panel.parquet")

    vl = ["pnr", "age", "children", "born_in_dk", "has_children", "assets", "grounds"]

    # ---- Female spouses of deportees

    spouse_df = spouse_df.loc[lambda d: d["female"].eq(1) & d["D"].eq(1)].copy()

    spouse_df = spouse_df.assign(
        born_in_dk=lambda d: d["ie_type"].ne(2),
        has_children=lambda d: d["children"].ge(1),
    )[vl]

    # Add covariates from panel (12 months prior to conviction)
    to_max = ["convicted", "charged", "incarcerated"]
    to_mean = ["partic", "fulltime", "wages", "transfers"]

    panel_df = panel_df.loc[lambda d: d["event_time"].between(-12, -1)].copy()
    panel_df = (
        panel_df.groupby("pnr")
        .agg({**{c: "max" for c in to_max}, **{c: "mean" for c in to_mean}})
        .reset_index()
    )

    spouse_df = spouse_df.merge(panel_df, on="pnr", how="left", validate="1:1")
    spouse_df["spouse"] = 1

    # ---- Female immigrants/descendants w. registered spouse or partner (2010)

    foreign_df = foreign_df.copy()

    # ---- Combine

    df = pd.concat([spouse_df, foreign_df], ignore_index=True)
    df = df.assign(
        danish=lambda d: d["grounds"].eq(0),
        eu=lambda d: d["grounds"].eq(1),
        asylum=lambda d: d["grounds"].eq(2),
        family=lambda d: d["grounds"].eq(4),
    ).drop(columns=["grounds"])
    df = df.astype(float)

    # ---- Tabulate

    # Order variables for table
    df = df[["spouse"] + [c for c in VAR_NAMES]]

    tbl = table_by_two(
        df.sort_values("spouse"),
        group="spouse",
        group_names=GROUP_NAMES,
        stat_names=STAT_NAMES,
        row_names=VAR_NAMES,
    )

    return tbl
