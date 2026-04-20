"""
Create Table 3: Descriptives of Spouses by Deportation Order and Migration.
"""


import pandas as pd

from deportations_fallout.results.tabulate import table_by_two
from deportations_fallout.config import PATHS

paths = PATHS


# Defaults
COLS_TO_MAX = [
    "female",
    "age",
    "children",
    "D",
    "convicted",
    "charged",
    "incarcerated",
    "has_children",
    "mover"
]

COLS_TO_MEAN = ["partic", "fulltime", "wages", "transfers", "assets"]

GROUP_NAMES = {"1.0": "Mover", "0.0": "Stayer", "p": "p-value"}

STAT_NAMES = {"count": "N", "mean": "Mean", "std": "SD"}

VAR_NAMES = {
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


def table_3():
    """
    Create Table 3: Descriptives of Spouses by Deportation Order and Migration.
    """

    # ---- Set up data

    df = pd.read_parquet(paths.temp / "panel.parquet")

    df = df.loc[
        lambda d: d["event_time"].lt(0)  # pre-conviction
        & d["month"].dt.year.between(2000, 2021)  # observation window
        & d["in_country"]  # in Denmark at time
    ].copy()

    df = df.assign(
        has_children=lambda d: d["children"].ge(1)
        )

    # Make variables numeric
    cols = COLS_TO_MAX + COLS_TO_MEAN
    df[cols] = df[cols].astype(float)

    # Collapse to individual level
    df = df.groupby("pnr").agg({
        **{c: "max" for c in COLS_TO_MAX},
        **{c: "mean" for c in COLS_TO_MEAN},
    })

    # Conviction only
    tbl_a = table_by_two(
        df.loc[df["D"].eq(0)].drop(columns=["D"]),
        group="mover",
        group_names=GROUP_NAMES,
        stat_names=STAT_NAMES,
        row_names=VAR_NAMES,
    )

    # Conviction and deportation order
    tbl_b = table_by_two(
        df.loc[df["D"].eq(1)].drop(columns=["D"]),
        group="mover",
        group_names=GROUP_NAMES,
        stat_names=STAT_NAMES,
        row_names=VAR_NAMES,
    )

    tbl = pd.concat(
        {"Conviction only": tbl_a, "Conviction and deportation order": tbl_b}, axis=1
    )

    return tbl
