"""
Create Table 1: Descriptives of Spouses by Deportation Order.
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
]

COLS_TO_MEAN = ["partic", "fulltime", "wages", "transfers", "assets"]

GROUP_NAMES = {
    "0.0": "Conviction only",
    "1.0": "Conviction and deportation order",
    "p": "p-value",
}

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

def table_1():
    """
    Create Table 1: Descriptives of Spouses by Deportation Order
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
    ).copy()

    # Make variables numeric
    cols = COLS_TO_MAX + COLS_TO_MEAN
    df[cols] = df[cols].astype(float)

    # Collapse to individual level
    df = df.groupby("pnr").agg({
        **{c: "max" for c in COLS_TO_MAX},
        **{c: "mean" for c in COLS_TO_MEAN},
    })

    # ---- Tabulate

    tbl = table_by_two(
        df,
        group="D",
        group_names=GROUP_NAMES,
        stat_names=STAT_NAMES,
        row_names=VAR_NAMES,
    )

    return tbl
