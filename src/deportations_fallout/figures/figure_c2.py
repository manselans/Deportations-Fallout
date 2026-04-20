"""
Create Figure C2: Types of Crimes by Deportation Order.
"""


import pandas as pd
import matplotlib.patches as mpatches

from deportations_fallout.utils.stata_io import gather
from deportations_fallout.etl.crime_categories import crime_type
from deportations_fallout.config import PATHS, setup_matplotlib 

paths = PATHS
setup_matplotlib()


def figure_c2():
    """
    Create Figure C2: Types of Crimes by Deportation Order.
    """

    df = pd.read_parquet(paths.temp/"population.parquet")
    ids = set(df["pnr"])

    # ---- Set up data

    # Gather conviction records
    kraf = gather(
        paths.data.dst_raw,
        names=range(1980, 2022),
        file_pattern="kraf{name}.dta",
        columns=["pnr", "afg_afgoedto", "afg_ger7"],
        filters={"pnr": ids},
        concatenate=True,
    )
    kraf = kraf.rename(columns={"afg_afgoedto": "conviction_date"})

    # Determine most severe crime of the day
    crime_rank = {
        "unknown": 0,
        "other": 1,
        "property": 2,
        "narcotics": 3,
        "violence": 4,
        "sex": 5,
    }

    kraf["crime_rank"] = pd.Series(crime_type(kraf["afg_ger7"]), index=kraf.index).map(
        crime_rank
    )

    kraf = (
        kraf.sort_values(["pnr", "conviction_date", "crime_rank"])
        .drop_duplicates(subset=["pnr", "conviction_date"], keep="last")
        .drop(columns=["crime_rank"])
    )

    # Add most severe crime code from day of conviction
    df = df.merge(kraf, on=["pnr", "conviction_date"], how="left", validate="1:1")
    df["ctype"] = crime_type(df.afg_ger7)

    crime_names = {
        "other": "Other",
        "property": "Property",
        "narcotics": "Narcotics",
        "violence": "Violence",
        "sex": "Sexual",
    }

    df["ctype"] = df["ctype"].map(crime_names)

    # Share by type
    shares = df.groupby("D")["ctype"].value_counts(normalize=True).unstack(fill_value=0)
    counts = df.groupby("D")["ctype"].value_counts().unstack(fill_value=0)

    # Discretion
    shares = shares.where(counts >= 3, 0)

    # ---- Plot

    colors = [".8", ".5", ".2", "1", "1"]
    ecolors = [".8", ".5", ".2", "0.2", ".2"]
    hatches = [None, None, None, "/", "."]

    ax = shares.plot(
        kind="barh", stacked=True, figsize=(8, 6), color="white", edgecolor="black"
    )

    for i, bari in enumerate(ax.patches):
        col = i // len(shares)
        bari.set_facecolor(colors[col])
        bari.set_edgecolor(ecolors[col])
        bari.set_hatch(hatches[col])

    # Edits
    ax.set_xlabel("Share")
    ax.set_ylabel("")
    ax.set_yticks(
        [0, 1], labels=["Conviction only", "Conviction and deportation order"]
    )
    ax.tick_params(axis="y", rotation=45)
    ax.set_xlim((0, 1))

    # Legend
    handles = [
        mpatches.Patch(facecolor=cl, edgecolor=ec, hatch=h, label=colname)
        for cl, ec, h, colname in zip(colors, ecolors, hatches, shares.columns)
    ]
    ax.legend(handles=handles, bbox_to_anchor=(0.5, -0.12), ncol=3)

    return ax.figure, ax
