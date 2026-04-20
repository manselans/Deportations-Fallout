"""
Find spouses of deportees and convicted immigrants.
"""

import pandas as pd

from deportations_fallout.utils.stata_io import gather
from deportations_fallout.config import PATHS
paths = PATHS

def run():
    """
    Find spouses of deportees and convicted immigrants.
    """

    # Load population
    population = pd.read_parquet(paths.temp/"population.parquet")

    # Remove those without registered spouse/partner
    spouses = population[
        population["aegte_id"].notna() | population["e_faelle_id"].notna()
    ]

    # Demarcate conflicts
    conflict = (
        spouses["aegte_id"].notna()
        & spouses["e_faelle_id"].notna()
        & (spouses["aegte_id"] != spouses["e_faelle_id"])
    )

    # Duplicate conflicts
    spouses = pd.concat(
        [
            spouses.loc[~conflict].assign(
                partner=spouses["aegte_id"].fillna(spouses["e_faelle_id"])
            ),
            spouses.loc[conflict].assign(partner=spouses["aegte_id"]),
            spouses.loc[conflict].assign(partner=spouses["e_faelle_id"]),
        ],
        ignore_index=True,
    )

    # Keep only a given spouse/partner's first experience
    spouses = spouses.sort_values(["partner", "conviction_date"]).drop_duplicates(
        "partner", keep="first"
    )

    # Edit
    vl = ["partner", "pnr", "conviction_date", "D", "residency", "kom", "year"]

    spouses = (
        spouses[vl]
        .rename(columns={"pnr": "convict", "partner": "pnr", "kom": "municipality"})
        .reset_index(drop=True)
    )

    # ---- Find spouses in population

    first = spouses.year.min()
    last = spouses.year.max()
    years = range(first - 1, last + 1)

    ids = set(spouses["pnr"])
    vl = ["pnr", "opr_land", "ie_type", "koen", "statsb", "alder", "antboernf"]

    bef = gather(
        paths.data.dst_raw,
        names=years,
        file_pattern="bef12_{name}.dta",
        columns=vl,
        filters={"pnr": ids},
        add_name="year",
        concatenate=True,
    )

    # Year prior
    bef["year"] = bef["year"] + 1

    spouses = (
        spouses.merge(bef, on=["pnr", "year"], how="inner", validate="1:1")
        .assign(koen=lambda d: d.koen.eq(2))
        .rename(
            columns={
                "opr_land": "origin",
                "statsb": "citizenship",
                "antboernf": "children",
                "koen": "female",
                "alder": "age",
            }
        )
    )

    spouses.to_parquet(paths.temp/"spouses.parquet")
