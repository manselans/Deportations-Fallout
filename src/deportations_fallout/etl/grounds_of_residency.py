"""
Gather spouses' grounds of residency.
"""

import pandas as pd
import numpy as np

from deportations_fallout.utils.stata_io import gather
from deportations_fallout.utils.lookup import load_lookup
from deportations_fallout.config import PATHS

paths = PATHS


def run():
    """
    Gather spouses' grounds of residency.
    """

    spouses = pd.read_parquet(paths.temp / "spouses.parquet")

    # ---- Grounds of residency

    ids = set(spouses["pnr"])
    years = range(1997, 2021)
    vl = ["pnr", "tilladelsesdato", "kategori"]

    ophg = gather(
        paths.data.dst_raw,
        names=years,
        file_pattern="ophg{name}.dta",
        columns=vl,
        filters={"pnr": ids},
        concatenate=True,
    )

    # Keep latest permit prior to conviction
    ophg = ophg.merge(spouses[["pnr", "conviction_date"]], on="pnr", how="left")

    dates = ["tilladelsesdato", "conviction_date"]  # Make monthly...
    ophg[dates] = ophg[dates].apply(
        lambda c: pd.to_datetime(c, errors="coerce").dt.to_period("M")
    )

    ophg = ophg[ophg["tilladelsesdato"] <= ophg["conviction_date"]]

    ophg = (
        ophg.sort_values("tilladelsesdato")
        .drop_duplicates("pnr", keep="last")
        .reset_index(drop=True)
        .rename(
            columns={
                "kategori": "category"  # Spouses' grounds of residency category (highest level)
            }
        )
    )

    spouses = spouses.merge(ophg[["pnr", "category"]], on="pnr", how="left")

    # Imputing grounds of residency based on citizenship
    spouses["grounds"] = np.nan

    kingdom = [5100, 5115, 5902, 5901, 5101]  # Denmark, Greenaland, Faroese Islands
    spouses.loc[spouses["citizenship"].isin(kingdom), "grounds"] = 0

    eu_eea = load_lookup("countries.csv").loc[
        lambda d: d.eu_eea.eq(1), "code"
    ]  # EU/EEA
    spouses.loc[
        spouses["citizenship"].isin(eu_eea) & ~spouses["citizenship"].isin(kingdom),
        "grounds",
    ] = 1

    # From OPHG
    mis = spouses["grounds"].isna()
    spouses.loc[mis & spouses["category"].eq(4), "grounds"] = 1  # EU/EEA
    spouses.loc[mis & spouses["category"].eq(1), "grounds"] = 2  # Asylum
    spouses.loc[mis & spouses["category"].isin([3, 6]), "grounds"] = 3  # Study/work
    spouses.loc[mis & spouses["category"].eq(5), "grounds"] = 4  # Family Reunified
    spouses.loc[mis & spouses["category"].eq(2), "grounds"] = 5  # Unspecified

    spouses = spouses.drop(columns=["category"])

    # Save
    spouses.to_parquet(paths.temp / "spouses.parquet")
