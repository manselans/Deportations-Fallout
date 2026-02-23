"""
Gathers data on the female foreign-born population in 2010 for Table A2.
Saves to temp.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from depo_paper.config import PATHS
from depo_paper.tools.io import fetch, gather, load_csv


def run():
    """
    Gathers data on the female foreign-born population in 2010.
    Saves to temp.
    """

    # expose paths (and create output folders if they do not exist)
    paths = PATHS

    # load spouses
    spouses = pd.read_parquet(paths.temp / "spouses.parquet")

    # ---- Female Foreign-Born Population, 2010

    ffb = (
        fetch(
            paths.dst / "bef12_2010.dta",
            columns=[
                "pnr",
                "alder",
                "antboernf",
                "ie_type",
                "e_faelle_id",
                "aegte_id",
                "statsb",
            ],
            filters={
                "pnr": lambda d: ~d.isin(spouses["pnr"]),
                "ie_type": [2, 3],  # Immigrant/descendant
                "koen": 2,  # Female
                "plads": lambda d: d.ne(3),  # Not a child living at home
            },
        )
        .loc[
            lambda d: d["e_faelle_id"].ne("") | d["aegte_id"].ne("")
        ]  # must have spouse/partner
        .drop(columns=["e_faelle_id", "aegte_id"])
        .assign(
            ie_type=lambda d: d["ie_type"].eq(3),
            has_children=lambda d: d["antboernf"].ge(1),
            spouse=0,
        )
        .rename(
            columns={"alder": "age", "antboernf": "children", "ie_type": "born_in_dk"}
        )
    )

    # Add covariates from 2010
    ids = set(ffb["pnr"])

    ffb = (
        ffb
        # Criminal records
        .assign(
            convicted=lambda d: d["pnr"].isin(
                fetch(  # penal crime convictions, 2010
                    paths.dst / "kraf2010.dta",
                    columns=["pnr"],
                    filters={"afg_ger7": range(1000000, 2000000)},
                )["pnr"]
            ),
            charged=lambda d: d["pnr"].isin(
                fetch(  # penal crime charges, 2010
                    paths.dst / "krsi2010.dta",
                    columns=["pnr"],
                    filters={"sig_ger7": range(1000000, 2000000)},
                )["pnr"]
            ),
            incarcerated=lambda d: d["pnr"].isin(
                fetch(  # non-arrest incarcerations
                    paths.crime / "krin_placering.dta",
                    columns=["pnr", "fgsldto", "losldto", "handelse"],
                    filters={
                        "handelse": lambda d: d.ge(3),
                        "fgsldto": lambda d: d.ge(pd.Timestamp("1991-01-01")),
                    }
                )
                # Missing release dates for transfers assumed day of incarceration
                .assign(
                    losldto=lambda d: d["fgsldto"].where(
                        d["handelse"].eq(6) & d["losldto"].isna(), d["losldto"]
                    )
                )
                # Remaining missing release dates assumed 1-day stays
                .assign(
                    losldto=lambda d: d["fgsldto"].where(
                        d["losldto"].isna(), d["losldto"]
                    )
                )
                # Only incarceration commenced before end of 2010 and not ending before 2010
                .loc[
                    lambda d: d["fgsldto"].lt(pd.Timestamp("2011-01-01"))
                    & ~d["losldto"].lt(pd.Timestamp("2010-01-01"))
                ]["pnr"]
            ),
        )
        # BFL: Wages and hours worked
        .merge(
            fetch(
                paths.dst / "bfl2010.dta",
                columns=["pnr", "ajo_smalt_loenbeloeb", "ajo_fuldtid_beskaeftiget"],
                filters={"pnr": ids},
            )
            .groupby("pnr")
            .agg(
                wages=(
                    "ajo_smalt_loenbeloeb",
                    lambda x: x.sum() / 1000,
                ),  # Total wages, 1.000 DKK
                fulltime=(
                    "ajo_fuldtid_beskaeftiget",
                    lambda x: x.sum() / 12,
                ),  # Av. share of fulltime
            ),
            on="pnr",
            how="left",
            validate="1:1",
        )
        # ILME: Public transfers
        .merge(
            fetch(
                paths.dst / "ilme12_2010.dta",
                columns=["pnr", "vmo_a_indk_am_bidrag_fri"],
                filters={"pnr": ids},
            )
            .groupby("pnr")
            .agg(
                transfers=(
                    "vmo_a_indk_am_bidrag_fri",
                    lambda x: x.sum() / 1000,
                ),  # Total transfers, 1.000 DKK
            ),
            on="pnr",
            how="left",
            validate="1:1",
        )
        # IND: Net assets
        .merge(
            fetch(
                paths.dst / "ind2010.dta",
                columns=["pnr", "formrest_ny05"],
                filters={"pnr": ids},
            )
            .groupby("pnr")
            .agg(
                assets=("formrest_ny05", lambda x: x.sum() / 1000)
            ),  # Net assets, 1.000 DKK
            on="pnr",
            how="left",
            validate="1:1",
        )
    )

    # Assume no employment/wages/transfers/assets if missing
    ffb = ffb.fillna(0).assign(partic=lambda d: d["wages"].gt(0))

    # Add legal grounds for residency
    ophg = (
        gather(
            paths.dst,
            names=range(1997, 2020),
            file_pattern="ophg{name}.dta",
            columns=["pnr", "tilladelsesdato", "kategori"],
            filters={"pnr": ids},
            concatenate=True,
        )
        .assign(
            tilladelsesdato=lambda d: pd.to_datetime(
                d["tilladelsesdato"], errors="coerce"
            )
        )
        .loc[lambda d: d["tilladelsesdato"].dt.year.le(2010)]
        .sort_values("tilladelsesdato")
        .drop_duplicates("pnr", keep="last")[["pnr", "kategori"]]
    )

    ffb = ffb.merge(ophg, on="pnr", how="left", validate="1:1")

    # Imputing grounds of residency based on citizenship
    ffb["grounds"] = np.nan

    kingdom = [5100, 5115, 5902, 5901, 5101]  # Denmark, Greenaland, Faroese Islands
    ffb.loc[ffb["statsb"].isin(kingdom), "grounds"] = 0

    eu_eea = load_csv("countries.csv").loc[lambda d: d.eu_eea.eq(1), "code"]  # EU/EEA
    ffb.loc[
        ffb["statsb"].isin(eu_eea) &
        ~ffb["statsb"].isin(kingdom),
        "grounds"] = 1

    # From OPHG
    still_mis = ffb["grounds"].isna()
    ffb.loc[still_mis & ffb["kategori"].eq(4), "grounds"] = 1  # EU/EEA
    ffb.loc[still_mis & ffb["kategori"].eq(1), "grounds"] = 2  # Asylum
    ffb.loc[still_mis & ffb["kategori"].isin([3, 6]), "grounds"] = 3  # Study/work
    ffb.loc[still_mis & ffb["kategori"].eq(5), "grounds"] = 4  # Family Reunified
    ffb.loc[still_mis & ffb["kategori"].eq(2), "grounds"] = 5  # Unspecified

    ffb.drop(columns=["kategori"], inplace=True)

    ffb.to_parquet(paths.temp / "female_foreign_born.parquet")


if __name__ == "__main__":
    run()
    