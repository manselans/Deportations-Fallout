"""
    Gathers spouses' outcomes and covariates.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from tools.paths import init
from tools.io import fetch, gather, load_csv


def run():
    """
    Gathers
    - Spouses' wages, transfers and hours worked (and saves as data sets).
    - Spouses' criminal records (and saves as data sets).
    - Spouses' legal grounds of residency (and adds to spouse data).
    """

    # expose paths (and create output folders if they do not exist)
    paths = init(Path.cwd())

    # load spouses
    spouses = pd.read_parquet(paths.data / "spouses.parquet")

    ids = set(spouses["pnr"])
    years = range(2008, 2022)  # data availability

    # ---- BFL: Wages and share of fulltime
    vl = [
        "pnr",
        "ajo_job_slut_dato",
        "ajo_smalt_loenbeloeb",
        "ajo_fuldtid_beskaeftiget",
    ]

    bfl = gather(
        paths.dst,
        names=years,
        file_pattern="bfl{name}.dta",
        columns=vl,
        filters={"pnr": ids},
        concatenate=True,
    )

    # Make monthly
    bfl = (
        bfl.assign(
            month=pd.to_datetime(
                bfl["ajo_job_slut_dato"], errors="coerce"
            ).dt.to_period("M")
        )
        .dropna(subset=["month"])
        .groupby(["pnr", "month"])
        .agg(
            wages=(
                "ajo_smalt_loenbeloeb",
                lambda x: x.sum() / 1000,
            ),  # Total wages, 1.000 DKK
            fulltime=(
                "ajo_fuldtid_beskaeftiget",
                lambda x: x.sum(),
            ),  # Share of fulltime
        )
        .reset_index()
    )

    # ---- ILME: Public transfers
    vl = ["pnr", "vmo_slutdato", "vmo_a_indk_am_bidrag_fri"]

    ilme = gather(
        paths.dst,
        names=years,
        file_pattern="ilme12_{name}.dta",
        columns=vl,
        filters={"pnr": ids},
        concatenate=True,
    )

    # Make monthly
    ilme = (
        ilme.assign(
            month=pd.to_datetime(ilme["vmo_slutdato"], errors="coerce").dt.to_period(
                "M"
            )
        )
        .dropna(subset=["month"])
        .groupby(["pnr", "month"])
        .agg(
            transfers=(
                "vmo_a_indk_am_bidrag_fri",
                lambda x: x.sum() / 1000,
            ),  # Total transfers, 1.000 DKK
        )
        .reset_index()
    )

    # Truncate
    for c in ["wages", "fulltime"]:
        bfl[c].clip(lower=0, upper=bfl[c].quantile(0.99), inplace=True)

    ilme["transfers"].clip(
        lower=0, upper=ilme["transfers"].quantile(0.99), inplace=True
    )

    # ---- Criminal records

    # KRAF: Convictions, penal law
    years = range(1998, 2022)
    vl = ["pnr", "afg_ger7", "afg_afgoedto"]

    kraf = gather(
        paths.dst,
        names=years,
        file_pattern="kraf{name}.dta",
        columns=vl,
        concatenate=True,
        filters={"pnr": ids, "afg_ger7": range(1000000, 2000000)},
    )

    # Make KRAF monthly
    kraf = (
        kraf.assign(
            month=pd.to_datetime(kraf["afg_afgoedto"], errors="coerce").dt.to_period(
                "M"
            )
        )[["pnr", "month"]]
        .dropna()
        .drop_duplicates()
    )

    # KRSI: Charges, penal law
    years = range(1998, 2022)
    vl = ["pnr", "sig_ger7", "sig_sigtdto"]

    krsi = gather(
        paths.dst,
        names=years,
        file_pattern="krsi{name}.dta",
        columns=vl,
        concatenate=True,
        filters={"pnr": ids, "sig_ger7": range(1000000, 2000000)},
    )

    # Make KRSI monthly
    krsi = (
        krsi.assign(
            month=pd.to_datetime(krsi["sig_sigtdto"], errors="coerce").dt.to_period("M")
        )[["pnr", "month"]]
        .dropna()
        .drop_duplicates()
    )

    # KRIN: Non-arrest incarcerations
    vl = ["pnr", "fgsldto", "losldto", "handelse"]

    krin = fetch(
        paths.crime / "krin_placering.dta",
        columns=vl,
        filters={"pnr": ids, "handelse": lambda c: ~c.isin([1, 2])},
    )

    krin = (
        krin
        # Keep only incarcerations since 1998
        .loc[lambda d: (d["fgsldto"].dt.year >= 1998) | (d["losldto"].dt.year >= 1998)]
        # ---- Dealing w. missing release dates
        # Missing release dates for transfers are set to day of entry
        .assign(
            losldto=lambda d: d["losldto"].mask(
                d["losldto"].isna() & d["handelse"].eq(6), d["fgsldto"]
            )
        )
        # Keep latest release by entry
        .sort_values(
            ["pnr", "fgsldto", "losldto"], na_position="first"
        ).drop_duplicates(["pnr", "fgsldto"], keep="last")
        # Recent incarcerations (likely not finished)
        .assign(
            losldto=lambda d: d["losldto"].mask(
                d["losldto"].isna() & d["fgsldto"].dt.year.ge(2021),
                pd.Timestamp(day=31, month=12, year=2023),
            )
        )
        # Set remaining to 1-day stays
        .assign(losldto=lambda d: d["losldto"].fillna(d["fgsldto"]))
    )

    # Make KRIN monthly
    krin[["fgsldto", "losldto"]] = krin[["fgsldto", "losldto"]].apply(
        lambda c: c.dt.to_period("M")
    )

    krin["month"] = krin.apply(
        lambda r: pd.period_range(r["fgsldto"], r["losldto"], freq="M"), axis=1
    )

    krin = (
        krin[["pnr", "month"]]
        .explode("month", ignore_index=True)
        .drop_duplicates(["pnr", "month"])  # Remove overlapping intervals
    )

    # ---- IND: Assets

    years = range(1999, 2022)
    vl = ["pnr", "formrest_ny05"]

    ind = gather(
        paths.dst,
        names=years,
        file_pattern="ind{name}.dta",
        columns=vl,
        filters={"pnr": ids},
        concatenate=True,
        add_name="year",
    )

    # Collapse by person
    ind = ind.groupby(["pnr", "year"], as_index=False).agg(
        assets=("formrest_ny05", lambda x: x.sum() / 1000)
    )  # Net assets, 1.000 DKK

    # Year prior
    ind["year"] = ind["year"] + 1
    spouses = spouses.merge(ind, on=["pnr", "year"], how="left")

    # If missing, but in population register: assume 0
    spouses.loc[spouses["assets"].isna(), "assets"] = 0

    # Truncate
    spouses["assets"].clip(
        lower=spouses["assets"].quantile(0.01),
        upper=spouses["assets"].quantile(0.99),
        inplace=True,
    )

    # ---- Grounds of residency

    years = range(1997, 2021)
    vl = ["pnr", "tilladelsesdato", "kategori"]

    ophg = gather(
        paths.dst,
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

    eu_eea = load_csv("countries.csv").loc[lambda d: d.eu_eea.eq(1), "code"]  # EU/EEA
    spouses.loc[
        spouses["citizenship"].isin(eu_eea) & ~spouses["citizenship"].isin(kingdom),
        "grounds",
    ] = 1

    # From OPHG
    still_mis = spouses["grounds"].isna()
    spouses.loc[still_mis & spouses["category"].eq(4), "grounds"] = 1  # EU/EEA
    spouses.loc[still_mis & spouses["category"].eq(1), "grounds"] = 2  # Asylum
    spouses.loc[still_mis & spouses["category"].isin([3, 6]), "grounds"] = (
        3  # Study/work
    )
    spouses.loc[still_mis & spouses["category"].eq(5), "grounds"] = (
        4  # Family Reunified
    )
    spouses.loc[still_mis & spouses["category"].eq(2), "grounds"] = 5  # Unspecified

    spouses.drop(columns=["category"], inplace=True)

    # Save data
    bfl.to_parquet(paths.data / "bfl.parquet")
    ilme.to_parquet(paths.data / "ilme.parquet")
    kraf.to_parquet(paths.data / "kraf.parquet")
    krsi.to_parquet(paths.data / "krsi.parquet")
    krin.to_parquet(paths.data / "krin.parquet")
    spouses.to_parquet(paths.data / "spouses.parquet")


if __name__ == "__main__":
    run()
