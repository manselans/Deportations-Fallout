"""
Gather spouse outcomes: Labour market outcomes and criminal records.
"""

import pandas as pd

from deportations_fallout.utils.stata_io import fetch, gather
from deportations_fallout.config import PATHS

paths = PATHS


def run():
    """
    Gather spouse outcomes: Labour market outcomes and criminal records.
    """

    spouses = pd.read_parquet(paths.temp / "spouses.parquet")

    # ---- Wages and Transfers

    ids = set(spouses["pnr"])
    years = range(2008, 2022)  # data availability

    # BFL: Wages and share of fulltime

    vl = [
        "pnr",
        "ajo_job_slut_dato",
        "ajo_smalt_loenbeloeb",
        "ajo_fuldtid_beskaeftiget",
    ]

    bfl = gather(
        paths.data.dst_raw,
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

    # ILME: Public transfers

    vl = ["pnr", "vmo_slutdato", "vmo_a_indk_am_bidrag_fri"]

    ilme = gather(
        paths.data.dst_raw,
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

    # ---- Criminal Records

    # KRAF: Convictions, penal law
    years = range(1998, 2022)
    vl = ["pnr", "afg_ger7", "afg_afgoedto"]

    kraf = gather(
        paths.data.dst_raw,
        names=years,
        file_pattern="kraf{name}.dta",
        columns=vl,
        concatenate=True,
        filters={"pnr": ids, "afg_ger7": range(1000000, 2000000)},
    )

    # Make monthly
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
        paths.data.dst_raw,
        names=years,
        file_pattern="krsi{name}.dta",
        columns=vl,
        concatenate=True,
        filters={"pnr": ids, "sig_ger7": range(1000000, 2000000)},
    )

    # Make monthly
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
        paths.data.crime / "krin_placering.dta",
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

    # Make monthly
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
        paths.data.dst_raw,
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

    # Save
    bfl.to_parquet(paths.temp / "bfl.parquet")
    ilme.to_parquet(paths.temp / "ilme.parquet")
    kraf.to_parquet(paths.temp / "kraf.parquet")
    krsi.to_parquet(paths.temp / "krsi.parquet")
    krin.to_parquet(paths.temp / "krin.parquet")
    spouses.to_parquet(paths.temp / "spouses.parquet")
