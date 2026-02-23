"""Gathers spouses' outcomes and covariates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from depo_paper.config import PATHS
from depo_paper.tools.io import fetch, gather, load_csv


def _build_lmo(ids: set[int], paths):
    """Build monthly wages/fulltime/transfers outcomes."""

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

    bfl = (
        bfl.assign(
            month=pd.to_datetime(bfl["ajo_job_slut_dato"], errors="coerce").dt.to_period(
                "M"
            )
        )
        .dropna(subset=["month"])
        .groupby(["pnr", "month"])
        .agg(
            wages=(
                "ajo_smalt_loenbeloeb",
                lambda x: x.sum() / 1000,
            ),
            fulltime=("ajo_fuldtid_beskaeftiget", lambda x: x.sum()),
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
            ),
        )
        .reset_index()
    )

    # Truncate
    for c in ["wages", "fulltime"]:
        bfl[c].clip(lower=0, upper=bfl[c].quantile(0.99), inplace=True)

    ilme["transfers"].clip(
        lower=0, upper=ilme["transfers"].quantile(0.99), inplace=True
    )

    return bfl, ilme


def _build_crime_records(ids: set[int], paths):
    """Build monthly crime outcomes (conviction, charge, incarceration)."""

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
    vl = ["pnr", "sig_ger7", "sig_sigtdto"]

    krsi = gather(
        paths.dst,
        names=years,
        file_pattern="krsi{name}.dta",
        columns=vl,
        concatenate=True,
        filters={"pnr": ids, "sig_ger7": range(1000000, 2000000)},
    )

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
        krin.loc[lambda d: (d["fgsldto"].dt.year >= 1998) | (d["losldto"].dt.year >= 1998)]
        .assign(
            losldto=lambda d: d["losldto"].mask(
                d["losldto"].isna() & d["handelse"].eq(6), d["fgsldto"]
            )
        )
        .sort_values(["pnr", "fgsldto", "losldto"], na_position="first")
        .drop_duplicates(["pnr", "fgsldto"], keep="last")
        .assign(
            losldto=lambda d: d["losldto"].mask(
                d["losldto"].isna() & d["fgsldto"].dt.year.ge(2021),
                pd.Timestamp(day=31, month=12, year=2023),
            )
        )
        .assign(losldto=lambda d: d["losldto"].fillna(d["fgsldto"]))
    )

    krin[["fgsldto", "losldto"]] = krin[["fgsldto", "losldto"]].apply(
        lambda c: c.dt.to_period("M")
    )

    krin["month"] = krin.apply(
        lambda r: pd.period_range(r["fgsldto"], r["losldto"], freq="M"), axis=1
    )

    krin = (
        krin[["pnr", "month"]]
        .explode("month", ignore_index=True)
        .drop_duplicates(["pnr", "month"])
    )

    return kraf, krsi, krin


def _add_assets(spouses: pd.DataFrame, ids: set[int], paths) -> pd.DataFrame:
    """Merge prior-year net assets onto spouse sample."""

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

    ind = ind.groupby(["pnr", "year"], as_index=False).agg(
        assets=("formrest_ny05", lambda x: x.sum() / 1000)
    )

    ind["year"] = ind["year"] + 1
    spouses = spouses.merge(ind, on=["pnr", "year"], how="left")

    spouses.loc[spouses["assets"].isna(), "assets"] = 0
    spouses["assets"].clip(
        lower=spouses["assets"].quantile(0.01),
        upper=spouses["assets"].quantile(0.99),
        inplace=True,
    )

    return spouses


def _add_residency_grounds(spouses: pd.DataFrame, ids: set[int], paths) -> pd.DataFrame:
    """Merge legal grounds of residency and derive final grounds variable."""

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

    ophg = ophg.merge(spouses[["pnr", "conviction_date"]], on="pnr", how="left")

    dates = ["tilladelsesdato", "conviction_date"]
    ophg[dates] = ophg[dates].apply(
        lambda c: pd.to_datetime(c, errors="coerce").dt.to_period("M")
    )

    ophg = ophg[ophg["tilladelsesdato"] <= ophg["conviction_date"]]

    ophg = (
        ophg.sort_values("tilladelsesdato")
        .drop_duplicates("pnr", keep="last")
        .reset_index(drop=True)
        .rename(columns={"kategori": "category"})
    )

    spouses = spouses.merge(ophg[["pnr", "category"]], on="pnr", how="left")

    spouses["grounds"] = np.nan

    kingdom = [5100, 5115, 5902, 5901, 5101]
    spouses.loc[spouses["citizenship"].isin(kingdom), "grounds"] = 0

    eu_eea = load_csv("countries.csv").loc[lambda d: d.eu_eea.eq(1), "code"]
    spouses.loc[
        spouses["citizenship"].isin(eu_eea) & ~spouses["citizenship"].isin(kingdom),
        "grounds",
    ] = 1

    still_mis = spouses["grounds"].isna()
    spouses.loc[still_mis & spouses["category"].eq(4), "grounds"] = 1
    spouses.loc[still_mis & spouses["category"].eq(1), "grounds"] = 2
    spouses.loc[still_mis & spouses["category"].isin([3, 6]), "grounds"] = 3
    spouses.loc[still_mis & spouses["category"].eq(5), "grounds"] = 4
    spouses.loc[still_mis & spouses["category"].eq(2), "grounds"] = 5

    spouses.drop(columns=["category"], inplace=True)
    return spouses


def _save_outputs(spouses: pd.DataFrame, bfl, ilme, kraf, krsi, krin, paths) -> None:
    """Persist temporary build outputs."""

    bfl.to_parquet(paths.temp / "bfl.parquet")
    ilme.to_parquet(paths.temp / "ilme.parquet")
    kraf.to_parquet(paths.temp / "kraf.parquet")
    krsi.to_parquet(paths.temp / "krsi.parquet")
    krin.to_parquet(paths.temp / "krin.parquet")
    spouses.to_parquet(paths.temp / "spouses.parquet")


def run():
    """Build spouses' outcomes/covariates and save temporary datasets."""

    paths = PATHS
    spouses = pd.read_parquet(paths.temp / "spouses.parquet")

    ids = set(spouses["pnr"])

    bfl, ilme = _build_lmo(ids=ids, paths=paths)
    kraf, krsi, krin = _build_crime_records(ids=ids, paths=paths)
    spouses = _add_assets(spouses=spouses, ids=ids, paths=paths)
    spouses = _add_residency_grounds(spouses=spouses, ids=ids, paths=paths)

    _save_outputs(
        spouses=spouses,
        bfl=bfl,
        ilme=ilme,
        kraf=kraf,
        krsi=krsi,
        krin=krin,
        paths=paths,
    )


if __name__ == "__main__":
    run()
