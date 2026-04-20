"""
Gather deportation orders and convicted immigrants.
"""

import pandas as pd
import numpy as np

from deportations_fallout.utils.stata_io import gather
from deportations_fallout.etl.crime_categories import rd_exception
from deportations_fallout.config import PATHS
paths = PATHS

def run():
    """
    Gather deportation orders and convicted immigrants.
    """

    # ---- Deportation orders

    # Load PAC data
    pac_orders = pd.read_sas(
        paths.data.dst_raw / "Eksterne data/rf0961502_udrejseforbud.sas7bdat"
    )
    pac_orders = pac_orders[["INR", "AFGOERDTO"]]

    # Format and drop missings
    pac_orders["pnr"] = pd.to_numeric(pac_orders["INR"], errors="coerce")
    pac_orders["conviction_date"] = pd.to_datetime(
        pac_orders["AFGOERDTO"], errors="coerce"
    )
    pac_orders = pac_orders[["pnr", "conviction_date"]].dropna()

    # Keep only first deportation order per person
    pac_orders = pac_orders.groupby("pnr")["conviction_date"].min().reset_index()

    # ---- Eligible for deportation

    # Gather residency permits
    years = range(1997, 2021)
    vl = ["pnr", "tilladelsesdato"]

    ophg = gather(
        paths.data.dst_raw,
        names=years,
        file_pattern="ophg{name}.dta",
        columns=vl,
        concatenate=True,
    )
    ophg["residency"] = pd.to_datetime(ophg["tilladelsesdato"], errors="coerce")

    # Keep earliest permit per person
    ophg = (
        ophg
        .sort_values(["pnr", "residency"], na_position="last")
        .drop_duplicates("pnr", keep="first")
        .dropna()
        [["pnr", "residency"]]
    )

    # Gather conviction records
    years = range(1980, 2022)
    vl = ["pnr", "afg_ger7", "afg_bstrflgd", "afg_ubstrflg", "afg_afgoedto"]
    ids = set(pd.concat([ophg["pnr"], pac_orders["pnr"]]))

    kraf = gather(
        paths.data.dst_raw,
        names=years,
        file_pattern="kraf{name}.dta",
        columns=vl,
        filters={"pnr": ids},
        concatenate=True,
    )

    kraf.rename(
        columns={
            "afg_bstrflgd": "suspended_sentence",
            "afg_ubstrflg": "prison_sentence",
            "afg_afgoedto": "conviction_date",
        },
        inplace=True,
    )

    # Fix invalid values
    kraf.loc[kraf["prison_sentence"] == 9999, "prison_sentence"] = np.nan

    # Demarcate prison sentences
    kraf["prison"] = (kraf.prison_sentence.fillna(0) > 0) | (
        kraf.suspended_sentence.fillna(0) > 0
    )

    # Demarcate criminal offenses exempt from seniority rules
    kraf["exception"] = rd_exception(kraf["afg_ger7"])

    # ---- Defining control group

    # Convicted but never deported
    control = ophg.loc[lambda d: d.pnr.isin(kraf.pnr) & ~d.pnr.isin(pac_orders.pnr)]

    # Refine
    control = (
        control
        # Add conviction records
        .merge(kraf, on="pnr", how="left", validate="1:m")
        # Drop anyone convicted prior to recieving their residency permit
        .loc[lambda d: d.conviction_date >= d.residency]
        # Keep only those with prison sentences
        .loc[lambda d: d.prison]
        # Collapse same-day convictions
        .groupby(["pnr", "conviction_date"])
        .agg(
            {
                **{"residency": "last"},  # does not vary within person
                **{c: "max" for c in ["prison", "exception"]},  # any
                **{
                    c: "sum" for c in ["suspended_sentence", "prison_sentence"]
                },  # total
            }
        )
        .reset_index()
        # Keep earliest recorded conviction
        .sort_values(["pnr", "conviction_date"])
        .drop_duplicates(["pnr"], keep="first")
        # Keep only those convicted for the first time since 2000
        .loc[lambda d: d.conviction_date.dt.year >= 2000]
    )

    # ---- Link deportation orders to convictions

    pac_orders = (
        pac_orders
        # Add convictions from day of deportation order
        .merge(kraf, on=["pnr", "conviction_date"], how="left", validate="1:m")
        # Collapse same-day convictions
        .groupby(["pnr", "conviction_date"])
        .agg(
            {
                **{c: "max" for c in ["prison", "exception"]},  # any
                **{
                    c: "sum" for c in ["suspended_sentence", "prison_sentence"]
                },  # total
            }
        )
        .reset_index()
    )

    # Restore missingness
    pac_orders.loc[
        lambda d: d.prison.isna(), ["suspended_sentence", "prison_sentence"]
    ] = np.nan

    # Add date of residency
    pac_orders = pac_orders.merge(
        ophg[["pnr", "residency"]], on="pnr", how="left", validate="1:1"
    )

    population = pd.concat(
        [pac_orders.assign(D=1), control.assign(D=0)], ignore_index=True
    )

    # ---- Find in population registers

    population["year"] = population["conviction_date"].dt.year

    first = population.year.min()
    last = population.year.max()
    years = range(first - 1, last + 1)

    ids = set(population["pnr"])
    vl = [
        "pnr",
        "opr_land",
        "koen",
        "statsb",
        "kom",
        "foed_dag",
        "e_faelle_id",
        "aegte_id",
    ]

    bef = gather(
        paths.data.dst_raw,
        names=years,
        file_pattern="bef12_{name}.dta",
        columns=vl,
        filters={"pnr": ids},
        add_name="year",
        concatenate=True,
    )

    # From year of conviction
    vl_1 = ["pnr", "year", "opr_land", "statsb", "koen", "foed_dag"]
    population = population.merge(
        bef[vl_1], on=["pnr", "year"], how="left", validate="1:1"
    )

    # From year prior to conviction (spouses/partners always found here)
    bef["year"] = bef["year"] + 1
    vl_2 = [
        "pnr",
        "year",
        "opr_land",
        "statsb",
        "koen",
        "foed_dag",
        "kom",
        "e_faelle_id",
        "aegte_id",
    ]
    population = population.merge(
        bef[vl_2],
        on=["pnr", "year"],
        how="left",
        suffixes=("", "_prev"),
        validate="1:1",
    )

    # Update missing
    for var in vl_1:
        if var not in ["pnr", "year"]:
            population[var] = population[var].fillna(population[f"{var}_prev"])

    population = population[
        [var for var in population.columns if not var.endswith("_prev")]
    ]

    # Drop Danish citizens
    population = population[population["statsb"] != 5100]
    population = population.reset_index(drop=True).copy()

    # Destring
    vl = ["aegte_id", "e_faelle_id", "kom"]
    population[vl] = population[vl].apply(pd.to_numeric, errors="coerce")

    population.to_parquet(paths.temp/"population.parquet")
