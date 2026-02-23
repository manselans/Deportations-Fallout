"""
Builds the population and spouse data sets.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from depo_paper.config import PATHS
from depo_paper.tools.io import gather
from depo_paper.tools.rules import rd_exception


def run():
    """
    Finds the population (control + treatment) for the analysis:
    - Gathers relevant deportation orders
    - Identifies control group
    - Identifies spouses
    - Saves data sets on convicted population and their spouses, respectively
    """

    # expose paths (and create output folders if they do not exist)
    paths = PATHS

    # ---- Deportation orders

    # Load PAC data
    pac_orders = pd.read_sas(
        paths.dst / "Eksterne data/rf0961502_udrejseforbud.sas7bdat"
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

    # ---- Residency permits: All immigrants

    years = range(1997, 2021)
    vl = ["pnr", "tilladelsesdato"]

    ophg = gather(
        paths.dst,
        names=years,
        file_pattern="ophg{name}.dta",
        columns=vl,
        concatenate=True,
    )
    ophg["residency"] = pd.to_datetime(ophg["tilladelsesdato"], errors="coerce")

    # Keep earliest permit per person
    ophg = (
        ophg.sort_values(["pnr", "residency"], na_position="last")
        .drop_duplicates("pnr", keep="first")
        .dropna()[["pnr", "residency"]]
    )

    # ---- Conviction records

    years = range(1980, 2022)
    vl = ["pnr", "afg_ger7", "afg_bstrflgd", "afg_ubstrflg", "afg_afgoedto"]
    ids = set(pd.concat([ophg["pnr"], pac_orders["pnr"]]))

    kraf = gather(
        paths.dst,
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

    # Has recieved prison sentence in the past?
    kraf = kraf.sort_values(["pnr", "conviction_date"])
    kraf["repeat"] = kraf.groupby("pnr")["prison"].transform(
        lambda x: x.shift().cumsum() > 0
    )

    # Demarcate criminal offenses exempt from seniority rules
    kraf["exception"] = rd_exception(kraf["afg_ger7"])

    # ---- Defining control group

    # Has residency permit and been convicted but never derpoted
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
                **{c: "max" for c in ["prison", "repeat", "exception"]},  # any
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
                **{c: "max" for c in ["prison", "repeat", "exception"]},  # any
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

    # ---- Population: treatment and control

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
        paths.dst,
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

    # Update only missings
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

    # ---- Spouses

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
        paths.dst,
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

    # Save data
    population.to_parquet(paths.temp / "population.parquet")
    spouses.to_parquet(paths.temp / "spouses.parquet")


if __name__ == "__main__":
    run()
