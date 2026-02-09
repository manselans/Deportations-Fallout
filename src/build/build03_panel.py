"""
Assembles panel and adds migration patterns.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from tools.paths import init
from tools.io import fetch, gather


def run(window: iter = range(-18, 19)):
    """
    Creates panel with a given number of months before and after conviction.

    Parameters
    ----------
    window
        Event time periods to be included included in panel.
        Measured as months relative to month of conviction.
    """

    # expose paths (and create output folders if they do not exist)
    paths = init(Path.cwd())

    # load spouses
    spouses = pd.read_parquet(paths.data / "spouses.parquet")
    ids = set(spouses["pnr"])

    # ---- Create panel

    panel = (
        spouses
        # Ensure date-times
        .assign(
            conviction_date=pd.to_datetime(spouses["conviction_date"], errors="coerce"),
            residency=pd.to_datetime(spouses["residency"], errors="coerce"),
        )
        # Generate seniority and make dates monthly
        .assign(
            seniority=lambda d: (d["conviction_date"] - d["residency"]).dt.days,
            conviction_date=lambda d: d["conviction_date"].dt.to_period("M"),
        ).drop(columns=["residency"])
    )

    # Expand and define event and calendar time
    panel = panel.loc[panel.index.repeat(len(window)).copy()]
    panel["event_time"] = list(window) * len(spouses)

    panel["month"] = panel["conviction_date"] + panel["event_time"]

    # ---- Load migrations

    vnds = fetch(
        paths.dst / "vnds2021.dta",
        columns=["pnr", "handdto", "indud"],
        filters={"pnr": ids},
        convert_categoricals=True,
    )
    vnds["month"] = vnds["handdto"].dt.to_period("M")

    # Two same-day movements are ambiguous
    vnds["dup_day"] = vnds.duplicated(["pnr", "handdto"], keep=False)

    # Order within month and make monthly
    vnds = vnds.sort_values(["pnr", "month", "handdto"])

    g = vnds.groupby(["pnr", "month"], sort=False)
    moves = pd.DataFrame(
        {
            "first": g["indud"].first(),
            "last": g["indud"].last(),
            "n": g.size(),
            "ambig": g["dup_day"].any(),
        }
    ).reset_index()

    # ---- Define monthly movement based on movements within month

    # Default: Last
    moves["code"] = moves["last"]

    # Multiple, unambiguous, movements within month
    mask = (moves["n"] > 1) & (~moves["ambig"])

    moves["code"] = moves["code"].cat.add_categories(["NONE", "S"])

    # Case 1: Out-In in a month is defined as non-event (ignore time gone)
    moves.loc[mask & (moves["first"].eq("U") & moves["last"].eq("I")), "code"] = "NONE"

    # Case 2 & 3: In-In and Out-Out are just defined as In and Out, respectively

    # Case 4: In-Out is marked as special case
    moves.loc[mask & (moves["first"].eq("I") & moves["last"].eq("U")), "code"] = "S"

    # ---- Merge onto panel and determine migration state

    panel = panel.merge(
        moves[["pnr", "month", "code", "ambig"]], on=["pnr", "month"], how="left"
    )

    # Define a state-changer
    panel["set"] = np.nan
    panel.loc[panel["code"].eq("I"), "set"] = 1  # Set 1
    panel.loc[panel["code"].eq("U"), "set"] = 0  # Set 0

    # Carry forward state until change; intitial assumption: IN
    state = panel.groupby("pnr")["set"].ffill().fillna(1.0)

    # Adjust initial assumption: If first state-changer is IN, assume prior periods OUT
    has_set = panel["set"].notna()
    first_idx = (  # Earliest recorded month of movement
        panel.loc[has_set].groupby("pnr")["month"].idxmin()
    )
    first = panel.loc[first_idx, ["pnr", "month", "set"]].set_index("pnr")

    m_first = panel["pnr"].map(first["month"])

    first_is_in = panel["pnr"].map(first["set"]).eq(1)

    state = state.mask(first_is_in & (panel["month"] < m_first), 0)

    # Determine migration state
    panel["in_country"] = state.astype("boolean")
    panel.loc[panel["code"] == "S", "in_country"] = True  # In-Out in same month

    panel.drop(columns=["code", "ambig", "set"], inplace=True)

    # Define a mover
    cond = ~panel["in_country"] & (panel["event_time"] >= 0)
    panel["mover"] = cond.groupby(panel["pnr"]).transform("max")

    # ---- Assemble panel

    # load outcomes and covariates
    bfl = pd.read_parquet(paths.data / "bfl.parquet")
    ilme = pd.read_parquet(paths.data / "ilme.parquet")
    kraf = pd.read_parquet(paths.data / "kraf.parquet")
    krsi = pd.read_parquet(paths.data / "krsi.parquet")
    krin = pd.read_parquet(paths.data / "krin.parquet")

    panel = (
        panel
        # Add LMO
        .merge(bfl, on=["pnr", "month"], how="left", validate="1:1")
        .merge(ilme, on=["pnr", "month"], how="left", validate="1:1")
        # Add criminal records
        .merge(
            kraf, on=["pnr", "month"], how="left", validate="1:1", indicator="convicted"
        )
        .merge(
            krsi, on=["pnr", "month"], how="left", validate="1:1", indicator="charged"
        )
        .merge(
            krin,
            on=["pnr", "month"],
            how="left",
            validate="1:1",
            indicator="incarcerated",
        )
    )

    # ---- Assumptions

    present = panel["in_country"]
    lmo = panel["month"].dt.year.between(2008, 2021)

    # LMO: If present and no wages, assumes wages are 0
    for c in ["wages", "fulltime", "transfers"]:
        panel.loc[lambda d, c=c: present & lmo & d[c].isna(), c] = 0

    panel["partic"] = panel["wages"].gt(0).where(panel["wages"].notna())

    # Criminal records: If present and no crime recorded assume no crime
    for c in ["convicted", "charged", "incarcerated"]:
        panel.loc[lambda d, c=c: ~present & d[c].eq("left_only"), c] = np.nan
        panel[c] = panel[c].eq("both").where(panel[c].notna())

    # ---- Remove co-convicted spouses

    # control-group convicts and their partners
    control = spouses.loc[spouses.D.eq(0), ["pnr", "conviction_date", "convict"]]
    ids = set(pd.concat([control["pnr"], control["convict"]]))

    # Gather conviction records
    years = range(2000, 2022)
    vl = ["pnr", "afg_afgoedto", "journr"]

    kraf = gather(
        paths.dst,
        names=years,
        file_pattern="kraf{name}.dta",
        columns=vl,
        filters={"pnr": ids},
        concatenate=True,
    )

    kraf = kraf.rename(columns={"afg_afgoedto": "conviction_date", "pnr": "convict"})

    # add journal numbers from day of conviction (multiple possible)
    control = control.merge(kraf, on=["convict", "conviction_date"], how="left")

    # exclude couples convicted on same journal number from control group
    pairs = pd.MultiIndex.from_frame(control[["pnr", "journr"]])
    flag = pd.MultiIndex.from_frame(control[["convict", "journr"]]).isin(pairs)

    spouses = spouses.loc[lambda d: ~d.pnr.isin(control[flag].pnr)]
    panel = panel.loc[lambda d: d.pnr.isin(spouses.pnr)]

    # Save
    panel.to_parquet(paths.data / "panel.parquet")
    spouses.to_parquet(paths.data / "spouses.parquet")


if __name__ == "__main__":
    run()
    