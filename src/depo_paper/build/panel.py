"""Assembles panel and adds migration patterns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from depo_paper.config import PATHS
from depo_paper.tools.io import fetch, gather


def _initialize_panel(spouses: pd.DataFrame, window: iter) -> pd.DataFrame:
    """Expand spouse-level data into event-time panel skeleton."""

    panel = (
        spouses.assign(
            conviction_date=pd.to_datetime(spouses["conviction_date"], errors="coerce"),
            residency=pd.to_datetime(spouses["residency"], errors="coerce"),
        )
        .assign(
            seniority=lambda d: (d["conviction_date"] - d["residency"]).dt.days,
            conviction_date=lambda d: d["conviction_date"].dt.to_period("M"),
        )
        .drop(columns=["residency"])
    )

    panel = panel.loc[panel.index.repeat(len(window)).copy()]
    panel["event_time"] = list(window) * len(spouses)
    panel["month"] = panel["conviction_date"] + panel["event_time"]

    return panel


def _build_monthly_moves(ids, paths) -> pd.DataFrame:
    """Load and aggregate movement records to monthly movement code."""

    vnds = fetch(
        paths.dst / "vnds2021.dta",
        columns=["pnr", "handdto", "indud"],
        filters={"pnr": ids},
        convert_categoricals=True,
    )
    vnds["month"] = vnds["handdto"].dt.to_period("M")
    vnds["dup_day"] = vnds.duplicated(["pnr", "handdto"], keep=False)
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

    moves["code"] = moves["last"]
    mask = (moves["n"] > 1) & (~moves["ambig"])
    moves["code"] = moves["code"].cat.add_categories(["NONE", "S"])
    moves.loc[mask & (moves["first"].eq("U") & moves["last"].eq("I")), "code"] = "NONE"
    moves.loc[mask & (moves["first"].eq("I") & moves["last"].eq("U")), "code"] = "S"

    return moves[["pnr", "month", "code", "ambig"]]


def _apply_migration_states(panel: pd.DataFrame, moves: pd.DataFrame) -> pd.DataFrame:
    """Merge monthly moves and derive in-country state per month."""

    panel = panel.merge(moves, on=["pnr", "month"], how="left")

    panel["set"] = np.nan
    panel.loc[panel["code"].eq("I"), "set"] = 1
    panel.loc[panel["code"].eq("U"), "set"] = 0

    state = panel.groupby("pnr")["set"].ffill().fillna(1.0)

    has_set = panel["set"].notna()
    first_idx = panel.loc[has_set].groupby("pnr")["month"].idxmin()
    first = panel.loc[first_idx, ["pnr", "month", "set"]].set_index("pnr")

    m_first = panel["pnr"].map(first["month"])
    first_is_in = panel["pnr"].map(first["set"]).eq(1)
    state = state.mask(first_is_in & (panel["month"] < m_first), 0)

    panel["in_country"] = state.astype("boolean")
    panel.loc[panel["code"] == "S", "in_country"] = True
    panel.drop(columns=["code", "ambig", "set"], inplace=True)

    cond = ~panel["in_country"] & (panel["event_time"] >= 0)
    panel["mover"] = cond.groupby(panel["pnr"]).transform("max")

    return panel


def _merge_panel_outcomes(panel: pd.DataFrame, paths) -> pd.DataFrame:
    """Attach monthly outcomes and impose missingness assumptions."""

    bfl = pd.read_parquet(paths.temp / "bfl.parquet")
    ilme = pd.read_parquet(paths.temp / "ilme.parquet")
    kraf = pd.read_parquet(paths.temp / "kraf.parquet")
    krsi = pd.read_parquet(paths.temp / "krsi.parquet")
    krin = pd.read_parquet(paths.temp / "krin.parquet")

    panel = (
        panel.merge(bfl, on=["pnr", "month"], how="left", validate="1:1")
        .merge(ilme, on=["pnr", "month"], how="left", validate="1:1")
        .merge(kraf, on=["pnr", "month"], how="left", validate="1:1", indicator="convicted")
        .merge(krsi, on=["pnr", "month"], how="left", validate="1:1", indicator="charged")
        .merge(
            krin,
            on=["pnr", "month"],
            how="left",
            validate="1:1",
            indicator="incarcerated",
        )
    )

    present = panel["in_country"]
    lmo = panel["month"].dt.year.between(2008, 2021)

    for c in ["wages", "fulltime", "transfers"]:
        panel.loc[lambda d, c=c: present & lmo & d[c].isna(), c] = 0

    panel["partic"] = panel["wages"].gt(0).where(panel["wages"].notna())

    for c in ["convicted", "charged", "incarcerated"]:
        panel.loc[lambda d, c=c: ~present & d[c].eq("left_only"), c] = np.nan
        panel[c] = panel[c].eq("both").where(panel[c].notna())

    return panel


def _remove_coconvicted_controls(panel: pd.DataFrame, spouses: pd.DataFrame, paths):
    """Exclude control-group couples convicted on the same journal number."""

    control = spouses.loc[spouses.D.eq(0), ["pnr", "conviction_date", "convict"]]
    ids = set(pd.concat([control["pnr"], control["convict"]]))

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
    control = control.merge(kraf, on=["convict", "conviction_date"], how="left")

    pairs = pd.MultiIndex.from_frame(control[["pnr", "journr"]])
    flag = pd.MultiIndex.from_frame(control[["convict", "journr"]]).isin(pairs)

    spouses = spouses.loc[lambda d: ~d.pnr.isin(control[flag].pnr)]
    panel = panel.loc[lambda d: d.pnr.isin(spouses.pnr)]

    return panel, spouses


def run(window: iter = range(-18, 19)):
    """Creates panel with event-time periods around conviction month."""

    paths = PATHS
    spouses = pd.read_parquet(paths.temp / "spouses.parquet")
    ids = set(spouses["pnr"])

    panel = _initialize_panel(spouses=spouses, window=window)
    moves = _build_monthly_moves(ids=ids, paths=paths)
    panel = _apply_migration_states(panel=panel, moves=moves)
    panel = _merge_panel_outcomes(panel=panel, paths=paths)
    panel, spouses = _remove_coconvicted_controls(panel=panel, spouses=spouses, paths=paths)

    panel.to_parquet(paths.temp / "panel.parquet")
    spouses.to_parquet(paths.temp / "spouses.parquet")


if __name__ == "__main__":
    run()
