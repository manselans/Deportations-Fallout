"""
Create Table B1: Test of Change in Sample Size Around Cutoff.
"""


import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from deportations_fallout.results.tabulate import results_table
from deportations_fallout.config import PATHS

paths = PATHS

def table_b1():
    """
    Create Table B1: Test of Change in Sample Size Around Cutoff.
    """

    # ---- Set up data

    panel_df = pd.read_parquet(paths.temp/"panel.parquet")
    convicts_df = pd.read_parquet(paths.temp/"population.parquet")

    # Conviction cannot be exempt from seniority rule
    vl = ["pnr", "convict", "D", "seniority"]
    ids = set(convicts_df.loc[lambda d: d["exception"].eq(False), "pnr"])
    df = panel_df.loc[lambda d: d["convict"].isin(ids), vl].copy()

    # Keep only seniority 1 year around discontinuity (5 years)
    df["seniority"] = (
        df["seniority"] / 30 - 12 * 5 # seniority in months relative to 5 years
    ).apply(lambda v: np.floor(v) if v < 0 else np.ceil(v))

    df = df.loc[
        lambda d: d["seniority"].between(-12, 12) & d["seniority"].ne(0)
    ].dropna().drop_duplicates()

    df_t = df.groupby("seniority", as_index=False).agg(count=("pnr", "count"))
    df_t["post"] = (df_t["seniority"] >= 0).astype(int)
    df_t["inter"] = df_t["seniority"] * df_t["post"]

    # ---- Estimate

    res = {"": smf.ols("count ~ post + seniority + inter", data=df_t).fit()}

    # ---- Tabulate

    varnames = {"post": "D", "seniority": "Z < 0", "inter": "Z > 0", "nobs": "N"}

    _, tbl = results_table(
        res,
        rows=("post", "seniority", "inter", "Intercept"),
        add_stats=("nobs"),
        var_names=varnames,
        floatfmt="{:.2f}",
        model_names=["Number of observations"],
    )

    tbl.index.name = "Variable"

    # Order rows (N last)
    order = [r for r in tbl.index.tolist() if r != "N"] + ["N"]
    tbl = tbl.reindex(order)

    # ---- McCrary Test (data only)

    # avoid empty 0-bin
    mccrary = df.assign(seniority=lambda d: d.seniority - (d.seniority > 0).astype(int))

    return tbl, mccrary
