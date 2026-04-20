"""
Create Table B3: Conditional Balancing Test.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from deportations_fallout.results.tabulate import results_table
from deportations_fallout.config import PATHS

paths = PATHS

# Defaults
VSET_1 = ["age", "female", "children", "has_children"]
VSET_2 = ["charged", "convicted", "incarcerated"]
VSET_3 = ["partic", "fulltime", "wages", "transfers", "assets"]


def table_b3():
    """
    Create Table B3: Conditional Balancing Test.
    """

    cov_sets = [VSET_1, VSET_1 + VSET_2, VSET_1 + VSET_2 + VSET_3]

    # ---- Set up data

    panel_df = pd.read_parquet(paths.temp/"panel.parquet")
    convicts_df = pd.read_parquet(paths.temp/"population.parquet")

    # Conviction cannot be exempt from seniority rule
    ids = set(convicts_df.loc[lambda d: d["exception"].eq(False), "pnr"])
    df = panel_df.loc[lambda d: d["convict"].isin(ids)].copy()

    # Keep only seniority 1 year around discontinuity (5 years)
    df["seniority"] = (
        df["seniority"] / 30 - 12 * 5  # seniority in months relative to 5 years
    ).apply(lambda v: np.floor(v) if v < 0 else np.ceil(v))

    df = df.loc[lambda d: d["seniority"].between(-12, 12) & d["seniority"].ne(0)]

    # Drop post-conviction periods and collapse
    c_max = [
        "D",
        "age",
        "children",
        "female",
        "charged",
        "convicted",
        "incarcerated",
        "seniority",
    ]
    c_mean = ["partic", "fulltime", "wages", "transfers", "assets"]

    df = df.loc[lambda d: d["event_time"] < 0]
    df = df.groupby("pnr").agg(
        {**{c: "max" for c in c_max}, **{c: "mean" for c in c_mean}}
    )

    df["rd"] = (df["seniority"] > 0).astype(int)
    df["pre"] = (1 - df["rd"]) * df["seniority"]
    df["post"] = df["rd"] * df["seniority"]
    df["has_children"] = (df["children"] > 0).astype(int)

    vl = ["age", "female", "children", "has_children"]
    vl = vl + ["charged", "convicted", "incarcerated"]
    vl = vl + ["partic", "fulltime", "wages", "transfers", "assets"]

    for c in vl:
        df[c] = df[c].astype(float)

    # --- Estimate

    res = {}
    for i, cov in enumerate(cov_sets, start=1):
        formula = "rd ~ pre + post + " + " + ".join(cov)
        res[i] = smf.ols(formula, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df.dropna(subset=cov)["seniority"]}
        )

        # test significance of covariates
        hypothesis = [f"{v} = 0" for v in cov]
        ft = res[i].f_test(hypothesis)

        res[i].partial_f = float(np.asarray(ft.fvalue))
        res[i].partial_p = float(np.asarray(ft.pvalue))

    # tabulate
    varnames = {
        "pre": "Z_pre",
        "post": "Z_post",
        "age": "Age",
        "female": "Female",
        "children": "Number of children",
        "has_children": "Has children",
        "convicted": "Convicted",
        "charged": "Charged",
        "incarcerated": "Incarcerated",
        "fulltime": "Av. percent of fulltime employment",
        "wages": "Av. prior earnings",
        "transfers": "Av. prior transfer income",
        "partic": "Av. prior employment rate",
        "assets": "Net assets",
        "nobs": "N",
        "partial_f": "F-test",
        "partial_p": "p-value",
    }
    rows = [r for r in varnames if r not in ["nobs", "partial_f", "partial_p"]]

    _, tbl = results_table(
        res,
        rows=rows + ["Intercept"],
        add_stats=("nobs", "partial_f", "partial_p"),
        var_names=varnames,
        floatfmt="{:.2f}",
    )

    tbl.index.name = "Variable"

    # Order rows (N last)
    order = [r for r in tbl.index.tolist() if r != "N"] + ["N"]
    tbl = tbl.reindex(order)

    return tbl
