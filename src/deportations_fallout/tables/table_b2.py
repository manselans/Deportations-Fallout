"""
Create Table B2: Unconditional Balancing Test.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from deportations_fallout.results.tabulate import stars_from_p
from deportations_fallout.config import PATHS

paths = PATHS


def table_b2():
    """
    Create Table B2: Unconditional Balancing Test.
    """

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

    # ---- Estimate

    res = []
    for v in vl:
        m = smf.ols(f"{v} ~ rd + pre + post", data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df.dropna(subset=[v])["seniority"]}
        )
        res.append(
            {
                "Variable": v,
                "D": m.params["rd"],
                "SE": m.bse["rd"],
                "p": m.pvalues["rd"],
                "N": int(m.nobs),
            }
        )

    res = pd.DataFrame(res)

    res["D"] = res["D"].round(3).astype(str) + res["p"].apply(stars_from_p)
    res = res.set_index("Variable").drop(columns=["p"])

    row_names = {
        "age": "Age",
        "female": "Female",
        "children": "Number of children",
        "has_children": "Has children",
        "fulltime": "Av. percent of fulltime employment",
        "wages": "Av. prior earnings",
        "transfers": "Av. prior transfer income",
        "partic": "Av. prior employment rate",
        "assets": "Net assets",
        "convicted": "Convicted",
        "charged": "Charged",
        "incarcerated": "Incarcerated",
    }

    res = res.rename(index=row_names)

    return res
