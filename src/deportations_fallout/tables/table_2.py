"""
Create Table 2: Static Estimations.
"""


import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

from deportations_fallout.results.tabulate import results_table
from deportations_fallout.config import PATHS

paths = PATHS


def table_2():
    """
    Create Table 2: Static Estimations.
    """

    df = pd.read_parquet(paths.temp/"panel.parquet")

    vl = ["pnr", "D", "event_time", "female", "age", "seniority", "origin", "year", "partic"]
    df = df[vl].copy()

    # Round and truncate seniority and age
    df["seniority"] = np.floor((df["seniority"] / 365)).clip(upper=15)
    df["age"] = df["age"].clip(upper=50)

    df = df.astype(float).dropna()

    # ---- Estimation

    results = {}

    # Models 1A, 2A and 3A (A: No controls, B: Controls)
    m = 1
    for p in [6, 12, 18]:
        results[m] = smf.ols("partic ~ D", data=df.loc[df["event_time"] == p]).fit(
            cov_type="HC1"
        )
        m += 2

    # Models 1B, 2B and 3B (A: No controls, B: Controls)
    controls = ["female", "age", "seniority", "origin", "year"]
    rhs = " + ".join(["D"] + [f"C({c})" for c in controls])

    m = 2
    for p in [6, 12, 18]:
        results[m] = smf.ols(f"partic ~ {rhs}", data=df.loc[df["event_time"] == p]).fit(
            cov_type="HC1"
        )
        m += 2

    # Models 4 and 5: Post mean
    controls = ["D", "female", "age", "seniority", "origin", "year"]
    post = (
        df.loc[df["event_time"] > 0]
        .groupby("pnr", as_index=False)
        .agg({**{"partic": "mean"}, **{c: "last" for c in controls}})
    )

    results[7] = smf.ols("partic ~ D", data=post).fit(cov_type="HC1")

    rhs = " + ".join(["D"] + [f"C({c})" for c in controls if c != "D"])
    results[8] = smf.ols(f"partic ~ {rhs}", data=post).fit(cov_type="HC1")

    # Model 6: DiD
    pre = (  # Add pre-period for DiD
        df.loc[df["event_time"] < 0]
        .groupby("pnr", as_index=False)
        .agg({**{"partic": "mean"}, **{c: "last" for c in controls}})
        .assign(period=0)
    )

    did = pd.concat([post.assign(period=1), pre], ignore_index=True)
    did["did"] = did["D"] * did["period"]

    clt = did["pnr"]  # Cluster by individual

    results[9] = smf.ols("partic ~ D + period + did", data=did).fit(
        cov_type="cluster", cov_kwds={"groups": pd.Categorical(clt).codes}
    )

    # ---- Tabulate

    # Set up table
    var_names = {"did": "Deportation x Post", "D": "Deportation", "nobs": "N"}

    _, tbl = results_table(
        results,
        rows=("Intercept", "D", "did"),
        add_stats=("nobs"),
        var_names=var_names,
        floatfmt="{:.2f}",
    )

    # Add explanatory rows
    cntrl = pd.DataFrame(
        {
            "Model 1": ["", "t = 6", "CS"],
            "Model 2": ["&check;", "t = 6", "CS"],
            "Model 3": ["", "t = 12", "CS"],
            "Model 4": ["&check;", "t = 12", "CS"],
            "Model 5": ["", "t = 18", "CS"],
            "Model 6": ["&check;", "t = 18", "CS"],
            "Model 7": ["", "t > 0", "CS"],
            "Model 8": ["&check;", "t > 0", "CS"],
            "Model 9": ["", "DiD", "Panel"],
        },
        index=["Controls", "Note", "Data"],
    )

    tbl = pd.concat([tbl, cntrl])

    # Order rows (N last)
    order = [r for r in tbl.index.tolist() if r != "N"] + ["N"]
    tbl = tbl.reindex(order)

    return tbl
