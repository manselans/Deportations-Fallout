"""
Creates tables and figures with main results.
Content: Table 2 and Figures 2 and 3.
"""

from pathlib import Path

import numpy as np
import pandas as pd

import statsmodels.formula.api as smf

from tools.paths import init
from tools.tables import results_table
from tools.plots import dyn_did, raw_rates, did_plot
from tools.formatting import setup_pyplot


def run():
    """
    Creates:
    - Table 2: Static estimations
    - Figure 2: Spouses' Employment: Trends and Estimation Results
    - Figure 3: Other Labour Market Outcomes: Trends and Estimation Results
    """

    # expose paths (and create output folders if they do not exist)
    paths = init(Path.cwd())

    # setup plotting defaults
    setup_pyplot()

    # load data
    panel = pd.read_parquet(paths.data / "panel.parquet")

    # ---- Table 2: Static estimations

    df = (
        panel.assign(
            seniority=lambda d: np.floor((d["seniority"] / 365)).clip(upper=15),
            age=lambda d: d["age"].clip(upper=50),
        )
        .dropna(subset=["partic"])
        .reset_index(drop=True)
    )

    # Make "regressable"
    for c in df.select_dtypes(exclude=[pd.PeriodDtype, "datetime64[ns]"]):
        df[c] = df[c].astype(float)

    res = {}

    # Models 1A, 2A and 3A (A: No controls, B: Controls)
    m = 1
    for p in [6, 12, 18]:
        res[m] = smf.ols("partic ~ D", data=df.loc[df["event_time"] == p]).fit(
            cov_type="HC1"
        )
        m += 2

    # Models 1B, 2B and 3B (A: No controls, B: Controls)
    controls = ["female", "age", "seniority", "origin", "year"]
    rhs = " + ".join(["D"] + [f"C({c})" for c in controls])

    m = 2
    for p in [6, 12, 18]:
        res[m] = smf.ols(f"partic ~ {rhs}", data=df.loc[df["event_time"] == p]).fit(
            cov_type="HC1"
        )
        m += 2

    # Models 4 and 5
    controls = ["D", "female", "age", "seniority", "origin", "year"]
    post = (
        df.loc[df["event_time"] > 0]
        .groupby("pnr", as_index=False)
        .agg({**{"partic": "mean"}, **{c: "last" for c in controls}})
    )

    res[7] = smf.ols("partic ~ D", data=post).fit(cov_type="HC1")

    rhs = " + ".join(["D"] + [f"C({c})" for c in controls if c != "D"])
    res[8] = smf.ols(f"partic ~ {rhs}", data=post).fit(cov_type="HC1")

    # Model 6
    pre = (  # Add pre-period for DiD
        df.loc[df["event_time"] < 0]
        .groupby("pnr", as_index=False)
        .agg({**{"partic": "mean"}, **{c: "last" for c in controls}})
        .assign(period=0)
    )

    did = pd.concat([post.assign(period=1), pre], ignore_index=True)
    did["did"] = did["D"] * did["period"]

    clt = did["pnr"]  # Cluster by individual

    res[9] = smf.ols("partic ~ D + period + did", data=did).fit(
        cov_type="cluster", cov_kwds={"groups": pd.Categorical(clt).codes}
    )

    # Set up table
    var_names = {"did": "Deportation x Post", "D": "Deportation", "nobs": "N"}

    _, tbl = results_table(
        res,
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
    tbl.reindex(order).to_html(paths.tables / "Table 2.html", escape=False)

    # ---- Figure 2: Spouses' Employment: Trends and Estimation Results

    # Estimate
    _, out = dyn_did(df, y="partic")

    # Plot and save
    fig_a, ax_a = raw_rates(df, var="partic", ylabel="Employment")
    fig_b, _ = did_plot(out, ylim=(-0.1, 0.06))

    ax_a.legend(bbox_to_anchor=(0.5, -0.12), ncol=2)

    fig_a.savefig(paths.figures / "Figure 2A.png")
    fig_b.savefig(paths.figures / "Figure 2B.png")

    # ---- Figure 3: Other Labour Market Outcomes: Trends and Estimation Results

    name = [
        "Share of fulltime employment",
        "Pre-tax wage in 1.000 DKK",
        "Total income (wages + transfers) in 1.000 DKK",
    ]
    litra = ["A", "B", "C", "D", "E", "F"]

    df["income"] = df["wages"] + df["transfers"]

    for i, y in enumerate(["fulltime", "wages", "income"], start=1):

        # Estimate
        _, out = dyn_did(df, y=y)

        # Plot and save
        fig_a, ax_a = raw_rates(df, var=y, ylabel=name[i - 1], ylim=None)
        fig_b, _ = did_plot(out, ylim=None)

        ax_a.legend(bbox_to_anchor=(0.5, -0.12), ncol=2)

        fig_a.savefig(paths.figures / f"Figure 3{litra[i*2 - 2]}.png")
        fig_b.savefig(paths.figures / f"Figure 3{litra[i*2 - 1]}.png")


if __name__ == "__main__":
    run()
