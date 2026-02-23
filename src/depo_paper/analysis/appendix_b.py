"""
Creates figures and tables for Appendix B.
Herunder Figures B1 and B2 and Tables B1, B2 and B3
"""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

import statsmodels.formula.api as smf

from depo_paper.config import PATHS
from depo_paper.tools.formatting import setup_pyplot
from depo_paper.tools.plots import raw_rates, did_plot
from depo_paper.tools.tables import results_table, stars_from_p


def run():
    """
    Creates:
    - Figure B1: Deportation Risk by Residence Seniority
    - Table B1: Test of Change in Sample Size Around Cutoff
    - Table B2: Unconditional Balancing Test
    - Table B3: Conditional Balancing Test
    - Figure B2: Spouses' Employment: Regression Discontinuity
    """

    # expose paths (and create output folders if they do not exist)
    paths = PATHS

    # setup plotting defaults
    setup_pyplot()

    # load data
    panel = pd.read_parquet(paths.temp / "panel.parquet")
    population = pd.read_parquet(paths.temp / "population.parquet")

    # ---- Figure B1: Deportation Risk by Residence Seniority

    vl = ["pnr", "convict", "D", "seniority"]

    df = (
        panel[vl]
        # conviction cannot be exception to seniority rule
        .loc[
            lambda d: d["convict"].isin(
                population.loc[lambda d: d["exception"].eq(False), "pnr"]
            )
        ]
        # drop those missing seniority and keep only seniority 1 year around discontinuity (5 years)
        .assign(
            seniority=lambda d: (d["seniority"] / 30 - 12 * 5).apply(
                lambda v: np.floor(v) if v < 0 else np.ceil(v)
            )
        )
        .loc[lambda d: d.seniority.between(-12, 12) & d.seniority.ne(0)]
        .dropna()
        .drop_duplicates()
        .copy()
    )

    # shape data
    df_t = (
        df.assign(  # bin seniority
            seniority=lambda d: np.where(
                d.seniority < 0, (d.seniority // 2) * 2, ((d.seniority + 1) // 2) * 2
            )
        )
        .groupby("seniority")["D"]
        .agg(["mean", "count"])
        .reset_index()
        .loc[lambda d: d["count"] >= 3]  # discretion
    )

    left = df[df["seniority"] < 0]
    right = df[df["seniority"] > 0]

    b_left = np.polyfit(left["seniority"], left["D"], 1)
    b_right = np.polyfit(right["seniority"], right["D"], 1)

    x_left = [-12, -1]
    y_left = np.polyval(b_left, x_left)

    x_right = [1, 12]
    y_right = np.polyval(b_right, x_right)

    # plot
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.axvline(0, color="red", ls="--")
    ax.scatter(df_t["seniority"], df_t["mean"], color=".4")
    ax.plot(x_left, y_left, lw=2, color=".6")
    ax.plot(x_right, y_right, lw=2, color=".6")

    # set axes
    ax.set_xticks(np.arange(-12, 13, 2))

    ax.tick_params(axis="y", rotation=90)
    ax.set_ylim(-0.01, 0.25)
    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda x, _: (
                "0"
                if abs(x) < 1e-12
                else f"{x:.2f}".replace("0.", ".").replace("-0.", "-.")
            )
        )
    )

    ax.set_ylabel("Share with deportation order")
    ax.set_xlabel("Conviction month relative to 5 years of seniority")

    fig.tight_layout()
    fig.savefig(paths.figures / "Figure B1.png")

    # ---- Table B1: Test of Change in Sample Size Around Cutoff

    df_t = (
        df.groupby("seniority", as_index=False)
        .agg(count=("pnr", "count"))
        .assign(post=lambda d: (d.seniority > 0).astype(int))
        .assign(inter=lambda d: d.post * d.seniority)
    )

    # estimate
    res = {"": smf.ols("count ~ post + seniority + inter", data=df_t).fit()}

    # tabulate
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
    tbl.reindex(order).to_html(paths.tables / "Table B1.html", escape=False)

    # ---- McCrary Test (data only)

    # avoid empty 0-bin
    df_t = df.assign(seniority=lambda d: d.seniority - (d.seniority > 0).astype(int))
    df_t.to_stata(paths.temp / "mccrary.dta", write_index=False)

    # ---- Table B2: Unconditional Balancing Test

    df = (
        panel
        # conviction cannot be exception to seniority rule
        .loc[
            lambda d: d["convict"].isin(
                population.loc[lambda d: d["exception"].eq(False), "pnr"]
            )
        ]
        # drop those missing seniority and keep only seniority 1 year around discontinuity (5 years)
        .assign(
            seniority=lambda d: (d["seniority"] / 30 - 12 * 5).apply(
                lambda v: np.floor(v) if v < 0 else np.ceil(v)
            )
        )
        .loc[lambda d: d.seniority.between(-12, 12) & d.seniority.ne(0)]
        # drop post-conviction periods and collapse
        .loc[lambda d: d.event_time < 0]
        .groupby("pnr", as_index=False)
        .agg(
            {
                **{
                    c: "max"
                    for c in [
                        "D",
                        "age",
                        "children",
                        "female",
                        "charged",
                        "convicted",
                        "incarcerated",
                        "seniority",
                    ]
                },
                **{
                    c: "mean"
                    for c in ["partic", "fulltime", "wages", "transfers", "assets"]
                },
            }
        )
        .copy()
    )

    df["rd"] = (df.seniority > 0).astype(int)
    df["pre"] = (1 - df.rd) * df.seniority
    df["post"] = df.rd * df.seniority
    df["has_children"] = (df.children > 0).astype(int)

    vl = ["age", "female", "children", "has_children"]
    vl = vl + ["charged", "convicted", "incarcerated"]
    vl = vl + ["partic", "fulltime", "wages", "transfers", "assets"]

    # make regressable
    for c in vl:
        df[c] = df[c].astype(float)

    # estimate
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

    # Save
    res.to_html(paths.tables / "Table B2.html", escape=False)

    # ---- Table B3: Conditional Balancing Test

    vl_1 = ["age", "female", "children", "has_children"]
    vl_2 = ["charged", "convicted", "incarcerated"]
    vl_3 = ["partic", "fulltime", "wages", "transfers", "assets"]

    cov_sets = [vl_1, vl_1 + vl_2, vl_1 + vl_2 + vl_3]

    # estimate
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
    tbl.reindex(order).to_html(paths.tables / "Table B3.html", escape=False)

    # ---- Figure B2: Spouses' Employment: Regression Discontinuity

    vl = ["pnr", "convict", "D", "seniority", "partic", "event_time", "year"]

    df = (
        panel
        # conviction cannot be exception to seniority rule
        .loc[
            lambda d: d["convict"].isin(
                population.loc[lambda d: d["exception"].eq(False), "pnr"]
            ),
            vl,
        ]
        # drop those missing seniority and keep only seniority 1 year around discontinuity (5 years)
        .assign(
            seniority=lambda d: (d["seniority"] / 30 - 12 * 5).apply(
                lambda v: np.floor(v) if v < 0 else np.ceil(v)
            )
        )
        .loc[lambda d: d.seniority.between(-12, 12) & d.seniority.ne(0)]
        .dropna()
        .copy()
    )

    # RD variables
    df["post"] = (df.seniority > 0).astype(int)
    df["inter"] = df.post * df.seniority

    fig, ax = raw_rates(df, var="partic", ylabel="Employment", ylim=None)
    fig.savefig(paths.figures / "Figure B2A.png")

    # first stage regression
    fs = smf.ols("D ~ seniority + post + inter + C(year)", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["seniority"]}
    )

    df["yhat"] = fs.predict()
    df["partic"] = df["partic"].astype(float)

    # second stage
    results = []
    for t in range(df.event_time.min(), df.event_time.max() + 1):
        df_t = df.loc[df["event_time"] == t].copy()
        reg = smf.ols("partic ~ seniority + inter + yhat + C(year)", data=df_t).fit(
            cov_type="cluster", cov_kwds={"groups": df_t["seniority"]}
        )
        results.append(
            {
                "period": t,
                "estimate": reg.params["yhat"],
                "lower": reg.conf_int(alpha=0.1).loc["yhat", 0],
                "upper": reg.conf_int(alpha=0.1).loc["yhat", 1],
            }
        )

    fig, ax = did_plot(pd.DataFrame(results), ylim=None)

    ax.set_yticks(ax.get_yticks().tolist())
    ax.set_yticklabels([f"{t:.1f}" for t in ax.get_yticks()])

    fig.savefig(paths.figures / "Figure B2B.png")


if __name__ == "__main__":
    run()
