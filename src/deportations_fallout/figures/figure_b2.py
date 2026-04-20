"""
Create Figure B2: Spouses' Employment: Regression Discontinuity.
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

from deportations_fallout.results.dynamic_did import raw_rates, coef_plot
from deportations_fallout.config import PATHS, setup_matplotlib, scale_figsize

paths = PATHS
setup_matplotlib()


def figure_b2():
    """
    Create Figure B2: Spouses' Employment: Regression Discontinuity.
    """

    # ---- Set up data

    panel_df = pd.read_parquet(paths.temp/"panel.parquet")
    convicts_df = pd.read_parquet(paths.temp/"population.parquet")

    # Conviction cannot be exempt from seniority rule
    vl = ["pnr", "convict", "D", "seniority", "partic", "event_time", "year"]
    ids = set(convicts_df.loc[lambda d: d["exception"].eq(False), "pnr"])
    df = panel_df.loc[lambda d: d["convict"].isin(ids), vl].copy()

    # Keep only seniority 1 year around discontinuity (5 years)
    df["seniority"] = (
        df["seniority"] / 30 - 12 * 5 # seniority in months relative to 5 years
    ).apply(lambda v: np.floor(v) if v < 0 else np.ceil(v))

    df = df.loc[
        lambda d: d["seniority"].between(-12, 12) & d["seniority"].ne(0)
    ].dropna().drop_duplicates()

    # RD variables
    df["post"] = (df["seniority"] >= 0).astype(int)
    df["inter"] = df["seniority"] * df["post"]

    # ---- Plot

    fig, ax = plt.subplots(1, 2, figsize=scale_figsize(nrows=1, ncols=2))
    axes = ax.flatten()

    # Panel A: Raw rates
    raw_rates(df, var="partic", ylabel="Employment", ylim=None, ax=axes[0])
    # axes[0].set_title("(a) Raw means", fontstyle="italic")
    axes[0].legend(bbox_to_anchor=(0.5, -0.12), ncol=2)

    # Panel B: Regression results
    # First stage regression
    fs = smf.ols("D ~ seniority + post + inter + C(year)", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["seniority"]}
    )

    df["yhat"] = fs.predict()
    df["partic"] = df["partic"].astype(float)

    # Second stage
    results = []
    for t in range(df.event_time.min(), df.event_time.max() + 1):
        df_t = df.loc[df["event_time"] == t].copy()
        reg = smf.ols("partic ~ seniority + inter + yhat + C(year)", data=df_t).fit(
            cov_type="cluster", cov_kwds={"groups": df_t["seniority"]}
        )
        results.append({
            "period": t,
            "estimate": reg.params["yhat"],
            "lower": reg.conf_int(alpha=0.1).loc["yhat", 0],
            "upper": reg.conf_int(alpha=0.1).loc["yhat", 1],
        })

    coef_plot(pd.DataFrame(results), ylim=None, ax=axes[1])
    # axes[1].set_title("(b) Regression results: RD", fontstyle="italic")

    axes[1].set_yticks(axes[1].get_yticks().tolist())
    axes[1].set_yticklabels([f"{t:.1f}" for t in axes[1].get_yticks()])

    return fig, ax
