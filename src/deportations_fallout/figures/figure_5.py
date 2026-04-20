"""
Create Figure 5: Attrition: Manski Bounds on Main Results.
"""


import pandas as pd
import matplotlib.pyplot as plt

from deportations_fallout.results.dynamic_did import dyn_did, coef_plot

# Configuration
from deportations_fallout.config import PATHS, setup_matplotlib

paths = PATHS
setup_matplotlib()


def figure_5():
    """
    Create Figure 5: Attrition: Manski Bounds on Main Results.
    """

    df = pd.read_parquet(paths.temp/"panel.parquet")

    vl = ["pnr", "D", "event_time", "in_country", "mover", "partic", "month"]
    df = df[vl].copy()

    # ---- Define first migration out of denmark

    df["in_country"] = df["in_country"].astype(int)
    g = df.sort_values(["pnr", "month"]).groupby("pnr")
    leave = g["in_country"].diff().eq(-1)

    df["first_leave"] = (
        leave & ~leave.groupby(df["pnr"]).cummax().shift(fill_value=False)
    ).astype(int)

    df["after_move"] = leave.groupby(df["pnr"]).cumsum().gt(0).astype(int)

    # ---- Define bounds

    df = df.loc[lambda d: d["month"].dt.year.between(2000, 2022)]
    df = df.drop(columns="month")
    df = df.astype(float)

    df = df.assign(
        high=lambda d: d["partic"].mask(d["after_move"].eq(1), 1),
        low=lambda d: d["partic"].mask(d["after_move"].eq(1), 0),
    )

    # ---- Plot

    fig, ax = plt.subplots()

    _, out_all = dyn_did(df.dropna(subset=["high"]), y="high")
    _, out_none = dyn_did(df.dropna(subset=["low"]), y="low")
    _, out_main = dyn_did(df.dropna(subset=["partic"]), y="partic")

    coef_plot(
        out_all,
        ax=ax,
        drop_ci=True,
        plot_kwargs={
            "linestyle": "--",
            "marker": "^",
            "label": "Employed",
            "color": "0.4",
        },
    )
    coef_plot(
        out_none,
        ax=ax,
        drop_ci=True,
        plot_kwargs={
            "linestyle": "--",
            "marker": "v",
            "label": "Unemployed",
            "color": "0.4",
        },
    )
    coef_plot(out_main, ax=ax, drop_ci=True, plot_kwargs={"label": "Missing"})

    ax.set_ylim((-0.1, 0.1))

    # ---- Testing informative attrition

    # Remove pre-entry periods
    df = df.loc[lambda d: d["after_move"].eq(1) | d["partic"].notna()].copy()

    # Define R_t and R_t+1
    df["R"] = df["partic"].notna().astype(int)  # i observed in t
    df["R_lead"] = df.groupby("pnr")["R"].shift(-1)  # i observed in t+1

    # Include lead and estimate...
    res, _ = dyn_did(df, y="partic", controls=["R_lead"])
    print(
        "Informative attrition test: \n",
        f'    Coef.: {res.params["R_lead"].round(3)}; p-value: {res.pvalues["R_lead"].round(3)}',
    )

    return fig, ax
