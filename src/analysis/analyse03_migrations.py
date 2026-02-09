"""
Creates figures with migration/attrition results.
Content: Figures 4, 5, 6 and 7.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from tools.paths import init
from tools.formatting import setup_pyplot
from tools.plots import dyn_did, raw_rates, did_plot
from tools.ipcw import ipcw
from tools.knn_hd import knn_hotdeck_impute, KNNConfig


def run():
    """
    Creates:
    - Figure 4: Migrations out of Denmark
    - Figure 5: Attrition: Manski Bounds on Main Results
    - Figure 6: Attrition: Spouses' Labour Market Outcomes, ATT-Rs
    - Figure 7: Estimation Results, Spouses' Employment: IPCW and kNN hot-deck
    """

    # expose paths (and create output folders if they do not exist)
    paths = init(Path.cwd())

    # setup plotting defaults
    setup_pyplot()

    # load data
    panel = pd.read_parquet(paths.data / "panel.parquet")

    # ---- Figure 4: Migrations out of Denmark

    df = panel.assign(ooc=lambda d: 1 - d.in_country.astype(int))

    fig, ax = raw_rates(df, var="ooc", ylabel="Share out of country", ylim=(0, 0.13))

    ax.axvline(-7, color="0.4", linestyle="--")
    ax.legend(bbox_to_anchor=(0.5, -0.12), ncol=2)

    fig.savefig(paths.figures / "Figure 4.png", dpi=500, bbox_inches="tight")

    # ---- Figure 5: Attrition: Manski Bounds on Main Results

    df = (
        panel.loc[
            lambda d: d.month.dt.year.between(2008, 2022),
            ["pnr", "D", "event_time", "in_country", "mover", "partic"],
        ]
        .sort_values(["pnr", "event_time"])
        .reset_index(drop=True)
    )

    # Make "regressable"
    for c in df.select_dtypes(exclude=[pd.PeriodDtype]):
        df[c] = df[c].astype("float64")

    # define first migration out of denmark
    g = df.groupby("pnr")
    leave = g["in_country"].diff().eq(-1)

    df["first_leave"] = (
        leave & ~leave.groupby(df["pnr"]).cummax().shift(fill_value=False)
    ).astype(int)

    df["after_move"] = leave.groupby(df["pnr"]).cumsum().gt(0).astype(int)

    # define bounds
    df = df.assign(
        high=lambda d: d["partic"].mask(d["after_move"].eq(1), 1),
        low=lambda d: d["partic"].mask(d["after_move"].eq(1), 0),
    )

    # plot
    fig, ax = plt.subplots(figsize=(8, 6))

    _, out_all = dyn_did(df.dropna(subset=["high"]), y="high")
    _, out_none = dyn_did(df.dropna(subset=["low"]), y="low")
    _, out_main = dyn_did(df.dropna(subset=["partic"]), y="partic")

    did_plot(
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
    did_plot(
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
    did_plot(out_main, ax=ax, drop_ci=True, plot_kwargs={"label": "Missing"})

    ax.set_ylim((-0.1, 0.1))

    fig.savefig(paths.figures / "Figure 5.png", dpi=500, bbox_inches="tight")

    # ---- Testing Informative Attrition

    df = (
        panel.loc[
            lambda d: d.month.dt.year.between(2008, 2022),
            ["pnr", "D", "event_time", "in_country", "mover", "partic"],
        ]
        .sort_values(["pnr", "event_time"])
        .reset_index(drop=True)
    )

    # Make "regressable"
    for c in df.select_dtypes(exclude=[pd.PeriodDtype]):
        df[c] = df[c].astype("float64")

    # Remove pre-entry periods
    g = df.groupby("pnr")
    leave = g["in_country"].diff().eq(-1)

    df["first_leave"] = (
        leave & ~leave.groupby(df["pnr"]).cummax().shift(fill_value=False)
    ).astype(int)
    df["after_move"] = leave.groupby(df["pnr"]).cumsum().gt(0).astype(int)

    df = df.loc[df["after_move"] | df["partic"].notna()].copy()

    # Define R_t and R_t+1
    df["R"] = df["partic"].notna().astype(int)  # i observed in t
    df["R_lead"] = df.groupby("pnr")["R"].shift(-1)  # i observed in t+1

    # Include lead and estimate...
    res, _ = dyn_did(df, y="partic", controls=["R_lead"])
    print(
        "Informative attrition test: \n",
        f'    Coef.: {res.params["R_lead"].round(3)}; p-value: {res.pvalues["R_lead"].round(3)}',
    )

    # ---- Figure 6: Attrition: Spouses' Labour Market Outcomes, ATT-Rs

    df = panel.loc[lambda d: ~d.mover].reset_index(drop=True).copy()  # Stayers only

    # Make "regressable"
    for c in df.select_dtypes(exclude=[pd.PeriodDtype, "datetime64[ns]"]):
        df[c] = df[c].astype(float)

    name = [
        "Employment",
        "Share of fulltime employment",
        "Pre-tax wage in 1.000 DKK",
        "Total income (wages + transfers) in 1.000 DKK",
    ]
    litra = ["A", "B", "C", "D", "E", "F", "G", "H"]

    df["income"] = df["wages"] + df["transfers"]

    for i, y in enumerate(["partic", "fulltime", "wages", "income"], start=1):

        # Estimate
        _, out = dyn_did(df, y=y)

        # Plot and save
        fig_a, ax_a = raw_rates(df, var=y, ylabel=name[i - 1], ylim=None)
        fig_b, _ = did_plot(out, ylim=None)

        ax_a.legend(bbox_to_anchor=(0.5, -0.12), ncol=2)

        fig_a.savefig(paths.figures / f"Figure 6{litra[i*2 - 2]}.png")
        fig_b.savefig(paths.figures / f"Figure 6{litra[i*2 - 1]}.png")

    # ---- Figure 7A: IPCW

    vl = [
        "pnr",
        "D",
        "year",
        "ie_type",
        "female",
        "age",
        "children",
        "in_country",
        "partic",
        "event_time",
        "month",
        "convicted",
        "charged",
        "incarcerated",
    ]
    df = panel[vl].sort_values(["pnr", "event_time"]).copy()

    # make "regressable"
    for c in df.select_dtypes(exclude=[pd.PeriodDtype]):
        df[c] = df[c].astype("float64")

    # define first migration out
    g = df.groupby("pnr")
    leave = g["in_country"].diff().eq(-1)
    df["first_leave"] = leave & ~leave.groupby(df["pnr"]).cummax().shift(
        fill_value=False
    ).astype(int)
    df["after_move"] = leave.groupby(df["pnr"]).cumsum().gt(0)

    df = (
        df.assign(
            has_children=lambda d: d["children"].ge(1),
            pre_move_y=lambda d: d.groupby("pnr")["partic"].transform(
                lambda x: x[~d.loc[x.index, "after_move"]].mean()
            ),
            seniority=lambda d: d.groupby("pnr")["in_country"].transform(
                lambda x: x[d.loc[x.index, "event_time"].lt(0)].mean()
            ),
            charged=lambda d: d["charged"].cumsum().gt(0),
            convicted=lambda d: d["convicted"].cumsum().gt(0),
            incarcerated=lambda d: d["incarcerated"].cumsum().gt(0),
        )
        .loc[lambda d: d["month"].dt.year.between(2008, 2021)]
        .drop(columns=["month"])
        .reset_index(drop=True)
        .astype(float)
        .copy()
    )

    # define attrition variables
    df["R"] = df["partic"].notna().astype(int)  # i observed in t
    df["R_lag"] = df.groupby("pnr")["R"].shift(1)  # i observed in t-1
    df["at_risk"] = (df["R_lag"] == 1).astype(int)  # in risk set at t

    df["H"] = ((df["R_lag"] == 1) & (df["R"] == 0)).astype(int)  # i left in t
    df["left"] = (df.groupby("pnr")["H"].cumsum() >= 1).astype(int)  # attritioned out

    df = df.loc[
        ~(df["partic"].isna() & df["left"].eq(0))
    ]  # exclude those not yet arrived
    df["t_entry"] = df["pnr"].map(df.groupby("pnr")["event_time"].min())

    #  risk set
    risk = df[df["at_risk"] == 1].dropna(subset=["pre_move_y"]).copy()

    cat_feats = [
        "D",
        "event_time",
        "ie_type",
        "female",
        "has_children",
        "year",
        "charged",
        "convicted",
        "incarcerated",
    ]
    num_feats = ["age", "seniority", "pre_move_y"]

    # setup for logit estimator
    model_kwargs = {"penalty": "l2", "C": 100, "solver": "lbfgs", "max_iter": 5000}

    # predict hazards
    risk, _ = ipcw(
        risk,
        hazard_col="H",
        cat_features=cat_feats,
        num_features=num_feats,
        model="logit",
        model_kwargs=model_kwargs,
    )

    df = df.merge(
        risk[["pnr", "event_time", "predicted_hazards"]],
        on=["pnr", "event_time"],
        how="left",
        validate="1:1",
    )

    # survival
    df["period_survival"] = np.where(
        df.groupby("pnr").cumcount().gt(0), 1 - df["predicted_hazards"], 1
    )
    df["S_it"] = df.groupby("pnr", sort=False)["period_survival"].cumprod()

    df["ipcw"] = np.where(df["R"] == 1, 1.0 / df["S_it"], 0.0)
    df["ipcw"] = df["ipcw"].clip(
        lower=df["ipcw"].quantile(0.01), upper=df["ipcw"].quantile(0.99)
    )

    # Estimate
    _, out = dyn_did(df.dropna(subset=["partic", "ipcw"]), y = "partic", w_col = "ipcw")

    # Plot and save
    fig, ax = did_plot(out, ylim = (-.1, .06))
    fig.savefig(paths.figures/"Figure 7A.png")

    # ---- Figure 7B: K-Nearest-Neighbours Imputation

    vl = [
        "pnr",
        "D",
        "year",
        "ie_type",
        "female",
        "age",
        "children",
        "in_country",
        "partic",
        "month",
        "event_time",
        "convicted",
        "charged",
        "incarcerated",
    ]

    df = (
        panel.loc[lambda d: d.month.dt.year.between(2008, 2021), vl]
        .sort_values(["pnr", "event_time"])
        .reset_index(drop=True)
        .copy()
    )

    # make "regressable"
    for c in df.select_dtypes(exclude=[pd.PeriodDtype]):
        df[c] = df[c].astype("float64")

    # determine pre-/post-move periods
    g = df.groupby("pnr")
    leave = g["in_country"].diff().eq(-1)

    df["first_leave"] = (
        leave & ~leave.groupby(df["pnr"]).cummax().shift(fill_value=False)
    ).astype(int)

    df["after_move"] = leave.groupby(df["pnr"]).cumsum().gt(0)

    # define variables
    df = df.assign(
        pre_move_y=lambda d: d.groupby("pnr")["partic"].transform(
            lambda x: x[~d.loc[x.index, "after_move"]].mean()
        ),
        seniority=lambda d: d.groupby("pnr")["in_country"].transform(
            lambda x: x[d.loc[x.index, "event_time"].lt(0)].mean()
        ),
        has_children=lambda d: d["children"].ge(1),
        charged=lambda d: d["charged"].cumsum().gt(0),
        convicted=lambda d: d["convicted"].cumsum().gt(0),
        incarcerated=lambda d: d["incarcerated"].cumsum().gt(0),
    )

    # impute
    cfg = KNNConfig()
    cfg.stochastic = False
    cfg.distance_categorical = ["charged", "convicted", "incarcerated"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        imp = knn_hotdeck_impute(df, cfg)

    df = df.assign(partic_imp=lambda d: d["partic"].fillna(imp["y_imputed"]))

    # estimate based on imputed values
    _, out = dyn_did(df, y="partic_imp")
    fig, ax = did_plot(out, ylim=(-0.1, 0.06))

    fig.savefig(paths.figures / "Figure 7B.png")


if __name__ == "__main__":
    run()
