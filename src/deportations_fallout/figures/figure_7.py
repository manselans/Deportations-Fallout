"""
Create Figure 7: IPCW and kNN.
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from deportations_fallout.results.dynamic_did import dyn_did, coef_plot
from deportations_fallout.results.ipcw import ipcw
from deportations_fallout.results.knn_hd import KNNConfig, knn_hotdeck_impute
from deportations_fallout.config import PATHS, setup_matplotlib, scale_figsize

paths = PATHS
setup_matplotlib()

# Defaults
CAT_FEATS_IPCW = [
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

NUM_FEATS_IPCW = ["age", "seniority", "pre_move_y"]

CAT_FEATS_KNN = ["charged", "convicted", "incarcerated"]


def figure_7():
    """
    Create Figure 7: IPCW and kNN.
    """

    fig, ax = plt.subplots(1, 2, figsize=scale_figsize(nrows=1, ncols=2))
    axes = ax.flatten()

    # ---- Set up data

    df = pd.read_parquet(paths.temp/"panel.parquet")

    for c in df.select_dtypes(exclude=[pd.PeriodDtype, "datetime64[ns]"]):
        df[c] = df[c].astype(float)

    # Define first migration out
    g = df.groupby("pnr")
    leave = g["in_country"].diff().eq(-1)
    df["first_leave"] = leave & ~leave.groupby(df["pnr"]).cummax().shift(
        fill_value=False
    ).astype(int)
    df["after_move"] = leave.groupby(df["pnr"]).cumsum().gt(0)

    # Define covariates
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
    )

    # ---- Figure 7A: IPCW

    df_ipcw = df.copy()

    # Define attrition variables
    df_ipcw["R"] = df_ipcw["partic"].notna().astype(int)  # i observed in t
    df_ipcw["R_lag"] = df_ipcw.groupby("pnr")["R"].shift(1)  # i observed in t-1
    df_ipcw["at_risk"] = (df_ipcw["R_lag"] == 1).astype(int)  # in risk set at t

    df_ipcw["H"] = ((df_ipcw["R_lag"] == 1) & (df_ipcw["R"] == 0)).astype(
        int
    )  # i left in t
    df_ipcw["left"] = (df_ipcw.groupby("pnr")["H"].cumsum() >= 1).astype(
        int
    )  # attritioned out

    df_ipcw = df_ipcw.loc[
        ~(df_ipcw["partic"].isna() & df_ipcw["left"].eq(0))
    ]  # exclude those not yet arrived
    df_ipcw["t_entry"] = df_ipcw["pnr"].map(df.groupby("pnr")["event_time"].min())

    # Risk set
    risk = df_ipcw[df_ipcw["at_risk"] == 1].dropna(subset=["pre_move_y"]).copy()

    # Model setup
    cat_feats = CAT_FEATS_IPCW
    num_feats = NUM_FEATS_IPCW

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

    df_ipcw = df_ipcw.merge(
        risk[["pnr", "event_time", "predicted_hazards"]],
        on=["pnr", "event_time"],
        how="left",
        validate="1:1",
    )

    # Survival
    df_ipcw["period_survival"] = np.where(
        df_ipcw.groupby("pnr").cumcount().gt(0), 1 - df_ipcw["predicted_hazards"], 1
    )
    df_ipcw["S_it"] = df_ipcw.groupby("pnr", sort=False)["period_survival"].cumprod()

    df_ipcw["ipcw"] = np.where(df_ipcw["R"] == 1, 1.0 / df_ipcw["S_it"], 0.0)
    df_ipcw["ipcw"] = df_ipcw["ipcw"].clip(
        lower=df_ipcw["ipcw"].quantile(0.01), upper=df_ipcw["ipcw"].quantile(0.99)
    )

    # Estimate
    _, out = dyn_did(
        df_ipcw.dropna(subset=["partic", "ipcw"]), y="partic", w_col="ipcw"
    )

    # Plot and save
    coef_plot(out, ylim=(-0.1, 0.06), ax=axes[0])
    axes[0].set_title("(a) IPCW", fontstyle="italic")

    # ---- Figure 7B: kNN

    cfg = KNNConfig()
    cfg.stochastic = False
    cfg.distance_categorical = CAT_FEATS_KNN
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        imp = knn_hotdeck_impute(df, cfg)

    df_knn = df.assign(partic_imp=lambda d: d["partic"].fillna(imp["y_imputed"]))

    # Estimate based on imputed values
    _, out = dyn_did(df_knn, y="partic_imp")
    coef_plot(out, ylim=(-0.1, 0.06), ax=axes[1])
    axes[1].set_title("(b) kNN hot-deck imputation", fontstyle="italic")

    return fig, ax
