"""
Functions to generate the main figures in the paper:
- Period means
- Dynamic DiD regression + coefficient plot
"""


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from linearmodels.panel import PanelOLS

# ---- Defaults

REFERENCE_PERIOD: int = -7
ALPHA: float = .1 # significance level
WINDOW: iter = range(-18, 19)


def raw_rates(
    df: pd.DataFrame,
    var: str,
    *,
    event_time: str = "event_time",
    group: str = "D",
    labels: dict = None,
    window: iter = WINDOW,
    disc_lvl: int = 5,
    ylabel: str = None,
    ax=None,
    xjump: int = 3,
    ylim: tuple = (0.3, 0.5),
    xlab: str = "Months since conviction",
    fig_kwargs: dict = None,
):
    """
    Plots raw means of a variable across event time by group.

    Parameters
    ----------
    df
        Input data containing outcome variable `var` (y-axis)
        and event time variable `event_time` (x-axis).
    var
        Name of outcome variable (y-axis).
    event_time
        Name of event time variable (x-axis).
    group
        Name of grouping variable.
    labels
        Labels for values of grouping variable.
    window
        Event time periods to be included.
    disc_lvl
        Discretion level. Minimum number of observations allowed.
    ylabel
        y-axis title
    ax
        Axis to plot graph on. If None, a new axis is provided.
    xjump
        Distance between x-ticks and labels.
    ylim
        Limit of y-axis.
    xlab
        x-axis title.
    """

    if labels is None:
        labels = {0: "Conviction only", 1: "Conviction and deportation order"}

    # ---- Set up data

    df = (
        df
        .loc[lambda d: d[event_time].isin(window), [event_time, var, group]]
        # Hygiene
        .assign(
            **{
                c: pd.to_numeric(df[c], errors="coerce")
                for c in [event_time, var, group]
            }
        )
        .dropna()
        # Group-period means
        .groupby([group, event_time], sort=True)[var]
        .agg(mean="mean", n="count")
        .reset_index()
        .copy()
    )

    # Discretion
    df.loc[df["n"] < disc_lvl, ["mean", "n"]] = np.nan

    # Reshape
    df = df.pivot(index=event_time, columns=group).sort_index()

    # ---- Create figure/axes

    if ax is None:
        if fig_kwargs is None:
            fig_kwargs = {}

        fig, ax = plt.subplots(**fig_kwargs)
    else:
        fig = ax.figure

    # ---- Plot

    ax.axvline(0, linestyle="--", color="red")

    for tval in df["mean"].columns:
        y = df["mean"][tval]
        if tval == df["mean"].columns[0]:
            ax.plot(
                df.index, y, marker="o", color=".6", label=labels.get(tval, str(tval))
            )
        else:
            ax.plot(
                df.index, y, marker="o", color=".3", label=labels.get(tval, str(tval))
            )

    ax.set_xlim((df.index.min(), df.index.max()))
    ax.set_ylim(ylim)
    ax.set_xlabel(xlab)

    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.set_xticks(np.arange(min(window), max(window) + 1, xjump))
    ax.tick_params(axis="y", rotation=90)

    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda x, _: (
                "0"
                if abs(x) < 1e-12
                else f"{x:.2f}".replace("0.", ".").replace("-0.", "-.")
            )
        )
    )

    return fig, ax


def dyn_did(
    df: pd.DataFrame,
    y: str,
    *,
    id_col: str = "pnr",
    treatment: str = "D",
    event_time: str = "event_time",
    window: iter = WINDOW,
    time_of_reference: int = REFERENCE_PERIOD,
    alpha: float = ALPHA,
    controls: list = None,
    w_col: str = None,
):
    """
    Estimates a dynamic DiD with optional controls.
    Std. errors are clustered based on panel unit id, `id_col`.

    Parameters
    ----------
    df
        Input data.
    y
        Name of outcome variable.
    id_col
        Name of panel unit variable.
    treatment
        Name of treatment indicator variable.
    event_time
        Name of event time variable.
    window
        Event time periods to be included.
    time_of_reference
        Reference period to be excluded form regression model.
    alpha
        Significance level for confidence intervals.
    control
        List of control variables.
    w_col
        Name of column with regression weigths.
        If None, regression is unweighted.

    Returns
    -------
    res
        Regression results from linearmodels.PanelOLS
    out
        pandas.DataFrame of event time coefficients
    """

    controls = list(controls) if controls is not None else []
    w_col = [w_col] if w_col is not None else []

    # Hygiene
    need = [id_col, treatment, event_time, y] + controls + w_col
    must_be_numeric = [y, event_time, treatment] + controls + w_col
    df = (
        df[need]
        .assign(**{c: pd.to_numeric(df[c], errors="coerce") for c in must_be_numeric})
        .dropna()
        .copy()
    )
    df = df.loc[lambda d: d[event_time].isin(window)]

    # Build event-time and interaction dummies (drop reference period in both)
    periods_all = np.sort(df[event_time].unique())
    keep_periods = [p for p in periods_all if p != time_of_reference]

    i_cols, t_cols = [], []
    for p in keep_periods:
        ci = f"I:{p}"
        ct = f"T:{p}"
        df[ci] = (df[event_time] == p).astype(int)
        df[ct] = df[ci] * df[treatment]
        i_cols.append(ci)
        t_cols.append(ct)

    x_cols = i_cols + t_cols + controls

    df = df.set_index([id_col, event_time]).sort_index()
    y = df[y]
    x_mat = df[x_cols]

    # Regress
    if w_col:
        w = df[w_col[0]]
        res = PanelOLS(y, x_mat, entity_effects=True, weights=w).fit(
            cov_type="clustered", cluster_entity=True
        )
    else:
        res = PanelOLS(y, x_mat, entity_effects=True).fit(
            cov_type="clustered", cluster_entity=True
        )

    # create output dataframe
    out = pd.DataFrame({"period": np.sort(np.append(keep_periods, time_of_reference))})
    out["estimate"] = np.nan
    out["lower"] = np.nan
    out["upper"] = np.nan

    # reference period set to 0
    out.loc[out["period"] == time_of_reference, "estimate"] = 0.0

    # fill in the rest
    ci_tbl = res.conf_int(level=1 - alpha)
    for p in keep_periods:
        name = f"T:{p}"
        if name in res.params.index:
            out.loc[out["period"] == p, "estimate"] = res.params[name]
            out.loc[out["period"] == p, ["lower", "upper"]] = ci_tbl.loc[name].values

    out = out.reset_index(drop=True)

    return res, out


def coef_plot(
    df: pd.DataFrame,
    *,
    ylab="Estimated effect",
    xlab="Months since conviction",
    time_of_event: int = 0,
    xjump: int = 3,
    ylim: tuple = (-0.1, 0.05),
    drop_ci: bool = False,
    ax=None,
    fig_kwargs: dict = None,
    plot_kwargs: dict = None,
):
    """
    Plots event time coefficients recieved from dyn_did.

    Parameters
    ----------
    df
        Input data. Should be output dataframe from dyn_did().
    ylab
        y-axis title.
    xlab
        x-axis title.
    time_of_event
        Event time to put vertical line at.
    xjump
        Distance between x-ticks/labels.
    ylim
        Limit of y-axis.
    drop_ci
        If True, does not plot CIs as error bars.
    ax
        Axis to plot graph on. If None, a new axis is provided.
    """

    df = df.sort_values("period").copy()

    x = df["period"].to_numpy()
    y = df["estimate"].to_numpy(dtype=float)
    upper = df["upper"].to_numpy()
    lower = df["lower"].to_numpy()

    # Error bars with reference period set to 0
    err_plus = np.where(np.isfinite(upper), upper - y, 0.0)
    err_minus = np.where(np.isfinite(lower), y - lower, 0.0)
    yerr = np.vstack([err_minus, err_plus])

    # ---- Create figure/axes

    if ax is None:
        if fig_kwargs is None:
            fig_kwargs = {}
        fig, ax = plt.subplots(**fig_kwargs)
    else:
        fig = ax.figure

    if plot_kwargs is None:
        plot_kwargs = {}

    defaults = {"linestyle": "-", "marker": "o", "color": "0", "label": None}

    for k, v in defaults.items():
        plot_kwargs.setdefault(k, v)

    _, caplines, barlines = ax.errorbar(
        x,
        y,
        yerr=yerr,
        ecolor="0.4",
        elinewidth=0.8,
        linewidth=1,
        markersize=5,
        capsize=4,
        **plot_kwargs,
    )

    if drop_ci:
        for obj in caplines + barlines:
            obj.remove()

    ax.axvline(time_of_event, color="red", linestyle="--", linewidth=1)
    ax.axhline(0, color=".6", linestyle=":", linewidth=1)

    ax.set_xticks(np.arange(x.min(), x.max() + 1, xjump))
    ax.tick_params(axis="y", rotation=90)

    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_ylim(ylim)

    ax.margins(x=0.02)

    ax.yaxis.set_major_formatter(
        FuncFormatter(
            lambda x, _: (
                "0"
                if abs(x) < 1e-12
                else f"{x:.2f}".replace("0.", ".").replace("-0.", "-.")
            )
        )
    )

    return fig, ax
