"""
Module of functions to help create tables, hereunder:
- Descriptive tables by group
- Tables of results
- Significance stars
"""

import pandas as pd
import numpy as np
import scipy.stats


def table_by_two(
    df: pd.DataFrame,
    group: str,
    *,
    cols: list = None,
    stats: tuple = ("mean", "std", "count"),
    add_p: bool = True,
    star_cuts: tuple = (0.1, 0.05, 0.01),
    stat_decimals: int = 2,
    p_decimals: int = 2,
    group_names: dict = None,
    stat_names: dict = None,
    row_names: dict = None,
) -> pd.DataFrame:
    """
    Creates a table of discriptive statistics across 2 groups.

    Parameters
    ----------
    df
        Input data
    group
        Name of column containing grouping variable. Must have exactly 2 distinct values.
    cols
        List of column names of columns to be included in the table.
        If None, includes all numeric columns of `df`.
    stats
        Tuple of statistics to be included. Naming as in pandas.agg().
    add_p
        If True, adds column of p-values for mean-differences across groups.
    star_cuts
        Significance levels for stars to be added based on p-values for mean-differences.
        If None, no significance stars are added.
    stat_decimals
        Number of decimals to be displayed for statistics.
    p_decimals
        Number of decimals to be displayed for p-values.
    group_names
        Dictionary of names for the two groups (top-column names).
    stat_names
        Dictionary of names for the statistics (lower-column names).
    row_names
        Dictionary of names for the variables (row names).
    """
    # Determine columns
    if cols is None:
        cols = [c for c in df.select_dtypes(include="number").columns if c != group]
    if not cols:
        raise ValueError("No numeric columns to summarize")

    # Two groups
    levels = [g for g in pd.unique(df[group]) if pd.notna(g)]
    if len(levels) != 2:
        raise ValueError(
            f"`{group}` must have exactly two non-missing levels; got {len(levels)}"
        )
    g0, g1 = levels

    # Coerce chosen cols to numeric
    x0 = df.loc[df[group] == g0, cols].apply(pd.to_numeric, errors="coerce")
    x1 = df.loc[df[group] == g1, cols].apply(pd.to_numeric, errors="coerce")

    # Group summaries
    tab0 = x0.agg(stats).T
    tab1 = x1.agg(stats).T
    tbl = pd.concat({str(g0): tab0, str(g1): tab1}, axis=1)

    # Optional p-values (Welch t-test)
    if add_p:

        pvals = {}
        for c in cols:
            a = x0[c].to_numpy()
            b = x1[c].to_numpy()
            if np.isfinite(a).any() and np.isfinite(b).any():
                p = scipy.stats.ttest_ind(
                    a, b, equal_var=False, nan_policy="omit"
                ).pvalue
            else:
                p = np.nan
            pvals[c] = p

        # Round
        p = pd.Series(pvals, dtype="float")
        p_fmt = p.apply(
            lambda v: (
                "" if pd.isna(v) else f"{round(float(v), p_decimals):.{p_decimals}f}"
            )
        )

        if star_cuts is not None:
            a1, a2, a3 = star_cuts

            def _star(p):
                if pd.isna(p):
                    return ""
                return "***" if p < a3 else "**" if p < a2 else "*" if p < a1 else ""

            p_fmt = p_fmt + p.map(_star).fillna("")

        tbl[("p", "")] = p_fmt

    # Formatting
    tbl = tbl.round(stat_decimals)

    if group_names is not None:
        tbl = tbl.rename(columns=group_names, level=0)

    if stat_names is not None:
        tbl = tbl.rename(columns=stat_names, level=1)

    if row_names is not None:
        tbl = tbl.rename(index=row_names)

    return tbl


def results_table(
    res_dict: dict,
    rows: list,
    *,
    model_names: list = None,
    std_errors: bool = True,
    alpha: float = 0.10,
    floatfmt="{:.3f}",
    brackets: tuple = ("(", ")"),
    add_stats: tuple = None,
    var_names: dict = None,
):
    """
    Posts regression results as a table in pandas.DataFrame and HTML.

    Parameters
    ----------
    res_dict
        Dictionary of regression results to be included.
        Each value will be a column. If `model_names` are None,
        keys will be used as corresponding column names.
    rows
        List of coefficient/variable names form regression results to be included.
    std_errors
        If True, include standard errors. If False, confidence intervals are added instead.
    alpha
        Significance level for confidence intervals.
    floatfmt
        Format of floats in table.
    brackets
        Type of brackets around standard error/confidence intervals.
    add_stats
        Add stats from regression results. E.g., number of obs or R-squared.
    var_names
        Dictionary for renaming rows/variables.

    Returns
    -------
    raw
        pandas.DataFrame of regression results.
    display
        HTML of regression results formatted for display.
    """

    keys = list(sorted(res_dict.keys()))
    if model_names is None:
        model_names = [f"Model {k}" for k in keys]

    # Build small data frame per model
    blocks = []
    for k in keys:

        r = res_dict[k]

        # Extract and align
        params = r.params.reindex(rows)
        ci = r.conf_int(alpha=alpha).reindex(rows)
        pvals = r.pvalues.reindex(rows)
        bse = r.bse.reindex(rows)

        # Robust handling of missing rows
        if ci is not None and hasattr(ci, "shape") and ci.shape[1] == 2:
            lb, ub = ci.iloc[:, 0], ci.iloc[:, 1]
        else:
            lb = ub = pd.Series(np.nan, index=rows)

        if std_errors:
            block = pd.DataFrame(
                {"coef": params, "bse": bse, "pval": pvals}, index=rows
            )
            blocks.append(block)
        else:
            block = pd.DataFrame(
                {
                    "coef": params,
                    "lb": lb,
                    "ub": ub,
                    "pval": pvals,
                },
                index=rows,
            )
            blocks.append(block)

    # Numeric table with MultiIndex columns: (Model, metric)
    raw = pd.concat(blocks, axis=1, keys=model_names)

    # ---- Basic table setup ----

    # Formatted table
    lbr, rbr = brackets
    disp_cols = []
    for name in model_names:
        b = raw[name]

        if std_errors:
            disp = b.apply(
                lambda row: (
                    f'{_fmt(row["coef"], floatfmt)}{stars_from_p(row["pval"])}'
                    + _breaker(row["coef"])
                    + (
                        ""
                        if pd.isna(row["coef"])
                        else f'{lbr}{_fmt(row["bse"], floatfmt)}{rbr}'
                    )
                ),
                axis=1,
            )
        else:
            disp = b.apply(
                lambda row: (
                    f'{_fmt(row["coef"], floatfmt)}{stars_from_p(row["pval"])}'
                    + _breaker(row["coef"])
                    + (
                        ""
                        if pd.isna(row["coef"])
                        else f'{lbr}{_fmt(row["lb"], floatfmt)}, {_fmt(row["ub"], floatfmt)}{rbr}'
                    )
                ),
                axis=1,
            )

        disp_cols.append(disp.to_frame(name))
    display = pd.concat(disp_cols, axis=1)

    # ---- Appending stats ----

    if add_stats is not None:
        if isinstance(add_stats, str):
            add_stats = (add_stats,)
        else:
            add_stats = tuple(add_stats)

        stats_num = pd.DataFrame(
            {
                name: [_get_stat(res_dict[k], s) for s in add_stats]
                for k, name in zip(keys, model_names)
            },
            index=add_stats,
        )

        stats_str = stats_num.copy().astype(object)
        for s in add_stats:
            for name in model_names:
                v = stats_num.loc[s, name]
                if s == "nobs" and pd.notna(v):
                    stats_str.loc[s, name] = f"{int(v):,}"
                else:
                    stats_str.loc[s, name] = _fmt_num(v, floatfmt)

        display = pd.concat([display, stats_str], axis=0)

    # ---- Renaming variables ----

    if var_names is not None:
        display.rename(index=var_names, inplace=True)
        raw.rename(index=var_names, inplace=True)

    return raw, display


def _fmt_num(x, floatfmt):
    if x is None:
        return ""
    try:
        if isinstance(x, (int, np.integer)):
            return f"{x:,}"
        x = float(x)
        if np.isnan(x) or np.isinf(x):
            return ""
        return floatfmt.format(x)
    except (TypeError, ValueError):
        return ""


def _get_stat(res, name):
    return getattr(res, name, np.nan)


def _fmt(x, floatfmt):
    return "" if pd.isna(x) else floatfmt.format(x)


def _breaker(x):
    return "" if pd.isna(x) else "<br>"


def stars_from_p(p: float):
    """
    Returns a string of significance stars based on a p-value `p`.
    """
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "†"
    return ""
