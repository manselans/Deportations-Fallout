"""
Creates figures and tables for Appendix C.
Content: Figures C1 and C2 and Table C1.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches

import statsmodels.formula.api as smf
import statsmodels.api as sm

from depo_paper.config import PATHS
from depo_paper.tools.formatting import setup_pyplot
from depo_paper.tools.io import load_csv, gather
from depo_paper.tools.rules import crime_type


def run():
    """
    Creates:
    - Table C1: Deportation Probabilities by Court
    - Figure C1: Residency Seniority by Deportation Order
    - Figure C2: Types of Crimes by Deportation Order
    """

    # expose paths (and create output folders if they do not exist)
    paths = PATHS

    # setup plotting defaults
    setup_pyplot()

    # load data
    panel = pd.read_parquet(paths.temp / "panel.parquet")
    population = pd.read_parquet(paths.temp / "population.parquet")

    jd = load_csv("judicial_districts.csv").rename(columns={"komkode": "municipality"})

    # Combine Frederiksberg and København (due to overlaps)
    mask = jd["kommune"].str.contains("Frederiksberg") | jd["kommune"].str.contains(
        "København"
    )

    jd.loc[mask, "municipality"] = 101
    jd = jd.groupby("municipality", as_index=False).agg({"ret": "last"})

    # ---- Table C1: Deportation Probabilities by Court

    df = (
        population.loc[population["kom"].notna() & population["residency"].notna()]
        .assign(
            seniority=lambda d: (
                np.floor((d["conviction_date"] - d["residency"]).dt.days / 365).clip(
                    upper=15
                )
            ),
            age=lambda d: (d["year"] - d["foed_dag"].dt.year),
            female=lambda d: (d["koen"].astype("Int64") == 2),
            municipality=lambda d: d["kom"].mask(
                d["kom"].eq(147), 101
            ),  # combine Frederiksberg and København
        )
        .rename(columns={"opr_land": "origin"})[
            ["pnr", "D", "year", "municipality", "female", "age", "origin", "seniority"]
        ]
        .merge(jd, on="municipality", how="left")
    )

    df = df.astype({**{c: "float64" for c in df.select_dtypes(exclude="object")}})

    # Remove smallest districts
    counts = df.groupby(["ret", "D"]).size().unstack(["D"], fill_value=0)
    mask = counts.index[counts.min(axis=1) >= 10]
    df = df[df["ret"].isin(mask)]

    def adjusted_risk(
        df: pd.DataFrame,
        group,
        outcome,
        adjusters,
        alpha=0.1,
        glm=False,
        cov_type="HC1",
    ):
        """
        Function do derive adjusted probabilities.
        """

        if adjusters:
            rhs = " + ".join([f"C({group})"] + [f"C({a})" for a in adjusters])
        else:
            rhs = f"C({group})"

        if glm:
            model = smf.glm(
                f"{outcome} ~ {rhs}", data=df, family=sm.families.Binomial()
            ).fit(cov_type=cov_type)
        else:
            model = smf.ols(f"{outcome} ~ {rhs}", data=df).fit(cov_type=cov_type)

        out = []
        for g in df[group].unique():
            dtmp = df.copy()
            dtmp[group] = g
            pred = model.get_prediction(dtmp).summary_frame(alpha=alpha)
            out.append(
                {
                    group: g,
                    "Risk": pred["mean"].mean(),
                    "Lower": pred["mean_ci_lower"].mean(),
                    "Upper": pred["mean_ci_lower"].mean(),
                    "Std. error": pred["mean_se"].mean(),
                    "nobs": len(dtmp),
                }
            )

        return pd.DataFrame(out).set_index(group)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # To many parameters lead to warnings...

        risk1 = adjusted_risk(df, group="ret", outcome="D", adjusters=None)[
            ["Risk", "Std. error"]
        ]

        risk2 = adjusted_risk(
            df, group="ret", outcome="D", adjusters=["age", "female"]
        )[["Risk", "Std. error"]]

        risk3 = adjusted_risk(
            df,
            group="ret",
            outcome="D",
            adjusters=["age", "female", "seniority", "year", "origin"],
        )[["Risk", "Std. error"]]

    risks = pd.concat(
        [risk1, risk2, risk3],
        axis=1,
        keys=["No controls", "Age and sex", "+ Seniority, origin and year"],
    ).round(2)

    risks.index.name = "Judicial district"
    risks.to_html(paths.tables / "Table C1.html", escape=False)


    # ---- Figure C1: Residency Seniority by Deportation Order

    df = (
        panel.dropna(subset=["seniority"])
        .assign(seniority=lambda d: np.floor((d["seniority"] / 365)))[
            ["pnr", "D", "seniority"]
        ]
        .drop_duplicates()
    )

    # Discretion
    df_c = df.loc[df["D"] == 0].copy()
    df_t = df.loc[df["D"] == 1].copy()

    counts = df_c.groupby(["seniority"]).size()
    df_c = df_c[df_c["seniority"].isin(counts.index[counts >= 5])]

    counts = df_t.groupby(["seniority"]).size()
    df_t = df_t[df_t["seniority"].isin(counts.index[counts >= 5])]

    # Plot as histogram
    fig, ax = plt.subplots(figsize=(8, 6))

    plt.hist(  # control group
        df_c["seniority"],
        bins=25,
        density=True,
        alpha=0.25,
        color="0",
        label="Conviction only",
    )

    plt.hist(  # treatment group
        df_t["seniority"],
        bins=25,
        density=True,
        alpha=0.25,
        color="red",
        label="Conviction and deportation order",
    )

    plt.xlabel("Years of seniority")
    plt.ylabel("Density")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)

    plt.yticks(rotation=90)
    plt.gca().yaxis.set_major_formatter(
        FuncFormatter(
            lambda x, _: (
                "0"
                if abs(x) < 1e-12
                else f"{x:.2f}".replace("0.", ".").replace("-0.", "-.")
            )
        )
    )

    fig.savefig(paths.figures / "Figure C1A.png")

    # As QQ-plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.qqplot_2samples(df_c["seniority"], df_t["seniority"], line="45", ax=ax)

    ax.set_ylabel("Quantiles of seniority, Conviction only")
    ax.set_xlabel("Quantiles of seniority, Conviction and deportation order")

    ax.lines[0].set_markerfacecolor("black")
    ax.lines[0].set_markeredgecolor("black")
    ax.lines[1].set_zorder(1)

    fig.savefig(paths.figures / "Figure C1B.png", dpi=500, bbox_inches="tight")

    # ---- Figure C2: Types of Crimes by Deportation Order

    ids = set(population.pnr)

    # gather conviction records
    kraf = gather(
        paths.dst,
        names=range(1980, 2022),
        file_pattern="kraf{name}.dta",
        columns=["pnr", "afg_afgoedto", "afg_ger7"],
        filters={"pnr": ids},
        concatenate=True,
    )

    # determine most severe crime of the day
    crime_rank = {
        "unknown": 0,
        "other": 1,
        "property": 2,
        "narcotics": 3,
        "violence": 4,
        "sex": 5,
    }

    kraf = (
        kraf.rename(columns={"afg_afgoedto": "conviction_date"})
        .assign(
            crime_rank=lambda d: pd.Series(crime_type(d.afg_ger7), index=d.index).map(
                crime_rank
            )
        )
        .sort_values(["pnr", "conviction_date", "crime_rank"])
        .drop_duplicates(subset=["pnr", "conviction_date"], keep="last")
        .drop(columns=["crime_rank"])
    )

    # add most severe crime codes from day of conviction
    df = population.merge(
        kraf, on=["pnr", "conviction_date"], how="left", validate="1:1"
    )
    df["ctype"] = crime_type(df.afg_ger7)

    crime_names = {
        "other" : "Other",
        "property" : "Property",
        "narcotics" : "Narcotics",
        "violence" : "Violence",
        "sex" : "Sexual"
    }

    # crime type shares
    df["ctype"] = df["ctype"].map(crime_names)
    shares = df.groupby("D")["ctype"].value_counts(normalize=True).unstack(fill_value=0)
    counts = df.groupby("D")["ctype"].value_counts().unstack(fill_value=0)

    # discretion
    shares = shares.where(counts >= 3, 0)

    # plot
    colors = [".8", ".5", ".2", "1", "1"]
    ecolors = [".8", ".5", ".2", "0.2", ".2"]
    hatches = [None, None, None, "/", "."]

    ax = shares.plot(
        kind="barh", stacked=True, figsize=(8, 6), color="white", edgecolor="black"
    )

    for i, bari in enumerate(ax.patches):
        col = i // len(shares)
        bari.set_facecolor(colors[col])
        bari.set_edgecolor(ecolors[col])
        bari.set_hatch(hatches[col])

    ax.set_xlabel("Share")
    ax.set_ylabel("")
    ax.set_yticks(
        [0, 1], labels=["Conviction only", "Conviction and deportation order"]
    )
    ax.tick_params(axis="y", rotation=45)

    ax.set_xlim((0, 1))

    handles = [
        mpatches.Patch(facecolor=cl, edgecolor=ec, hatch=h, label=colname)
        for cl, ec, h, colname in zip(colors, ecolors, hatches, shares.columns)
    ]
    ax.legend(handles=handles, bbox_to_anchor=(0.5, -0.12), ncol=3)

    ax.figure.savefig(paths.figures / "Figure C2.png")


if __name__ == "__main__":
    run()
