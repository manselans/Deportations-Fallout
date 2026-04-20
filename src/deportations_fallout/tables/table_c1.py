"""
Create Table C1: Deportation Probabilities by Court.
"""

import warnings
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm

from deportations_fallout.utils.lookup import load_lookup
from deportations_fallout.config import PATHS

paths = PATHS


def table_c1():
    """
    Create Table C1: Deportation Probabilities by Court.
    """

    # ---- Judicial districts

    jd = load_lookup("judicial_districts.csv")[["ret", "komkode"]]
    jd = jd.rename(columns={"komkode": "kom"})

    # Combine Frederiksberg and København (due to overlaps)
    jd.loc[lambda d: d["ret"] == "Retten på Frederiksberg", "ret"] = "Københavns Byret"
    jd = jd.drop_duplicates()

    # ---- Set up data

    df = pd.read_parquet(paths.temp/"population.parquet")

    vl = ["pnr", "D", "year", "kom", "female", "age", "opr_land", "seniority"]
    df = df.dropna(subset=["kom", "residency"]).copy()

    df = df.assign(
        seniority=lambda d: np.floor(
            (d["conviction_date"] - d["residency"]).dt.days / 365
        ).clip(upper=15),
        age=lambda d: (d["year"] - d["foed_dag"].dt.year),
        female=lambda d: (d["koen"].astype("Int64") == 2),
    )

    df = df.merge(jd, on="kom", how="left", validate="m:1")
    df[vl] = df[vl].astype(float)

    # Remove smallest districts
    counts = df.groupby(["ret", "D"]).size().unstack(["D"], fill_value=0)
    mask = counts.index[counts.min(axis=1) >= 10]
    df = df[df["ret"].isin(mask)]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # To many parameters lead to warnings...

        risk1 = _adjusted_risk(df, group="ret", outcome="D", adjusters=None)[
            ["Estimate", "SE"]
        ]

        risk2 = _adjusted_risk(
            df, group="ret", outcome="D", adjusters=["age", "female"]
        )[["Estimate", "SE"]]

        risk3 = _adjusted_risk(
            df,
            group="ret",
            outcome="D",
            adjusters=["age", "female", "seniority", "year", "opr_land"],
        )[["Estimate", "SE"]]

    risks = pd.concat(
        [risk1, risk2, risk3],
        axis=1,
        keys=["No controls", "Age and sex", "+ Seniority, origin and year"],
    ).round(2)

    risks.index.name = "Court"

    return risks


def _adjusted_risk(
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
                "Estimate": pred["mean"].mean(),
                "Lower": pred["mean_ci_lower"].mean(),
                "Upper": pred["mean_ci_lower"].mean(),
                "SE": pred["mean_se"].mean(),
                "nobs": len(dtmp),
            }
        )

    return pd.DataFrame(out).set_index(group)
