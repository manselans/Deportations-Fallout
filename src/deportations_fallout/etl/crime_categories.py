"""
Functions sorting types of crime by Statistic Denmark's 
7-digit cirminal offense code. 
"""

import pandas as pd
import numpy as np


def rd_exception(s: pd.Series) -> pd.Series:
    """
    Function to determine if a crime is exempt from regression discontinuity.

    Parameters
    ----------
    s
        pandas Series containing 7-digit criminal codes (DST, AFGER7).

    Returns
    -------
    out
        pandas Series of bools. True implies that the crime is exempt from the disconinuity.
    """

    s = s.copy()

    # Demarcate criminal offenses exempt from seniority rules
    out = [
        s.between(3200000, 3299999) | (s == 1435707),  # Drugs
        s.between(1100000, 1199999),  # Sex crimes
        s.between(1200000, 1299999),  # Violent crimes + illegal coercion
        s.isin([1312505, 1312510]),  # Arson
        s.between(1354505, 1372505),  # Fraud
        s.between(1376506, 1376509),  # Fencing of stolen goods
        s.between(1380305, 1380911),  # Robbery
        s.isin([1384505, 1384510, 1384515, 1384705]),  # Tax fraud
        s.isin(
            [
                1390505,
                1390506,
                1390510,
                1390511,
                1390520,
                1390521,
                1390525,
                1390526,
                1390705,
            ]
        ),  # Vandalism
        s.between(1410303, 1410754),  # Crimes against the state
        s.isin(
            [1445705, 1445710, 1445720, 1445725, 1445735, 1445740, 1445765]
        ),  #  Crimes against public health
        s.isin(
            [1445777, 1445778, 1445779, 1445780, 1445781, 1445782]
        ),  # Weapons and explosives
        (s == 1455505),  # Bigamy
        (s == 1455530),  # Evasion of parenthood
        (s == 3810207),  # Alien's Act (false information to obtain residence permit)
    ]

    out = np.logical_or.reduce(out)

    return out


def crime_type(x: pd.Series) -> pd.Series:
    """
    Function to determine the type of crime commited based on 7-digit criminal codes.

    Parameters
    ----------
    s
        pandas Series containing 7-digit criminal codes (DST, AFGER7).

    Returns
    -------
    out
        pandas Series of strings.
    """

    if not pd.api.types.is_numeric_dtype(x):
        raise ValueError("Input must be a numeric pandas series.")

    t = ["sex", "violence", "property", "narcotics", "unknown"]
    c = [
        x.between(1100000, 1199999),
        x.between(1200000, 1299999),
        x.between(1300000, 1399999),
        x.between(1435000, 1440999),
        x.eq(9999999),
    ]

    return np.select(c, t, default="other")
