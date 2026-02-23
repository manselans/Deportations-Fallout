"""
Defines a function to estimate attrition hazards from a risk set.
"""

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, LinearRegression


def ipcw(
    df: pd.DataFrame,
    hazard_col: str,
    *,
    cat_features: list = None,
    num_features: list = None,
    model: str = "logit",
    model_kwargs: dict = None,
    clip: float = 1e-12
):
    """
    Estimate attrition / survival hazards via a
    user-specified hazard model.

    The function assumes that input data `df` already represents the "risk set",
    i.e. includes only observation of individuals at time t, who were observed in t-1. 

    Parameters
    ----------
    df :
        Input data. The risk-set.
    hazard_col :
        Column name of the binary hazard / attrition indicator.
        (1 = event, 0 = still at risk)
    cat_features :
        Categorical covariates to be one-hot encoded.
    num_features :
        Numerical covariates to be standardized.
    model : {"logit", "ols"}, default "logit"
        Type of estimator
        - "logit": LogisticRegression
        - "ols": LinearRegression
    model_kwargs :
        Keyword arguments passed directly to the estimator.
    clip :
        Lower and upper bound for clipping predicted hazards.

    Returns
    -------
    df_out : pandas.DataFrame
        Copy of `df`with an added column `predicted_hazard`.
    pipe :
        Fitted preprocessing + estimation pipeline.
    """

    df = df.copy()

    # Keep only features actually present in df
    if cat_features is not None:
        cat_features = [c for c in cat_features if c in df.columns]
    else:
        cat_features = []
    if num_features is not None:
        num_features = [c for c in num_features if c in df.columns]
    else:
        num_features = []

    if model_kwargs is None:
        model_kwargs = {}

    X_cols = cat_features + num_features
    if len(X_cols) == 0:
        raise ValueError("No valid covariates supplied.")

    # Pre-processing
    pre = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="error"), cat_features),
            ("num", StandardScaler(), num_features),
        ],
        remainder="drop",
    )

    # Estimator
    if model == "logit":
        estimator = LogisticRegression(**model_kwargs)
        use_proba = True
    elif model == "ols":
        estimator = LinearRegression(**model_kwargs)
        use_proba = False
    else:
        raise ValueError("model must be either 'logit' or 'ols'")

    pipe = Pipeline(steps=[("pre", pre), ("estimate", estimator)])

    X = df[X_cols]
    y = df[hazard_col]

    pipe.fit(X, y)

    # Predictions
    if use_proba:
        hhat = pipe.predict_proba(X)[:, 1]
    else:
        hhat = pipe.predict(X)

    df["predicted_hazards"] = np.clip(hhat, clip, 1 - clip)

    return df, pipe
