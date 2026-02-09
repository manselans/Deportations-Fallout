from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class KNNConfig:

    # core columns
    id_col: str = "pnr"
    t_col: str = "event_time"
    y_col: str = "partic"

    # stratification + year of event with back-off
    strata_fixed: List[str] = None
    year_event_col: str = "year"

    # distance features
    distance_numeric: List[str] = None
    distance_categorical: List[str] = None

    # KNN and back-off controls
    k: int = 3
    max_backoff_years: int = 2
    stochastic: bool = True
    random_state: Optional[int] = 1


def _standardize_numeric(donors_num: pd.DataFrame, target_num: pd.Series):
    """Standardize numeric features. Fill NaNs with donor medians"""

    if donors_num.shape[1] == 0:
        # no numeric features
        return donors_num.to_numpy(), target_num.to_numpy()

    donors_num = donors_num.copy()
    target_num = target_num.copy()

    # fill NaNs
    donors_num = donors_num.apply(lambda s: s.fillna(s.median()))
    target_num = target_num.fillna(donors_num.median())

    # standardize
    means = donors_num.mean()
    stds = donors_num.std(ddof=0).replace(0, 1.0)

    donors_std = (donors_num - means) / stds
    target_std = (target_num - means) / stds

    return donors_std.to_numpy(), target_std.to_numpy()


def _one_hot_align(donors_cat: pd.DataFrame, target_cat: pd.Series):
    """One-hot encode categoricals using donors' categories, align target accordingly"""
    if donors_cat.shape[1] == 0:
        # no categorical features
        return np.empty((len(donors_cat), 0)), np.empty((0,), dtype=float)

    donors_ohe = pd.get_dummies(donors_cat.astype("object"), dummy_na=True)
    donor_cols = donors_ohe.columns

    target_ohe = pd.get_dummies(
        pd.DataFrame([target_cat.astype("object")]), dummy_na=True
    )
    target_ohe = target_ohe.reindex(columns=donor_cols, fill_value=0)

    return donors_ohe.to_numpy(), target_ohe.to_numpy().ravel()


def _distance_matrix(donors_num, target_num, donors_cat, target_cat):
    """Concatenate numeric+categorical features and return Euclidian distances (donor-by-1)"""
    # concatenate
    if donors_num.size == 0:
        Xd_num, xt_num = np.empty((len(donors_cat), 0)), np.empty((0,))
    else:
        Xd_num, xt_num = donors_num, target_num

    if donors_cat.size == 0:
        Xd_cat, xt_cat = np.empty((len(donors_num), 0)), np.empty((0,))
    else:
        Xd_cat, xt_cat = donors_cat, target_cat

    Xd = np.concatenate([Xd_num, Xd_cat], axis=1)
    xt = np.concatenate([xt_num, xt_cat], axis=0).reshape(1, -1)

    # Euclidean dinstances
    diffs = Xd - xt
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    return dists


def knn_hotdeck_impute(df: pd.DataFrame, cfg: KNNConfig) -> pd.DataFrame:
    """
    KNN hot-deck imputation for missing binary y at (i, t), using donors at t restricted by a stratification.
    Distance within donors is computed using selected features and calendar time.

    Returns a copy of df with:
        - y_imputed: outcome after imputation (still NaN if failed)
        - p_hat: donor mean used for imputation (NaN if failed)
        - donor_count: number of donors used (k OR 0 if failed)
        - backoff_used: the time radius actually used (0..max)
    """
    df = df.copy()

    # defaults
    if cfg.strata_fixed is None:
        cfg.strata_fixed = ["D", "female", "ie_type", "has_children"]
    if cfg.distance_numeric is None:
        cfg.distance_numeric = ["age", "pre_move_y", "seniority"]
    if cfg.distance_categorical is None:
        cfg.distance_categorical = []

    # required columns check
    needed = set(
        [cfg.id_col, cfg.t_col, cfg.y_col, cfg.year_event_col] + cfg.strata_fixed
    )
    missing_cols = [c for c in needed if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # output columns
    df["y_imputed"] = df[cfg.y_col]
    df["p_hat"] = np.nan
    df["donor_count"] = 0
    df["backoff_used"] = np.nan

    rng = np.random.default_rng(cfg.random_state) if cfg.stochastic else None

    # indeces of missing y
    miss_idx = df.index[df[cfg.y_col].isna()].tolist()

    for idx in miss_idx:
        row = df.loc[idx]
        t_target = row[cfg.t_col]
        year0 = row[cfg.year_event_col]

        # fixed strata mask
        mask_fixed = pd.Series(True, index=df.index)
        for col in cfg.strata_fixed:
            mask_fixed &= df[col] == row[col]

        # same event time t and observed donors only
        mask_t = df[cfg.t_col] == t_target
        mask_obs = df[cfg.y_col].notna()

        donors_found = False
        # back-off mechanism: extend time window as needed (until max reached)
        for radius in range(0, cfg.max_backoff_years + 1):
            mask_year = (df[cfg.year_event_col] - year0).abs() <= radius
            donors_mask = mask_fixed & mask_t & mask_year & mask_obs

            if donors_mask.sum() < cfg.k:
                continue

            donors = df.loc[donors_mask].copy()
            n_donors = len(donors)

            # if exactly k donors, take all (no distances needed)
            if n_donors == cfg.k:
                donor_indices = donors.index.to_numpy()
                p_hat = float(df.loc[donor_indices, cfg.y_col].astype(float).mean())
                if cfg.stochastic:
                    draw = rng.binomial(1, p_hat)
                    df.at[idx, "y_imputed"] = float(draw)
                else:
                    df.at[idx, "y_imputed"] = float(p_hat)
                df.at[idx, "p_hat"] = p_hat
                df.at[idx, "donor_count"] = int(cfg.k)
                df.at[idx, "backoff_used"] = radius
                donors_found = True
                break

            # if more than k donors, compute distances (if features are available)
            num_cols = (
                [c for c in cfg.distance_numeric if c in df.columns]
                if cfg.distance_numeric
                else []
            )
            cat_cols = (
                [c for c in cfg.distance_categorical if c in df.columns]
                if cfg.distance_categorical
                else []
            )
            have_distance_features = (len(num_cols) + len(cat_cols)) > 0

            if not have_distance_features:
                # edge case: no features -> select k donors from strata at random
                donor_indices = rng.choice(
                    donors.index.to_numpy(), size=cfg.k, replace=False
                )
                p_hat = float(df.loc[donor_indices, cfg.y_col].astype(float).mean())
            else:
                # build numeric features
                donors_num = (
                    donors[num_cols] if num_cols else pd.DataFrame(index=donors.index)
                )
                target_num = row[num_cols] if num_cols else pd.Series(dtype=float)

                # add year of event to prioritise temporal nearness if multiple years are included
                donors_num = donors_num.assign(
                    time=donors[cfg.year_event_col].astype(float)
                )
                target_num = target_num.reindex(donors_num.columns)
                target_num["time"] = float(year0)

                donors_num_std, target_num_std = _standardize_numeric(
                    donors_num, target_num
                )

                # build categorical features
                donors_cat = (
                    donors[cat_cols] if cat_cols else pd.DataFrame(index=donors.index)
                )
                target_cat = row[cat_cols] if cat_cols else pd.Series(dtype=object)

                donors_cat_ohe, target_cat_ohe = _one_hot_align(
                    donors_cat,
                    (
                        target_cat
                        if isinstance(target_cat, pd.Series)
                        else pd.Series(target_cat)
                    ),
                )

                dists = _distance_matrix(
                    donors_num_std, target_num_std, donors_cat_ohe, target_cat_ohe
                )

                nn_idx = np.argpartition(dists, cfg.k - 1)[: cfg.k]
                donor_indices = donors.index[nn_idx]
                p_hat = float(df.loc[donor_indices, cfg.y_col].astype(float).mean())

                if cfg.stochastic:
                    df.at[idx, "y_imputed"] = float(rng.binomial(1, p_hat))
                else:
                    df.at[idx, "y_imputed"] = float(p_hat)

                df.at[idx, "p_hat"] = p_hat
                df.at[idx, "donor_count"] = int(cfg.k)
                df.at[idx, "backoff_used"] = radius
                donors_found = True
                break

            if not donors_found:
                df.at[idx, "donor_count"] = int(0)

    return df
