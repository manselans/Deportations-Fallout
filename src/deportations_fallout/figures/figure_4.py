"""
Create Figure 4: Migrations Out of Denmark.
"""

import pandas as pd

from deportations_fallout.results.dynamic_did import raw_rates

# Configuration
from deportations_fallout.config import PATHS, setup_matplotlib

paths = PATHS
setup_matplotlib()


def figure_4():
    """
    Create Figure 4: Migrations Out of Denmark.
    """

    df = pd.read_parquet(paths.temp/"panel.parquet")

    df = df[["in_country", "event_time", "D"]].copy()

    df["out_of_country"] = 1 - df["in_country"].astype(int)

    fig, ax = raw_rates(
        df, var="out_of_country", ylabel="Share out of country", ylim=(0, 0.13)
    )

    ax.axvline(-7, color="0.4", linestyle="--")
    ax.legend(bbox_to_anchor=(0.5, -0.12), ncol=2)

    return fig, ax
