"""
Project setup: 
- Sets default settings for plots in matplotlib.pyplot. 
"""

from __future__ import annotations
import matplotlib.pyplot as plt


def setup_pyplot() -> None:
    """Set plotting defaults for matplotlib.pyplot"""

    plt.rcParams.update(
        {
            "figure.dpi": 80,
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 12,
            "legend.frameon": False,
            "legend.loc": "upper center",
            "savefig.bbox": "tight",
        }
    )

