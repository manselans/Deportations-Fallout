#!/usr/bin/env python3
"""Master replication entry point for depo_paper."""

from depo_paper.config import PATHS, setup_matplotlib
from scripts.collect_data import main as collect_data
from scripts.run_analysis import main as run_analysis
from scripts.appendix import main as appendix


def main() -> None:
    PATHS.ensure_dirs()
    setup_matplotlib()

    print("Running data collection...")
    collect_data()

    print("Running main analysis...")
    run_analysis()

    print("Running appendix analyses...")
    appendix()

    print("Replication complete!")


if __name__ == "__main__":
    main()
