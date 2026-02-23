"""Run all main analysis modules and generate tables/figures."""

from depo_paper.analysis.descriptives import run as descriptives
from depo_paper.analysis.main_results import run as main_results
from depo_paper.analysis.migrations import run as migrations
from depo_paper.analysis.grounds_of_residency import run as grounds_of_residency


def main() -> None:
    descriptives()
    main_results()
    migrations()
    grounds_of_residency()


if __name__ == "__main__":
    main()
