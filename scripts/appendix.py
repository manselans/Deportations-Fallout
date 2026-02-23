"""Run appendix analyses."""

from depo_paper.analysis.appendix_a import run as appendix_a
from depo_paper.analysis.appendix_b import run as appendix_b
from depo_paper.analysis.appendix_c import run as appendix_c


def main() -> None:
    appendix_a()
    appendix_b()
    appendix_c()


if __name__ == "__main__":
    main()
