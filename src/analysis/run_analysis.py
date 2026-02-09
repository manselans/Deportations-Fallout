"""
Creates the tables and figures shown in the paper.
"""

from analysis.analyse01_descriptives import run as desc
from analysis.analyse02_main_results import run as mres
from analysis.analyse03_migrations import run as migr
from analysis.analyse04_grounds_of_residency import run as gres
from analysis.appendix_a import run as ap_a
from analysis.appendix_b import run as ap_b
from analysis.appendix_c import run as ap_c


def main():
    """
    Runs the six modules responsible for generating figrues and tables:
    - Descriptives
    - Main results
    - Forced migration/attrition
    - Results by grounds of residency
    - Appendices A, B and C
    Saves the results to /output; figures to /output/figures and tables to /output/tables
    """

    desc()
    mres()
    migr()
    gres()
    ap_a()
    ap_b()
    ap_c()

if __name__ == "__main__":
    main()
