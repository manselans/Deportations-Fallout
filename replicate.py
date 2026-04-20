"""
Replicate paper.
Prepare all data sets used in the analysis and generate 
all tables and figures included in paper.
Save data to ``temp/``. 
Save figures and tables to ``output/``.
"""

from run.collect_data import collect_data
from run.run_analysis import run_analysis

def replicate():
    """
    Replicate paper.
    Prepare all data sets used in the analysis and generate 
    all tables and figures included in paper.
    Save data to ``temp/``. 
    Save figures and tables to ``output/``.
    """

    print("Collecting data...")
    collect_data()

    print("Generating figures and tables...")
    run_analysis()

    print("Replication complete.")


if __name__ == "__main__":
    replicate()
