"""
Collect data for analysis.
"""

from deportations_fallout.etl.define_population import run as define_population
from deportations_fallout.etl.find_spouses import run as find_spouses
from deportations_fallout.etl.spouse_outcomes import run as spouse_outcomes
from deportations_fallout.etl.grounds_of_residency import run as grounds_of_residency
from deportations_fallout.etl.construct_panel import run as construct_panel
from deportations_fallout.etl.female_foreignborn import run as female_foreignborn


def collect_data():
    """Collect data for analysis."""

    print("Defining population...")
    define_population()

    print("Finding spouses...")
    find_spouses()

    print("Gather spouses' outcomes...")
    spouse_outcomes()

    print("Find grounds of residency...")
    grounds_of_residency()

    print("Construct panel and identify migration patterns...")
    construct_panel()

    print("Gather data on female foreign-borns...")
    female_foreignborn()

    print("Data collection complete.")


if __name__ == "__main__":
    collect_data()
