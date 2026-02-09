"""
Builds the analysis datasets used in the project.
"""

from pathlib import Path
from tools.paths import init

from build.build01_population import run as pop
from build.build02_outcomes import run as cov
from build.build03_panel import run as panel
from build.build04_female_foreign import run as ffb


def main():
    """
    Runs the four modules responsible for building the data sets used in the analysis:
    - Gathers population (treatment + control) and finds their spouses
    - Finds outcomes and covariates on spouses
    - Builds panel
    - Gathers information on the female foreign-born population in 2010
    Saves the above data to /data
    """
    paths = init(Path.cwd())

    pop()
    cov()
    panel()
    ffb()

    # remove files not used in analysis
    (paths.data / "bfl.parquet").unlink()
    (paths.data / "ilme.parquet").unlink()
    (paths.data / "kraf.parquet").unlink()
    (paths.data / "krsi.parquet").unlink()
    (paths.data / "krin.parquet").unlink()


if __name__ == "__main__":
    main()
