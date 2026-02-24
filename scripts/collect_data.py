"""Build analysis datasets used by the project."""

from depo_paper.config import PATHS
from depo_paper.build.population import run as population
from depo_paper.build.outcomes import run as outcomes
from depo_paper.build.panel import run as panel
from depo_paper.build.female_foreign import run as female_foreign


def main() -> None:
    paths = PATHS

    population()
    outcomes()
    panel()
    female_foreign()

    # Delete intermediate datasets; lines may be commented out 
    for file_name in ["bfl.parquet", "ilme.parquet", "kraf.parquet", "krsi.parquet", "krin.parquet"]:
        file_path = paths.temp / file_name
        if file_path.exists():
            file_path.unlink()


if __name__ == "__main__":
    main()
