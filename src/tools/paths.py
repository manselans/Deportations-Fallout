"""
Project setup: Creates project folders and expose their paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    """
    Paths used in project. Exposed in init().
    """
    root: Path
    data: Path
    output: Path
    figures: Path
    tables: Path
    dst: Path
    crime: Path
    formats: Path
    disced: Path


def init(root: Path) -> Paths:
    """
    Infer project root from the notebook location,
    ensure standard folders exist, and return their paths.

    Parameters
    ----------
    root
        Project root path (where folders data and output will be placed).
    Returns
    -------
    Paths
        Container with Path objects for standard project folders.
    """

    data = root / "data"
    output = root / "output"
    figures = output / "figures"
    tables = output / "tables"

    for p in (data, output, figures, tables):
        p.mkdir(parents=True, exist_ok=True)

    # Data and formats from DST (change appropriately)
    dst = Path(r"E:\Data\rawdata\703566")
    crime = Path(r"E:\Data\workdata\703566\crimedata")
    formats = Path(r"\\srvfsenas1\data\Formater\SAS formater i Danmarks Statistik")
    disced = formats / Path(r"STATA_datasaet\Disced")

    return Paths(
        root=root,
        data=data,
        output=output,
        figures=figures,
        tables=tables,
        dst=dst,
        crime=crime,
        formats=formats,
        disced=disced,
    )
