"""
Example local configuration for project.

Copy this file to local_config.py
and edit the values below to match your local environment.
"""

from pathlib import Path

# ---------------------------------------------------------------------
# Required: Paths to the data on your server
# ---------------------------------------------------------------------

DATA_PATHS = {
    "dst_raw": Path(r"paths/to/raw/data"),
    "crime": Path(r"path/to/crime/data"),
}
