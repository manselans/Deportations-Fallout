"""Example local configuration.

Copy this file to local_config.py and edit DATA_PATHS.
"""

from pathlib import Path

DATA_PATHS = {
    "dst_raw": Path("/path/to/dst_raw"),
    "crime": Path("/path/to/crime"),
    "formats": Path("/path/to/formats"),
    "disced": Path("/path/to/disced"),
}
