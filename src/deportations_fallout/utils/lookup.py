"""
Helper to load lookup CSV files from `src/depo_children/lookups`.
"""

from pathlib import Path
from typing import Optional
import pandas as pd

# Resolve once at import time
_LOOKUP_DIR = Path(__file__).resolve().parent.parent / "lookups"


def load_lookup(filename: str, dtype: Optional[dict] = None) -> pd.DataFrame:
    """
    Load a lookup CSV from `src/depo_children/lookups`.
    """
    path = _LOOKUP_DIR / filename

    if not path.exists():
        raise FileNotFoundError(f"Lookup file not found: {path}")

    return pd.read_csv(path, dtype=dtype)
