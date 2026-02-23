#!/usr/bin/env python3
"""Validate local setup and expected raw-data file presence without loading datasets."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "local_config.py"
REQUIRED_KEYS = {"dst_raw", "crime", "formats", "disced"}


def _load_data_paths() -> dict:
    if not CONFIG_FILE.exists():
        raise RuntimeError(
            "Missing local_config.py. Copy local_config.example.py and define DATA_PATHS."
        )

    spec = importlib.util.spec_from_file_location("local_config", CONFIG_FILE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    if not hasattr(module, "DATA_PATHS"):
        raise RuntimeError("local_config.py must define a DATA_PATHS dictionary.")

    config_paths = module.DATA_PATHS
    missing = REQUIRED_KEYS - config_paths.keys()
    extra = config_paths.keys() - REQUIRED_KEYS

    if missing:
        raise RuntimeError(f"Missing required DATA_PATHS keys: {missing}")
    if extra:
        raise RuntimeError(f"Unexpected DATA_PATHS keys: {extra}")

    resolved = {}
    for key, value in config_paths.items():
        path = Path(value).expanduser().resolve()
        if not path.exists():
            raise RuntimeError(f"Configured path does not exist: {path}")
        resolved[key] = path

    return resolved


def _expected_files(paths: dict[str, Path]) -> list[Path]:
    """Return expected external raw files used by the replication pipeline."""

    dst = paths["dst_raw"]
    crime = paths["crime"]

    files = [
        dst / "Eksterne data/rf0961502_udrejseforbud.sas7bdat",
        dst / "vnds2021.dta",
        crime / "krin_placering.dta",
    ]

    # Yearly registers loaded in build modules
    for year in range(1997, 2021):
        files.append(dst / f"ophg{year}.dta")

    for year in range(1980, 2022):
        files.append(dst / f"kraf{year}.dta")

    for year in range(1998, 2022):
        files.append(dst / f"krsi{year}.dta")

    for year in range(1999, 2022):
        files.append(dst / f"ind{year}.dta")

    for year in range(2008, 2022):
        files.append(dst / f"bfl{year}.dta")
        files.append(dst / f"ilme12_{year}.dta")

    # Population registers (bef12) are loaded by population/female_foreign modules.
    # The required year interval depends on observed conviction years in raw data.
    # We validate the full paper-relevant period used across scripts.
    for year in range(1999, 2022):
        files.append(dst / f"bef12_{year}.dta")

    return files


def main() -> int:
    print("Validating local configuration and expected raw file presence (dry-run; no data loading)...")

    try:
        paths = _load_data_paths()
    except Exception as exc:
        print(f"❌ Config validation failed: {exc}")
        return 1

    print("✅ Config is valid.")
    for key, value in paths.items():
        print(f"  - {key}: {value}")

    missing = [p for p in _expected_files(paths) if not p.exists()]

    if missing:
        print(f"❌ Missing expected files: {len(missing)}")
        preview = missing[:25]
        for item in preview:
            print(f"  - {item}")
        if len(missing) > len(preview):
            print(f"  ... and {len(missing) - len(preview)} more")
        return 1

    print("✅ All expected file paths were found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
