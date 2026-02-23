"""Project configuration and paths for depo_paper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "output"

CONFIG_FILE = ROOT / "local_config.py"
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

CONFIG_PATHS = module.DATA_PATHS
REQUIRED_KEYS = {"dst_raw", "crime", "formats", "disced"}

missing = REQUIRED_KEYS - CONFIG_PATHS.keys()
extra = CONFIG_PATHS.keys() - REQUIRED_KEYS
if missing:
    raise RuntimeError(f"Missing required DATA_PATHS keys: {missing}")
if extra:
    raise RuntimeError(f"Unexpected DATA_PATHS keys: {extra}")


def _resolve_path(path_like: str | Path) -> Path:
    """Resolve and validate a configured filesystem path."""
    resolved = Path(path_like).expanduser().resolve()
    if not resolved.exists():
        raise RuntimeError(f"Configured path does not exist: {resolved}")
    return resolved


@dataclass(frozen=True)
class Paths:
    """Container for project directories and external data locations."""

    root: Path
    temp: Path
    output: Path
    figures: Path
    tables: Path
    dst: Path
    crime: Path
    formats: Path
    disced: Path

    def ensure_dirs(self) -> None:
        """Create output directories used by the analysis pipeline."""
        for dir_path in (self.temp, self.output, self.figures, self.tables):
            dir_path.mkdir(parents=True, exist_ok=True)


PATHS = Paths(
    root=ROOT,
    temp=ROOT / "temp",
    output=OUTPUT,
    figures=OUTPUT / "figures",
    tables=OUTPUT / "tables",
    dst=_resolve_path(CONFIG_PATHS["dst_raw"]),
    crime=_resolve_path(CONFIG_PATHS["crime"]),
    formats=_resolve_path(CONFIG_PATHS["formats"]),
    disced=_resolve_path(CONFIG_PATHS["disced"]),
)

PATHS.ensure_dirs()


def setup_matplotlib() -> None:
    """Apply a shared Matplotlib style used in generated paper figures."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )


__all__ = ["PATHS", "setup_matplotlib"]
