# Deportation's Fallout: Evidence from Denmark — Replication Package

This repository contains the replication code for *Deportation's Fallout: Evidence from Denmark* by Mike Light, Lars Højsgaard Andersen, and Noa Hendel.

The project is organized as a standard Python package using `pyproject.toml` and a `src/` layout.

## Repository structure

- `src/build/`: data construction pipeline
- `src/analysis/`: generation of figures and tables
- `src/tools/`: project utilities, path management, and plotting/table helpers
- `requirements.txt`: frozen dependency versions used for the paper
- `pyproject.toml`: package metadata and development dependency specification

## Data access and paths (important)

The underlying microdata come from **Statistics Denmark** and cannot be shared in this repository.

Before running the pipeline, edit `src/tools/paths.py` and point it to your local copies of the restricted data.

Example:

```python
# src/tools/paths.py

dst = Path(r"/absolute/path/to/raw/data")
crime = Path(r"/absolute/path/to/crime/registers")
```

## Environment setup

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

For strict replication with frozen versions:

```bash
pip install -r requirements.txt
```

For editable package usage:

```bash
pip install -e .
```

## Running the replication pipeline

Run from repository root:

```bash
run-data
run-analysis
```

Equivalent module entry points:

```bash
python -m build.run_data
python -m analysis.run_analysis
```

## Stata code

Appendix B includes a McCrary test implemented in Stata (`src/mccrary.do`).

## License and data restrictions

- **Code**: MIT License (see `LICENSE`)
- **Data**: Restricted-access administrative data from Statistics Denmark (cannot be redistributed)
