# Deportation's Fallout: Evidence from Denmark - Replication Package

This repository contains the replication code for the project *Deportation's Fallout: Evidence from Denmark* 
by Mike Light, Lars Højsgaard Andersen and Noa Hendel.  

The main analysis pipeline is written in Python, with a small number of auxiliary Stata scripts.

The repository is structured to allow full replication of all results, conditional on access to the underlying data.

## Repository structure

src/build/ # data construction pipeline
src/analysis/ # generation of figures and tables
src/tools/ # paths, helpers, utilities

## Data access and paths (IMPORTANT)

The underlying microdata come from **Statistics Denmark** and **cannot be shared** in this repository.

To run the code, you must edit:

src/tools/paths.py

and point it to your **local copies** of the restricted data.

Example:

```python
# src/tools/paths.py

dst = Path(r"/absolute/path/to/raw/data")
crime = Path(r"/absolute/path/to/crime/registers")

```
## Environment setup

### Python version

The code was run using **Python 3.x** (see `requirements.txt` for package versions).

### Dependencies (important distinction)

This repository uses **two dependency specifications**:

- `requirements.txt`  
  → **Exact, frozen versions** used to produce the paper  
  → Use this for strict replication

- `pyproject.toml`  
  → Project metadata and logical dependencies  
  → Useful for development or editable installs

### Recommended (exact replication)

Install exact versions used in the paper:

`pip install -r requirements.txt`

Optional (development tools, e.g. Jupyter):

`pip install .[dev]`

## Replication steps

0. Go to `src/tools/paths.py` and edit data paths.
1. Create a virtual environment.
2. Install dependencies and local functions: ```pip install -e .```.
3. Run /src/build/run_data.py to recreate data.
4. Run /src/analysis/run_analysis.py to recreate tables and figures.
5. To obtain results from the McCrary test conducted in Appendix B, STATA is required; use /src/mccrary.do.  

## Stata code

A McCrary test is implemented in Stata.  
The corresponding `.do` file is included but is **not required** for the main replication.

## License and data restrictions

- **Code**: MIT License (see `LICENSE`)
- **Data**: Restricted-access administrative data from Statistics Denmark  (cannot be redistributed)

## Notes

This repository reflects the exact code state used for the paper.  
A tagged release corresponds to the submitted manuscript.