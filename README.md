# Deportation's Fallout: Evidence from Denmark - Replication Package

This repository contains the replication code for the paper **Deportation's Fallout: Evidence from Denmark**.

---

## Authors and Details

- [Mike Light](mlight@ssc.wisc.edu)* - University of Wisconsin-Madison
- [Lars Højsgaard Andersen](lha@rff.dk) - ROCKWOOL Foundation
- [Noa Hendel](nhe@rff.dk)† - ROCKWOOL Foundation

\* Corresponding author (paper-related questions)  
† Replication package contact (code/data questions)

- **Recommended Citation:** TBD
- **DOI / URL:** TBD

---

## Quick start

1. **Clone the repository**

```bash
git clone https://github.com/manselans/Deportations-Fallout.git
cd Deportations-Fallout
```

2. **Install dependencies**

```bash
pip install .
```

3. **Create local configuration**

```bash
cp local_config.example.py local_config.py
```

Edit `local_config.py` to point to your local data paths:

```python
DATA_PATHS = {
    "RAW": "/path/to/raw/data",
    "CRIME": "/path/to/crime/data",
    "FORMATS": "/path/to/formats",
}
```

> **Note:** The underlying microdata are restricted and require approval from Statistics Denmark.  
> See: https://www.dst.dk/en/TilSalg/data-til-forskning/mikrodataordninger  
> Once access has been granted, update `local_config.py` to reflect your approved data locations.

4. **Run the full replication**

```bash
python replicate.py
```

All required output directories (`temp`, `output`, `figures`, `tables`) are created automatically.

---

## Script-by-script execution

You can also run each section independently:

- `scripts/collect_data.py` — data preparation
- `scripts/run_analysis.py` — main analysis
- `scripts/appendix.py` — appendices

Example:

```bash
python scripts/collect_data.py
```

---

## Notebooks

For a step-by-step replication walkthrough, see the notebooks in `notebooks/`.  
These notebooks are primarily intended for reviewers and readers. They document key assumptions, intermediate checks, and core code logic in a transparent and reproducible way.

The notebooks require optional dependencies. To install them, run:

```bash
pip install .[notebooks]
```

---

## Repository Structure

```
scripts/                          # Pipeline entry points
src/depo_paper/build/              # Data construction modules
src/depo_paper/analysis/           # Figure and table generation
src/depo_paper/tools/              # Utilities (I/O, estimation, plotting)
src/depo_paper/config.py           # Project paths and matplotlib configuration
```

---

---

## Environment

### Python version

The code was run using **Python 3.x**.  
Exact package versions used for the paper are listed in `requirements.txt`.

### Dependency specifications

This repository includes two dependency specifications:

- `requirements.txt`  
  → Exact, frozen versions used to produce the paper  
  → Use for strict replication:

  ```bash
  pip install -r requirements.txt
  ```

- `pyproject.toml`  
  → Project metadata and logical dependencies  
  → Useful for development or editable installs

---

## Stata Code (Appendix B)

The McCrary density test reported in Appendix B is implemented in Stata.

The corresponding `.do` file is located at:

```
src/depo_paper/mccrary.do
```

Stata is **not required** for the main replication pipeline.

---

## License and Data Restrictions

- **Code:** MIT License (see `LICENSE`)
- **Data:** Restricted-access administrative data from Statistics Denmark (cannot be redistributed)

---

## Versioning

This repository reflects the exact code state used for the paper.  
A tagged release corresponds to the submitted manuscript.
