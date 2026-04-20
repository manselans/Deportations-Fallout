# Deportation's Fallout: Evidence from Denmark - Replication Package

This repository contains the replication code and supporting materials for the paper "Deportation's Fallout: Evidence from Denmark".

---


## Authors

- Mike Light <sup>1,*</sup>
- Lars Højsgaard Andersen<sup>2</sup>
- Noa Hendel<sup>2,†</sup>

Affiliations:

1. University of Wisconsin-Madison
2. ROCKWOOL Foundation

\* Corresponding author (paper-related questions): [mlight@ssc.wisc.edu](mlight@ssc.wisc.edu)  
† Replication package contact (code/data questions): [nhe@rff.dk](nhe@rff.dk)

- Recommended citation: TBD
- DOI / URL: TBD

---

## Quick start

1. Clone the repository:

```bash
git clone https://github.com/manselans/Deportations-Fallout.git
cd Deportations-Fallout
```

2. Install the package and its dependencies:

```bash
pip install .
```

3. Create a local configuration file and update the data paths:

```bash
cp local_config.example.py local_config.py
```

Edit `local_config.py` and set the `DATA_PATHS` mapping to your local directories, for example:

```python
DATA_PATHS = {
    "dst_raw": "/path/to/raw/data",
    "crime": "/path/to/crime/data",
}
```

> **Note**: The underlying microdata are restricted and require approval from Statistics Denmark: https://www.dst.dk/en/TilSalg/data-til-forskning/mikrodataordninger

4. Run the full replication:

```bash
python replicate.py
```

All required output directories (`temp`, `output`, `figures`, `tables`) will be created automatically.

---

## Script-by-script execution

You can also run each section independently:

- `scripts/collect_data.py` — data preparation
- `scripts/run_analysis.py` — main analysis and appendix

Example:

```bash
python scripts/collect_data.py
```

---

## Notebooks

See the `notebooks/` directory for step-by-step walkthroughs of the replication. Notebooks explain key assumptions, intermediate checks, and core analysis logic. The notebooks are also responsible for producing any exact numbers presented in the paper (e.g., test statistics, coefficients and observation counts); `replicate.py` only recreates tables and figures. 

To install optional notebook dependencies:

```bash
pip install .[notebooks]
```

---

## Environment

### Python version

The code was executed using Python 3.x. Exact package versions used to produce the paper are listed in `requirements.txt`.

### Dependency specifications

This repository provides two dependency specifications:

- `requirements.txt` — frozen versions used to produce the paper. Use for strict replication:

  ```bash
  pip install -r requirements.txt
  ```

- `pyproject.toml` — project metadata and flexible dependencies; useful for development or editable installs.

---

## License and data restrictions

- Code: MIT License (see `LICENSE`)
- Data: Restricted-access administrative data from Statistics Denmark; data cannot be redistributed.

---

## Versioning

This repository reflects the code state used for the paper at the time of submission.
