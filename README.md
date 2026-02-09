# Deportation's Fallout: Evidence from Denmark
Replication package for the paper Deportation's Fallout (2026) by Mike Light, Lars Højsgaard Andersen and Noa Hendel.

## Data
The data used in this project are from Statistics Denmark (DST) and may not be published. 

To replicate the results, the user must have access to the raw data on their system
and set the path to the raw data directory via `src/tools/paths.py`.

## Replication
0. Go to `src/tools/paths.py` and edit data paths.
1. Create a virtual environment.
2. Install dependencies and local functions: ```pip install -e .```.
3. Run /src/build/run_data.py to recreate data.
4. Run /src/analysis/run_analysis.py to recreate tables and figures.
5. To obtain results from the McCrary test conducted in Appendix B, STATA is required; use /src/mccrary.do.  

