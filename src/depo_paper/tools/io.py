"""
IO helpers for loading Stata .dta files into pandas.

Can load a single or multiple files into a concatenated dataframe or as items in a dictionary.

Rows and columns can be filtered directly based on a wide range of conditions. 
"""

import inspect
from collections.abc import Iterable
from pathlib import Path

import pandas as pd


def fetch(
    path, columns=None, *, filters=None, pairs=None, pair_cols=None, **read_kwargs
):
    """
    Load a .dta file with optional columns and row selection

    Parameters
    ----------
    path : str
        Path to .dta file.
    columns : list[str] | None
        Columns from .dta to return. If None, returns all.
    filters : dict[str, (scalar |iterable | callable)] | None
        Column-wise filters combined with AND:
            - scalar -> equality (==)
            - iterable -> isin(...)
            - callable -> boolean mask via col.apply(fn) or fn(series)
    pairs : Iterable[tuple] | None
        Exact combinations to match (e.g., list of (id, date)). ANDed with `filters`.
    pair_cols : list[str] | None
        Column names corresponding to each element of tuples in `pairs`.
    **read_kwargs
        Keyword arguments passed to pd.read_stata().
    """

    filters = filters or {}

    # Columns to be read from .dta: Requested + any needed in filters/pairs
    need_cols = set(columns or [])
    need_cols.update(filters.keys())
    if pairs is not None:
        if not pair_cols or not isinstance(pair_cols, (list, tuple)):
            raise ValueError("When using `pairs`, provide `pair_cols` as a list/tuple.")
        need_cols.update(pair_cols)

    # Read minimal set; if none specified, load all
    usecols = None if columns is None else list(need_cols)

    read_kwargs.setdefault("convert_categoricals", False)
    df = pd.read_stata(path, columns=usecols, **read_kwargs)

    # Build mask
    if df.shape[0]:
        mask = pd.Series(True, index=df.index)
    else:
        mask = pd.Series([], dtype=bool)

    # Column-wise filters
    for col, rule in filters.items():
        if callable(rule):
            # Allow vectorized callables: fn(series) -> bool Series
            sig = inspect.signature(rule)
            out = rule(df[col]) if len(sig.parameters) == 1 else df[col].apply(rule)
            mask &= out.astype(bool)
        elif isinstance(rule, Iterable) and not isinstance(rule, (str, bytes)):
            mask &= df[col].isin(rule)
        else:
            mask &= df[col] == rule

    # Exact pair filtering
    if pairs is not None:
        # MultiIndex membership
        mi = pd.MultiIndex.from_frame(df[pair_cols])
        pair_set = set(pairs)
        mask &= mi.isin(pair_set)

    df = df[mask]

    # Trim to requested columns if provided
    if columns is not None:
        df = df[list(columns)]

    return df.reset_index(drop=True)


def gather(
    path,
    names=None,
    *,
    concatenate=False,
    noisily=False,
    add_name=None,
    add_values=None,
    file_pattern="{name}.dta",
    recursive=False,
    **fetch_kwargs,
):
    """
    Load multiple .dta files via `fetch()` and return a dict[name->df] or a single concatenated df.

    Parameters
    ----------
    path : str
        Directory containing .dta file, a glob-able base, or a single .dta file.
    names : sequence[str] | None
        Dataset names to load. If None, auto-discovers *.dta under `path` (or uses the single file).
    concatenate : bool, default False
        If True, return a single concatenated DataFrame; otherwise a dict of DataFrames.
    noisily: bool, default False
        If True, print name of currently loading file to show progress.
    add_name : str | None
        Optional column name added to each df (e.g., "year") so you can track provenance.
    add_values : sequence | None
        Values to put in `add_name`. If not provided, uses the dataset's name.
        Length must match number of datasets if provided.
    file_pattern : str, default "{name}.dta"
        Pattern to resolve filenames when `names`is given (joined with `path`).
    recursive : bool, default False
        If auto-discovering, whether to search subfolders.
    **fetch_kwargs
        Forwarded to `fetch()` (e.g., columns = ..., filters = ..., etc.).

    Returns
    -------
    dict[name, DataFrame] or DataFrame

    """

    base = Path(path)

    # Resolve files
    if names:
        files = {n: (base / file_pattern.format(name=n)) for n in names}
    else:
        if base.is_file():
            files = {base.stem: base}
        else:
            pattern = "**/*.dta" if recursive else "*.dta"
            found = sorted(base.glob(pattern))
            if not found:
                raise FileNotFoundError(f"No .dta files found under {base}")
            files = {p.stem: p for p in found}

    # Check existence of files
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))

    # Optional values for add_name
    if (
        add_name is not None
        and add_values is not None
        and len(add_values) != len(files)
    ):
        raise ValueError("Length of add-values must match number of datasets.")

    # Load
    out = {}
    for i, (name, fpath) in enumerate(files.items()):
        if noisily:
            print(f"Loading {fpath}...")
        df = fetch(str(fpath), **fetch_kwargs)
        if add_name is not None:
            value = add_values[i] if add_values is not None else name
            df[add_name] = value
        out[name] = df

    if concatenate:
        return pd.concat(out.values(), ignore_index=True)

    return out


def load_csv(name: str) -> pd.DataFrame:
    """
    Load external csv-data from /src/depo_paper/tools/data folder.

    Parameters
    ----------
    name
        Name of .csv in src/depo_paper/tools/data

    Returns
    -------
    pandas.DataFrame
    """

    data_dir = Path(__file__).resolve().parent / "data"
    return pd.read_csv(data_dir / name)