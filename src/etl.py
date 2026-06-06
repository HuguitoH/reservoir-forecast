"""
ETL functions for Madrid reservoir capacity data.

Pipeline:
    load_raw_csvs()   → raw combined DataFrame
    clean_dataframe() → cleaned long-format DataFrame
    build_pivot()     → wide-format pivot with datetime index and total_hm3
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# Constants

MONTH_ORDER: list[str] = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]

MONTH_TO_INT: dict[str, int] = {m: i + 1 for i, m in enumerate(MONTH_ORDER)}

# Reservoir excluded — data ends in 2018, insufficient for time series
EXCLUDED_RESERVOIRS: list[str] = ["RioLosMorales_LosMorales"]

# Reservoir with all year values missing — reconstructed manually
MISSING_YEAR_RESERVOIR: str = "RioLozoya_Riosequillo"

YEAR_MIN: int = 1998
YEAR_MAX: int = 2021

# Data ends in March 2021 for all reservoirs
FINAL_YEAR_MONTHS: list[str] = ["enero", "febrero", "marzo"]

INVALID_MONTHS: list[str] = [
    "", " ", "nan",
    "a partir del 1 de octubre, comienzo del nuevo aã±o hidrolã³gico, "
    "no se considera este embalse, ya que no es del canal de isabel ii sa",
]


# Functions

def load_raw_csvs(data_dir: Path) -> pd.DataFrame:
    """
    Load and combine all reservoir CSVs from data_dir.

    Each CSV has a BOM-prefixed 'anio' column, semicolon separator,
    European decimal commas, and trailing empty columns.
    Reservoir name is derived from the filename.

    Returns a raw combined DataFrame with columns:
        year, month, capacity_hm3, reservoir
    """
    dfs = []
    for path in sorted(data_dir.glob("*.csv")):
        reservoir = path.stem.replace("AguaEmbalsada_", "").strip()

        # Try utf-8-sig first — handles BOM automatically
        # Fall back to latin-1 for files with European characters
        for encoding in ("utf-8-sig", "latin-1", "ISO-8859-1"):
            try:
                df = pd.read_csv(path, sep=";", encoding=encoding)
                break
            except (UnicodeDecodeError, Exception):
                continue

        df["reservoir"] = reservoir
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(f"No CSVs found in {data_dir}")

    df = pd.concat(dfs, ignore_index=True)

    # Strip BOM from column names, rename, drop trailing empty columns
    df.columns = df.columns.str.encode("utf-8").str.decode("utf-8-sig").str.strip()
    df = df.rename(columns={"anio": "year", "mes": "month", "hec_cub": "capacity_hm3"})
    junk = [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=junk + ["anio"], errors="ignore")
    df = df.loc[:, ~df.columns.duplicated()]
    df["reservoir"] = df["reservoir"].str.replace("AguaEmbalsada_", "", regex=False).str.strip()

    return df[["year", "month", "capacity_hm3", "reservoir"]]


def _clean_year(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["year"] = pd.to_numeric(out["year"], errors="coerce")

    # RioLozoya_Riosequillo — reconstruct year sequence
    if MISSING_YEAR_RESERVOIR in out["reservoir"].values:
        riosequillo_mask = out["reservoir"] == MISSING_YEAR_RESERVOIR
        riosequillo_idx  = out.loc[riosequillo_mask].index[:288]
        extra_idx        = out.loc[riosequillo_mask].index[288:]
        out = out.drop(extra_idx)
        years_sequence = [y for y in range(YEAR_MIN, YEAR_MAX + 1) for _ in range(12)]
        out.loc[riosequillo_idx, "year"] = years_sequence

    # Sort by reservoir to keep rows together before ffill
    out = out.sort_values("reservoir", kind="stable").reset_index(drop=True)

    # Forward fill PER RESERVOIR
    out["year"] = out.groupby("reservoir", sort=False)["year"].transform(
        lambda x: x.ffill().bfill()
    )

    # Filter to valid year range
    out = out[out["year"].between(YEAR_MIN, YEAR_MAX)]
    out["year"] = out["year"].astype(int)

    return out


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and standardise the raw combined DataFrame.

    Steps:
        1. Clean year — coerce to numeric, reconstruct missing years for
           RioLozoya_Riosequillo, forward fill remaining gaps
        2. Clean month — lowercase, strip, drop invalid values
        3. Clean capacity — European decimal comma to float
        4. Drop excluded reservoirs (insufficient data)
        5. Truncate to March 2021 (last complete month across all reservoirs)

    Returns a clean long-format DataFrame.
    """
    out = df.copy()

    # 1. Year
    out = _clean_year(out)

    # 2. Month
    out["month"] = out["month"].astype(str).str.lower().str.strip()
    out["month"] = out["month"].replace(INVALID_MONTHS, pd.NA)
    out = out.dropna(subset=["month"])

    # 3. Capacity
    out["capacity_hm3"] = (
        out["capacity_hm3"]
        .astype(str)
        .str.replace(",", ".", regex=False)
        .str.strip()
        .str.replace(" ", "", regex=False)
    )
    out["capacity_hm3"] = pd.to_numeric(out["capacity_hm3"], errors="coerce")

    # 4. Drop excluded reservoirs
    out = out[~out["reservoir"].isin(EXCLUDED_RESERVOIRS)]

    # 5. Truncate to March 2021
    out = out[
        (out["year"] < YEAR_MAX) |
        ((out["year"] == YEAR_MAX) & (out["month"].isin(FINAL_YEAR_MONTHS)))
    ]

    out = out.reset_index(drop=True)
    return out


def build_pivot(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a wide-format pivot table from the clean long-format DataFrame.

    Columns: year, month, ds (datetime), one column per reservoir, total_hm3.
    Rows: one per month (279 rows: Jan 1998 – Mar 2021).

    Requires clean_dataframe() to have been applied first.
    """
    df = df.copy()
    df["month"] = pd.Categorical(df["month"], categories=MONTH_ORDER, ordered=True)

    pivot = df.pivot_table(
        index=["year", "month"],
        columns="reservoir",
        values="capacity_hm3",
        observed=False,
    ).reset_index()

    pivot.columns.name = None

    # Datetime index
    pivot["ds"] = pd.to_datetime(
        pivot["year"].astype(str) + "-" +
        pivot["month"].map(MONTH_TO_INT).astype(str) + "-01"
    )

    reservoir_cols = [c for c in pivot.columns if c not in ["year", "month", "ds"]]
    pivot["total_hm3"] = pivot[reservoir_cols].sum(axis=1, skipna=True)

    # Reorder columns
    pivot = pivot[["ds", "year", "month", "total_hm3"] + reservoir_cols]

    return pivot.sort_values("ds").reset_index(drop=True)
