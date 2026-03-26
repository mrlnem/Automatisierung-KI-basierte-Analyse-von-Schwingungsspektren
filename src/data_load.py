from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pandas as pd


def discover_csvs(data_root: Path, cutters) -> List[Tuple[Path, str]]:
    """
    Find CSV files under cutter folders (c1..c6).
    Works with kagglehub cache directory layouts.
    """
    refs: List[Tuple[Path, str]] = []
    data_root = Path(data_root)

    for c in cutters:
        cutter_dirs = [p for p in data_root.rglob(c) if p.is_dir() and p.name == c]
        for c_dir in cutter_dirs:
            for csv in sorted(c_dir.rglob("*.csv")):
                refs.append((csv, c))

    return refs


def load_signals(csv_path: Path) -> pd.DataFrame:
    """
    Robust CSV loader:
    - reads CSV
    - coerces everything to numeric (non-numeric -> NaN)
    - drops empty columns/rows
    """
    df = pd.read_csv(csv_path)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")

    return df


def extract_vibration_xyz(df: pd.DataFrame):
    """
    Extract 3 vibration channels (vx, vy, vz).

    Strategy:
    1) If there are >=6 numeric columns -> use cols[3], cols[4], cols[5] 
    2) Else if there are >=3 numeric columns -> use the last three numeric columns
    3) Else -> raise
    """
    if df is None or df.empty:
        raise ValueError("Empty dataframe after loading/cleaning")

    cols = list(df.columns)

    if len(cols) >= 6:
        return (
            df[cols[3]].to_numpy(),
            df[cols[4]].to_numpy(),
            df[cols[5]].to_numpy(),
        )

    if len(cols) >= 3:
        last3 = cols[-3:]
        return (
            df[last3[0]].to_numpy(),
            df[last3[1]].to_numpy(),
            df[last3[2]].to_numpy(),
        )

    raise ValueError(f"CSV has too few usable numeric columns: {cols}")
