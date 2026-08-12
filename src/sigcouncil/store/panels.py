"""Parquet panel store + the point-in-time (PIT) access rule.

THE core integrity rule of the whole system lives here:

    pit_filter(df, as_of) -> only rows whose observed_at (or `filed`) is <= as_of.

Features and models never read raw fundamentals directly — they go through
`pit_filter`. tests/test_pit.py plants deliberate leaks and fails the build if
this function lets them through.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from ..paths import PANELS

PRICES = PANELS / "prices.parquet"
FUNDAMENTALS = PANELS / "fundamentals.parquet"
MACRO = PANELS / "macro.parquet"
INSIDERS = PANELS / "insiders.parquet"
FEATURES = PANELS / "features.parquet"


def write_panel(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read_panel(path: Path) -> pd.DataFrame:
    if not Path(path).exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def upsert_prices(new: pd.DataFrame) -> pd.DataFrame:
    """Merge new price rows into the panel; last write wins per (ticker,date,source)."""
    old = read_panel(PRICES)
    if old.empty:
        merged = new.copy()
    else:
        merged = pd.concat([old, new], ignore_index=True)
    merged["date"] = pd.to_datetime(merged["date"])
    merged = merged.drop_duplicates(subset=["ticker", "date", "source"], keep="last")
    merged = merged.sort_values(["ticker", "date"]).reset_index(drop=True)
    write_panel(merged, PRICES)
    return merged


def primary_prices(panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """Single price series per (ticker,date): prefer yfinance, fall back to stooq."""
    df = panel if panel is not None else read_panel(PRICES)
    if df.empty:
        return df
    order = pd.CategoricalDtype(["yfinance", "stooq"], ordered=True)
    df = df.copy()
    df["source"] = df["source"].astype(order)
    df = df.sort_values(["ticker", "date", "source"])
    return df.drop_duplicates(subset=["ticker", "date"], keep="first")


def pit_filter(df: pd.DataFrame, as_of: date | str,
               observed_col: str = "filed") -> pd.DataFrame:
    """Return only rows observable on or before `as_of`. The one true PIT gate."""
    if df.empty or observed_col not in df.columns:
        return df
    ts = pd.to_datetime(df[observed_col], errors="coerce")
    mask = ts <= pd.Timestamp(as_of)
    return df[mask.fillna(False)]


def close_matrix(prices: pd.DataFrame, price_col: str = "adj_close") -> pd.DataFrame:
    """Wide matrix date x ticker of (adjusted) closes."""
    return prices.pivot_table(index="date", columns="ticker", values=price_col,
                              aggfunc="last").sort_index()
