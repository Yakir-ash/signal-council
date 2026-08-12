"""Universe resolution — current (for daily runs) and point-in-time (for backtests)."""
from __future__ import annotations

import functools
from datetime import date

import pandas as pd

from ..config import universe_cfg
from ..paths import ROOT


@functools.lru_cache(maxsize=1)
def _constituents_df() -> pd.DataFrame:
    cfg = universe_cfg()["groups"]["sp500"]
    df = pd.read_csv(ROOT / cfg["path"])
    df["ticker"] = df[cfg["symbol_col"]].str.strip().str.upper()
    return df


def current_stocks() -> list[str]:
    return sorted(_constituents_df()["ticker"].unique())


def etfs() -> list[str]:
    return list(universe_cfg()["groups"]["core_etfs"]["symbols"])


def regime_only() -> set[str]:
    return set(universe_cfg()["regime_only"])


def scorable_stocks() -> list[str]:
    return [t for t in current_stocks() if t not in regime_only()]


def all_symbols() -> list[str]:
    return sorted(set(current_stocks()) | set(etfs()))


def sector_map() -> dict[str, str]:
    df = _constituents_df()
    return dict(zip(df["ticker"], df["GICS Sector"]))


def sector_etf_map() -> dict[str, str]:
    return dict(universe_cfg()["sector_etf"])


@functools.lru_cache(maxsize=1)
def _historical_df() -> pd.DataFrame:
    cfg = universe_cfg()["groups"]["sp500"]
    df = pd.read_csv(ROOT / cfg["historical_path"])
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date")


def members_as_of(d: date) -> list[str]:
    """Point-in-time index membership for backtests (survivorship mitigation)."""
    df = _historical_df()
    rows = df[df["date"] <= pd.Timestamp(d)]
    if rows.empty:
        return []
    return sorted({t.strip().upper() for t in rows.iloc[-1]["tickers"].split(",")})
