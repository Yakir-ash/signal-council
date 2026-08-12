"""FRED macro series via the public fredgraph.csv endpoint (no API key).

Series used by the regime engine:
  DGS10        10y treasury yield
  DGS2         2y treasury yield
  T10Y2Y       10y-2y spread (curve)
  BAMLH0A0HYM2 high-yield OAS (credit stress)
  T10YIE       10y breakeven inflation
  VIXCLS       VIX close
"""
from __future__ import annotations

import io
from datetime import date

import pandas as pd

from ..logutil import get_logger
from .base import MacroProvider
from .http import Http

log = get_logger("fred")


class Fred(MacroProvider):
    name = "fred"

    def __init__(self):
        self.http = Http(min_interval=0.3)

    def series(self, series_id: str, start: date) -> pd.DataFrame:
        try:
            r = self.http.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                              params={"id": series_id}, cache_ttl=6 * 3600)
            df = pd.read_csv(io.StringIO(r.text))
        except Exception as e:  # noqa: BLE001
            log.warning("%s failed: %s", series_id, e)
            return pd.DataFrame(columns=["date", "value", "series", "source"])
        date_col = df.columns[0]  # 'DATE' or 'observation_date' depending on vintage
        df = df.rename(columns={date_col: "date", series_id: "value"})
        df["date"] = pd.to_datetime(df["date"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        df = df[df["date"] >= pd.Timestamp(start)]
        df["series"] = series_id
        df["source"] = self.name
        return df[["date", "value", "series", "source"]]
