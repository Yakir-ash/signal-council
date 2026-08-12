"""Stooq daily OHLCV — independent second price source.

Roles: (1) daily cross-validation of yfinance closes (quality gate),
(2) fallback if Yahoo breaks. Free, no key, CSV endpoint.
Note: Stooq provides split-adjusted (not dividend-adjusted) prices; the quality
gate compares raw closes on recent days where adjustment differences are nil.
"""
from __future__ import annotations

import io
from datetime import date, datetime, timezone

import pandas as pd

from ..logutil import get_logger
from .base import PriceProvider
from .http import Http

log = get_logger("stooq")


class StooqPrices(PriceProvider):
    name = "stooq"

    def __init__(self):
        self.http = Http(min_interval=0.25)

    @staticmethod
    def _symbol(ticker: str) -> str:
        return ticker.lower().replace(".", "-") + ".us"

    def daily_ohlcv(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        frames = []
        for t in tickers:
            try:
                r = self.http.get(
                    "https://stooq.com/q/d/l/",
                    params={"s": self._symbol(t), "i": "d",
                            "d1": start.strftime("%Y%m%d"), "d2": end.strftime("%Y%m%d")},
                    cache_ttl=6 * 3600,
                )
                df = pd.read_csv(io.StringIO(r.text))
            except Exception as e:  # noqa: BLE001
                log.warning("%s failed: %s", t, e)
                continue
            if df.empty or "Close" not in df.columns:
                continue
            out = pd.DataFrame({
                "ticker": t,
                "date": pd.to_datetime(df["Date"]),
                "open": df["Open"], "high": df["High"], "low": df["Low"],
                "close": df["Close"],
                "adj_close": df["Close"],  # stooq is split-adjusted only
                "volume": df.get("Volume", pd.Series(dtype=float)),
                "source": self.name,
            })
            frames.append(out)
        if not frames:
            return pd.DataFrame(columns=["ticker", "date", "open", "high", "low",
                                         "close", "adj_close", "volume", "source"])
        res = pd.concat(frames, ignore_index=True).dropna(subset=["close"])
        res["ingested_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return res
