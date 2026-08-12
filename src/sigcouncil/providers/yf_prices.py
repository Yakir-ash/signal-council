"""Primary price provider: yfinance (unofficial Yahoo API).

Treated as replaceable and fallible by design: batched downloads, defensive
normalization, and every row stamped source='yfinance' so the quality gate can
cross-validate against Stooq.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import pandas as pd

from ..logutil import get_logger
from .base import PriceProvider

log = get_logger("yf")

_COLS = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume", "source"]


class YFinancePrices(PriceProvider):
    name = "yfinance"

    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size

    def daily_ohlcv(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        import yfinance as yf

        frames: list[pd.DataFrame] = []
        for i in range(0, len(tickers), self.batch_size):
            batch = tickers[i : i + self.batch_size]
            try:
                raw = yf.download(
                    " ".join(batch), start=start.isoformat(), end=end.isoformat(),
                    auto_adjust=False, actions=False, group_by="ticker",
                    progress=False, threads=True,
                )
            except Exception as e:  # noqa: BLE001
                log.warning("batch %d download failed: %s", i, e)
                continue
            if raw is None or raw.empty:
                continue
            if not isinstance(raw.columns, pd.MultiIndex):  # single-ticker shape
                raw = pd.concat({batch[0]: raw}, axis=1)
            for t in batch:
                if t not in raw.columns.get_level_values(0):
                    continue
                df = raw[t].dropna(how="all")
                if df.empty:
                    continue
                out = pd.DataFrame({
                    "ticker": t,
                    "date": pd.to_datetime(df.index).tz_localize(None).normalize(),
                    "open": df["Open"].values,
                    "high": df["High"].values,
                    "low": df["Low"].values,
                    "close": df["Close"].values,
                    "adj_close": df["Adj Close"].values if "Adj Close" in df else df["Close"].values,
                    "volume": df["Volume"].values,
                })
                out["source"] = self.name
                frames.append(out)
        if not frames:
            return pd.DataFrame(columns=_COLS)
        res = pd.concat(frames, ignore_index=True)
        res = res.dropna(subset=["close"])
        res["ingested_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        return res
