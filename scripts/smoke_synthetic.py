#!/usr/bin/env python3
"""Offline smoke test: synthetic-but-realistic panels → full daily pipeline →
dashboard render. Verifies the engine end-to-end without network access.
NOT a substitute for CI runs on real data — a wiring check.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from sigcouncil.store import panels, universe

rng = np.random.default_rng(42)

N_DAYS = 460
tickers = universe.current_stocks()[:60]
etfs = universe.etfs()
syms = sorted(set(tickers + etfs))
dates = pd.bdate_range(end=pd.Timestamp.utcnow().tz_localize(None).normalize(), periods=N_DAYS)

rows = []
for t in syms:
    drift = rng.normal(0.0004, 0.0004)
    vol = rng.uniform(0.01, 0.03)
    r = rng.normal(drift, vol, N_DAYS)
    close = 50 * np.exp(np.cumsum(r)) * rng.uniform(0.5, 5)
    openp = close * (1 + rng.normal(0, 0.004, N_DAYS))
    hi = np.maximum(openp, close) * (1 + abs(rng.normal(0, 0.005, N_DAYS)))
    lo = np.minimum(openp, close) * (1 - abs(rng.normal(0, 0.005, N_DAYS)))
    volu = rng.integers(1_000_000, 30_000_000, N_DAYS)
    rows.append(pd.DataFrame({"ticker": t, "date": dates, "open": openp, "high": hi,
                              "low": lo, "close": close, "adj_close": close,
                              "volume": volu, "source": "yfinance"}))
    # second source with tiny noise for cross-validation path
    rows.append(pd.DataFrame({"ticker": t, "date": dates, "open": openp, "high": hi,
                              "low": lo, "close": close * (1 + rng.normal(0, 0.001, N_DAYS)),
                              "adj_close": close, "volume": volu, "source": "stooq"}))
px = pd.concat(rows, ignore_index=True)
px["ingested_at"] = "2026-08-12T00:00:00Z"
panels.write_panel(px, panels.PRICES)

# fundamentals: 10 quarters for each of the first 40 tickers
frows = []
for t in tickers[:40]:
    base_rev = rng.uniform(1e9, 2e10)
    g = rng.normal(0.02, 0.03)                       # q/q growth
    for q in range(10):
        pe = pd.Timestamp("2024-03-31") + pd.DateOffset(months=3 * q)
        ps = pe - pd.Timedelta(days=89)
        filed = pe + pd.Timedelta(days=35)
        rev = base_rev * (1 + g) ** q * rng.uniform(0.98, 1.02)
        ni = rev * rng.uniform(0.05, 0.25)
        for concept, val in [("revenue", rev), ("net_income", ni),
                             ("gross_profit", rev * rng.uniform(0.3, 0.6)),
                             ("operating_income", rev * rng.uniform(0.1, 0.3)),
                             ("operating_cf", ni * rng.uniform(1.0, 1.4)),
                             ("capex", rev * rng.uniform(0.02, 0.08)),
                             ("eps_diluted", ni / 1e9)]:
            frows.append({"ticker": t, "concept": concept, "tag": concept, "unit": "USD",
                          "period_start": str(ps.date()), "period_end": str(pe.date()),
                          "value": val, "form": "10-Q", "filed": str(filed.date()),
                          "fiscal_frame": None, "source": "synthetic"})
        for concept, val in [("assets", rev * 2), ("equity", rev * 0.8),
                             ("cash", rev * 0.3), ("long_term_debt", rev * 0.4),
                             ("shares_outstanding_dei", 1e9)]:
            frows.append({"ticker": t, "concept": concept, "tag": concept, "unit": "USD",
                          "period_start": None, "period_end": str(pe.date()),
                          "value": val, "form": "10-Q", "filed": str(filed.date()),
                          "fiscal_frame": None, "source": "synthetic"})
panels.write_panel(pd.DataFrame(frows), panels.FUNDAMENTALS)

# macro
mrows = []
for s, level, noise in [("DGS10", 4.2, 0.03), ("DGS2", 3.9, 0.03), ("T10Y2Y", 0.3, 0.02),
                        ("BAMLH0A0HYM2", 3.4, 0.05), ("T10YIE", 2.3, 0.02), ("VIXCLS", 16, 1.0)]:
    vals = level + np.cumsum(rng.normal(0, noise, N_DAYS)) * 0.1
    mrows.append(pd.DataFrame({"date": dates, "value": vals, "series": s, "source": "fred"}))
panels.write_panel(pd.concat(mrows, ignore_index=True), panels.MACRO)

# insiders: a cluster buy on one name
ins = pd.DataFrame([
    {"ticker": tickers[5], "filer": f"Officer {i}", "role": "CFO", "tx_date": "2026-07-20",
     "filed_at": "2026-07-22", "kind": "P", "shares": 10000, "price": 50.0, "value": 500000.0}
    for i in range(3)])
panels.write_panel(ins, panels.INSIDERS)

from sigcouncil.pipeline.daily import run_daily
pack = run_daily(skip_fetch=True)
print("\n--- PACK SUMMARY ---")
print("regime:", pack["regime"]["label"] if pack["regime"] else None)
print("no_high_conviction:", pack["no_high_conviction"])
print("top:", [t["ticker"] for t in pack["top_opportunities"]])
print("avoid:", [t["ticker"] for t in pack["avoid_list"]])
print("quality reasons:", pack["quality"]["reasons"])

from sigcouncil.report.html import render_latest
print("dashboard:", render_latest())
