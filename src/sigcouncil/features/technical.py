"""Technical / market-structure features (DESIGN.md §6).

Everything is computed from bars with date <= as_of. Indicators are not used
"because they exist" — each maps to a question the scoring engine actually asks
(trend? confirmation? contraction? liquidity? crowding?).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, n: int = 14) -> float:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    val = 100 - 100 / (1 + rs)
    return float(val.iloc[-1]) if len(val.dropna()) else np.nan


def compute(prices: pd.DataFrame, as_of: pd.Timestamp, benchmark: str = "SPY",
            sectors: dict[str, str] | None = None,
            sector_etf: dict[str, str] | None = None) -> pd.DataFrame:
    """prices: primary long panel (ticker,date,open,high,low,close,adj_close,volume).
    Returns one row per ticker of technical features as of `as_of`."""
    sectors = sectors or {}
    sector_etf = sector_etf or {}
    px = prices[prices["date"] <= as_of]
    closes = px.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last").sort_index()
    raw_close = px.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index()
    opens = px.pivot_table(index="date", columns="ticker", values="open", aggfunc="last").sort_index()
    highs = px.pivot_table(index="date", columns="ticker", values="high", aggfunc="last").sort_index()
    lows = px.pivot_table(index="date", columns="ticker", values="low", aggfunc="last").sort_index()
    vols = px.pivot_table(index="date", columns="ticker", values="volume", aggfunc="last").sort_index()

    closes = closes.tail(320 + 252)
    rets = closes.pct_change()
    bench = closes.get(benchmark)
    bench_r = bench.pct_change() if bench is not None else None

    rows = []
    for t in closes.columns:
        c = closes[t].dropna()
        if len(c) < 60:
            continue
        r = rets[t].dropna()
        last = float(c.iloc[-1])
        f: dict = {"ticker": t, "px_last": float(raw_close[t].dropna().iloc[-1]),
                   "px_last_date": str(c.index[-1].date())}

        def horizon_ret(n: int) -> float:
            return last / float(c.iloc[-n - 1]) - 1 if len(c) > n else np.nan

        f["ret_5"], f["ret_21"], f["ret_63"] = horizon_ret(5), horizon_ret(21), horizon_ret(63)
        f["ret_126"], f["ret_252"] = horizon_ret(126), horizon_ret(252)
        # 12-1 momentum: t-252 .. t-21 (skip the last month — reversal zone)
        f["mom_12_1"] = (float(c.iloc[-22]) / float(c.iloc[-253]) - 1) if len(c) > 253 else np.nan

        for n in (20, 50, 200):
            ma = c.rolling(n).mean()
            f[f"above_ma{n}"] = last / float(ma.iloc[-1]) - 1 if not np.isnan(ma.iloc[-1]) else np.nan
        ma50 = c.rolling(50).mean()
        f["ma50_slope_21"] = (float(ma50.iloc[-1]) / float(ma50.iloc[-22]) - 1) if len(ma50.dropna()) > 22 else np.nan

        f["rsi14"] = _rsi(c)
        ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
        macd = ema12 - ema26
        f["macd_hist_norm"] = float((macd - macd.ewm(span=9).mean()).iloc[-1]) / last

        h, l, rc = highs[t].reindex(c.index), lows[t].reindex(c.index), c.shift()
        tr = pd.concat([h - l, (h - rc).abs(), (l - rc).abs()], axis=1).max(axis=1)
        atr20 = tr.rolling(20).mean()
        atr100 = tr.rolling(100).mean()
        f["atr_pct"] = float(atr20.iloc[-1]) / last if not np.isnan(atr20.iloc[-1]) else np.nan
        f["vol_contraction"] = (float(atr20.iloc[-1] / atr100.iloc[-1])
                                if len(atr100.dropna()) else np.nan)  # <1 = contracting

        f["realized_vol_21"] = float(r.tail(21).std() * np.sqrt(252))
        f["realized_vol_63"] = float(r.tail(63).std() * np.sqrt(252))

        v = vols[t].dropna()
        if len(v) > 63:
            f["rel_volume_5_63"] = float(v.tail(5).mean() / v.tail(63).mean())
            dv = (v * c.reindex(v.index)).dropna()
            f["dollar_vol_med_60"] = float(dv.tail(60).median())
            f["amihud_63"] = float((r.reindex(dv.index).abs() / dv).tail(63).mean() * 1e9)
        hi52 = float(c.tail(252).max())
        lo52 = float(c.tail(252).min())
        f["dist_52w_high"] = last / hi52 - 1
        f["dist_52w_low"] = last / lo52 - 1
        f["drawdown"] = last / float(c.tail(252).cummax().iloc[-1]) - 1

        o = opens[t].reindex(c.index)
        gaps = (o / c.shift() - 1).abs()
        f["gap_freq_63"] = float((gaps.tail(63) > 0.02).mean())
        f["gap_max_63"] = float(gaps.tail(63).max())

        if bench_r is not None and t != benchmark:
            f["rs_spy_21"] = f["ret_21"] - (float(bench.iloc[-1] / bench.iloc[-22] - 1)
                                            if len(bench.dropna()) > 22 else np.nan)
            f["rs_spy_63"] = f["ret_63"] - (float(bench.iloc[-1] / bench.iloc[-64] - 1)
                                            if len(bench.dropna()) > 64 else np.nan)
            join = pd.concat([r, bench_r], axis=1).dropna().tail(252)
            if len(join) > 100:
                cov = np.cov(join.iloc[:, 0], join.iloc[:, 1])
                f["beta_252"] = float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else np.nan
        etf = sector_etf.get(sectors.get(t, ""), None)
        if etf and etf in closes.columns and t != etf:
            e = closes[etf].dropna()
            if len(e) > 64:
                f["rs_sector_63"] = f["ret_63"] - float(e.iloc[-1] / e.iloc[-64] - 1)

        rows.append(f)
    return pd.DataFrame(rows)
