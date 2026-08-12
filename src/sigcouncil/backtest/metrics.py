"""Performance & calibration metrics for backtests (DESIGN.md §8)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def summarize(returns: pd.Series, benchmark: pd.Series | None = None,
              freq: int = 252) -> dict:
    r = returns.dropna()
    if r.empty:
        return {"n": 0}
    eq = (1 + r).cumprod()
    years = len(r) / freq
    cagr = float(eq.iloc[-1] ** (1 / years) - 1) if years > 0 else np.nan
    vol = float(r.std() * np.sqrt(freq))
    dn = r[r < 0]
    downside = float(dn.std() * np.sqrt(freq)) if len(dn) else np.nan
    dd = float((eq / eq.cummax() - 1).min())
    out = {
        "n_periods": int(len(r)),
        "total_return": float(eq.iloc[-1] - 1),
        "cagr": cagr,
        "vol": vol,
        "sharpe": float(cagr / vol) if vol > 0 else np.nan,
        "sortino": float(cagr / downside) if downside and downside > 0 else np.nan,
        "max_drawdown": dd,
        "win_rate": float((r > 0).mean()),
        "avg_gain": float(r[r > 0].mean()) if (r > 0).any() else np.nan,
        "avg_loss": float(r[r < 0].mean()) if (r < 0).any() else np.nan,
    }
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    out["profit_factor"] = float(gains / losses) if losses > 0 else np.nan
    if benchmark is not None:
        b = benchmark.reindex(r.index).dropna()
        both = pd.concat([r, b], axis=1).dropna()
        if len(both) > 10:
            excess = both.iloc[:, 0] - both.iloc[:, 1]
            te = float(excess.std() * np.sqrt(freq))
            beq = (1 + both.iloc[:, 1]).cumprod()
            bcagr = float(beq.iloc[-1] ** (freq / len(both)) - 1)
            out["benchmark_cagr"] = bcagr
            out["excess_cagr"] = cagr - bcagr
            out["information_ratio"] = float(excess.mean() * freq / te) if te > 0 else np.nan
    return {k: (round(v, 4) if isinstance(v, float) and not np.isnan(v) else v)
            for k, v in out.items()}


def bucket_forward_returns(scores: pd.DataFrame, fwd: pd.DataFrame,
                           n_buckets: int = 5) -> pd.DataFrame:
    """scores: date,ticker,score; fwd: date,ticker,fwd_ret → mean fwd return per
    score bucket per date, then aggregated. The monotonicity of this table is the
    single most important sanity check on the whole scoring engine."""
    df = scores.merge(fwd, on=["date", "ticker"]).dropna()
    if df.empty:
        return pd.DataFrame()
    df["bucket"] = df.groupby("date")["score"].transform(
        lambda s: pd.qcut(s.rank(method="first"), n_buckets, labels=False, duplicates="drop"))
    agg = df.groupby("bucket").agg(
        n=("fwd_ret", "size"), mean_fwd=("fwd_ret", "mean"),
        median_fwd=("fwd_ret", "median"), hit=("fwd_ret", lambda x: (x > 0).mean())
    ).reset_index()
    return agg.round(5)
