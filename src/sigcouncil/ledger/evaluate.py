"""Evaluate matured predictions and measure calibration (DESIGN.md §19).

Outcome criteria are fixed at prediction time:
- outcome_positive: total return over the horizon > 0
- outcome_beat:     total return > benchmark total return over same window
Calibration: Brier scores + reliability bins for p_positive and p_beat.
Accuracy stats always come with N; the dashboard refuses to celebrate N<30.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..logutil import get_logger
from . import ledger

log = get_logger("evaluate")


def evaluate_matured(prices: pd.DataFrame, benchmark: str = "SPY") -> int:
    preds = ledger.load_predictions()
    if preds.empty:
        return 0
    done = set(ledger.load_outcomes()["prediction_id"]) if not ledger.load_outcomes().empty else set()

    closes = prices.pivot_table(index="date", columns="ticker", values="adj_close",
                                aggfunc="last").sort_index()
    bench = closes.get(benchmark)
    new_outcomes: list[dict] = []

    for _, p in preds.iterrows():
        if p["id"] in done or p["ticker"] not in closes.columns:
            continue
        start = pd.Timestamp(p["ts"][:10])
        c = closes[p["ticker"]].dropna()
        c_after = c[c.index >= start]
        if len(c_after) < p["horizon_days"] + 1:
            continue                                   # not matured yet
        entry = float(c_after.iloc[0])
        exit_ = float(c_after.iloc[int(p["horizon_days"])])
        realized = exit_ / entry - 1
        b_ret = np.nan
        if bench is not None:
            b = bench.dropna()
            b_after = b[b.index >= start]
            if len(b_after) > p["horizon_days"]:
                b_ret = float(b_after.iloc[int(p["horizon_days"])]) / float(b_after.iloc[0]) - 1
        new_outcomes.append({
            "prediction_id": p["id"],
            "ticker": p["ticker"],
            "horizon": p["horizon"],
            "matured_at": str(c_after.index[int(p["horizon_days"])].date()),
            "realized_return": round(realized, 5),
            "benchmark_return": None if np.isnan(b_ret) else round(b_ret, 5),
            "outcome_positive": bool(realized > 0),
            "outcome_beat": None if np.isnan(b_ret) else bool(realized > b_ret),
            "in_predicted_range": bool(p["exp_return_low"] <= realized <= p["exp_return_high"]),
            "below_downside_p5": bool(realized < p["downside_p5"]),
        })
    if new_outcomes:
        ledger.record_outcomes(new_outcomes)
        log.info("evaluated %d matured predictions", len(new_outcomes))
    return len(new_outcomes)


def calibration_report() -> dict:
    preds, outs = ledger.load_predictions(), ledger.load_outcomes()
    if preds.empty or outs.empty:
        return {"n": 0, "note": "no matured predictions yet"}
    df = preds.merge(outs, left_on="id", right_on="prediction_id",
                     suffixes=("", "_o"))
    if df.empty:
        return {"n": 0, "note": "no matured predictions yet"}

    rep: dict = {"n": int(len(df))}
    for prob_col, out_col in (("p_positive", "outcome_positive"),
                              ("p_beat_benchmark", "outcome_beat")):
        d = df.dropna(subset=[prob_col, out_col])
        d = d[d[out_col].notna()]
        if d.empty:
            continue
        y = d[out_col].astype(float)
        p = d[prob_col].astype(float)
        rep[f"brier_{prob_col}"] = round(float(((p - y) ** 2).mean()), 4)
        bins = pd.cut(p, bins=[0, .4, .5, .6, .7, 1.0])
        rel = d.groupby(bins, observed=True).agg(
            n=(out_col, "size"), predicted=(prob_col, "mean"),
            realized=(out_col, "mean")).reset_index()
        rel["bin"] = rel[prob_col].astype(str) if prob_col in rel else rel.iloc[:, 0].astype(str)
        rep[f"reliability_{prob_col}"] = [
            {"bin": str(r.iloc[0]), "n": int(r["n"]),
             "predicted": round(float(r["predicted"]), 3),
             "realized": round(float(r["realized"]), 3)}
            for _, r in rel.iterrows()]

    # windows
    df["ts_date"] = pd.to_datetime(df["ts"].str[:10])
    now = pd.Timestamp.utcnow().tz_localize(None)
    for label, days in (("30d", 30), ("90d", 90), ("365d", 365), ("all", 10_000)):
        w = df[df["ts_date"] >= now - pd.Timedelta(days=days)]
        wb = w.dropna(subset=["outcome_beat"])
        rep[f"window_{label}"] = {
            "n": int(len(w)),
            "hit_rate_positive": round(float(w["outcome_positive"].mean()), 3) if len(w) else None,
            "hit_rate_beat": round(float(wb["outcome_beat"].astype(float).mean()), 3) if len(wb) else None,
            "avg_realized": round(float(w["realized_return"].mean()), 4) if len(w) else None,
            "avg_excess": round(float((w["realized_return"] - w["benchmark_return"]).mean()), 4)
                          if len(w.dropna(subset=["benchmark_return"])) else None,
        }
    return rep
