"""Divergence / Mispricing Score (DESIGN.md §7) — the system's specialty.

Each pattern is an explicit, named rule describing a way the market may be
mispricing available evidence. The score is a capped sum of triggered pattern
points, and every trigger is reported by name with its inputs — no black box.

Note: `estimates_before_price` (analyst revisions leading price) requires paid
estimate-revision data and is intentionally ABSENT in v1 rather than faked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _g(row: pd.Series, col: str, default: float = np.nan) -> float:
    v = row.get(col, default)
    try:
        return float(v) if v is not None and not pd.isna(v) else default
    except (TypeError, ValueError):
        return default


def score_row(row: pd.Series, sector_ret_63: float | None) -> tuple[float, list[dict]]:
    triggered: list[dict] = []

    rev_accel = _g(row, "rev_accel")
    gm_d = _g(row, "gross_margin_delta")
    om_d = _g(row, "op_margin_delta")
    rs_spy = _g(row, "rs_spy_63")
    rs_sec = _g(row, "rs_sector_63")
    reaction = _g(row, "report_reaction")
    drift = _g(row, "post_report_drift")
    days_since = _g(row, "days_since_report")
    cluster = _g(row, "insider_cluster", 0)
    dist_hi = _g(row, "dist_52w_high")
    val_pct = _g(row, "ev_sales_pctile_3y_est")

    # 1) fundamentals improving while price lags the market
    if rev_accel > 0.02 and (gm_d > 0 or om_d > 0) and rs_spy < -0.02:
        triggered.append({"pattern": "fundamentals_up_price_lagging", "points": 25,
                          "inputs": {"rev_accel": rev_accel, "gm_delta": gm_d, "rs_spy_63": rs_spy}})

    # 2) strong report met with indiscriminate selling (fundamentals say otherwise)
    if reaction < -0.05 and days_since <= 45 and rev_accel > 0 and (gm_d >= 0 or om_d >= 0):
        triggered.append({"pattern": "strong_report_sold_off", "points": 20,
                          "inputs": {"report_reaction": reaction, "rev_accel": rev_accel}})

    # 3) classic PEAD continuation window
    if reaction > 0.04 and 2 <= days_since <= 40 and drift > -0.02:
        triggered.append({"pattern": "post_earnings_drift_window", "points": 15,
                          "inputs": {"report_reaction": reaction, "days_since_report": days_since}})

    # 4) insider cluster buying into pessimism
    if cluster >= 1 and dist_hi < -0.20:
        triggered.append({"pattern": "insider_cluster_into_drawdown", "points": 20,
                          "inputs": {"dist_52w_high": dist_hi,
                                     "buyers": _g(row, "insider_buyers_90", 0)}})

    # 5) margins expanding while valuation sits low in its own range
    if (gm_d > 0.01 or om_d > 0.01) and val_pct < 0.35:
        triggered.append({"pattern": "margin_expansion_compressed_valuation", "points": 20,
                          "inputs": {"om_delta": om_d, "gm_delta": gm_d,
                                     "ev_sales_pctile_3y": val_pct}})

    # 6) strong relative strength despite a weak sector
    if sector_ret_63 is not None and sector_ret_63 < -0.03 and rs_sec > 0.05:
        triggered.append({"pattern": "rs_strength_weak_sector", "points": 15,
                          "inputs": {"rs_sector_63": rs_sec, "sector_ret_63": sector_ret_63}})

    pts = min(100.0, sum(t["points"] for t in triggered))
    # base 30 when nothing triggers: absence of divergence isn't 0-worthy, it's unremarkable
    return (30.0 + 0.7 * pts if triggered else 30.0), triggered


def compute(feat: pd.DataFrame, sector_returns_63: dict[str, float],
            sectors: dict[str, str]) -> pd.DataFrame:
    rows = []
    for _, r in feat.iterrows():
        t = r["ticker"]
        sec_ret = sector_returns_63.get(sectors.get(t, ""), None)
        s, trig = score_row(r, sec_ret)
        rows.append({"ticker": t, "divergence": round(s, 1), "divergence_patterns": trig})
    return pd.DataFrame(rows)
