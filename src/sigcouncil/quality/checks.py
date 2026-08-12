"""Data-quality gate → per-ticker Data Confidence Score (0-100) + run-level verdict.

A sophisticated model on bad data is worse than useless (DESIGN.md §23): every
check below can BLOCK scoring for a name (quarantine) or the whole run.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import thresholds_cfg
from ..logutil import get_logger

log = get_logger("quality")


@dataclass
class QualityReport:
    run_ok: bool
    reasons: list[str]
    per_ticker: pd.DataFrame          # ticker, data_confidence, flags
    quarantined: list[str] = field(default_factory=list)


def assess(prices_all_sources: pd.DataFrame, fundamentals: pd.DataFrame,
           news_counts: dict[str, int] | None = None,
           filings_recent: dict[str, str] | None = None) -> QualityReport:
    cfg = thresholds_cfg()["quality_gate"]
    reasons: list[str] = []
    news_counts = news_counts or {}
    filings_recent = filings_recent or {}

    if prices_all_sources.empty:
        return QualityReport(False, ["price panel empty — run aborted"], pd.DataFrame(), [])

    px = prices_all_sources.copy()
    px["date"] = pd.to_datetime(px["date"])
    last_date = px["date"].max()
    bdays_stale = int(np.busday_count(last_date.date(), pd.Timestamp.utcnow().date()))
    if bdays_stale > cfg["max_stale_days_prices"]:
        reasons.append(f"prices stale: last bar {last_date.date()} ({bdays_stale} bdays old)")

    rows = []
    quarantined: list[str] = []
    recent_cut = last_date - pd.Timedelta(days=45)
    recent = px[px["date"] >= recent_cut]

    for t, g in recent.groupby("ticker"):
        score = 100.0
        flags: list[str] = []

        # --- freshness per name
        t_last = g["date"].max()
        t_stale = int(np.busday_count(t_last.date(), last_date.date()))
        if t_stale > 0:
            score -= min(30, 10 * t_stale)
            flags.append(f"stale:{t_stale}d")

        # --- cross-source agreement (yfinance vs stooq raw close)
        srcs = g.pivot_table(index="date", columns="source", values="close", aggfunc="last")
        if {"yfinance", "stooq"}.issubset(srcs.columns):
            both = srcs.dropna()
            if len(both) >= 5:
                rel = (both["yfinance"] - both["stooq"]).abs() / both["stooq"]
                med = float(rel.median())
                if med > cfg["max_price_cross_source_diff"]:
                    score -= 35
                    flags.append(f"xsource_diff:{med:.3%}")
        else:
            score -= 10                       # only one source available
            flags.append("single_source")

        # --- suspicious jumps: big move with no corroborating filing/news
        gg = g[g["source"] == g["source"].iloc[0]].sort_values("date")
        rets = gg["close"].pct_change().abs()
        big = rets[rets > cfg["outlier_move_no_news"]]
        if len(big) > 0:
            has_event = news_counts.get(t, 0) > 0 or t in filings_recent
            if not has_event:
                score -= 40
                flags.append(f"unexplained_move:{float(big.max()):.0%}")
                quarantined.append(t)

        # --- zero/negative prices, zero-volume streaks
        if (gg["close"] <= 0).any():
            score = 0
            flags.append("nonpositive_price")
            quarantined.append(t)
        if (gg["volume"].fillna(0) == 0).mean() > 0.3:
            score -= 15
            flags.append("thin_volume_data")

        # --- fundamentals coverage
        if not fundamentals.empty:
            f = fundamentals[fundamentals["ticker"] == t]
            core = {"revenue", "net_income", "operating_cf", "assets"}
            have = core & set(f["concept"].unique())
            missing = core - have
            if missing:
                score -= 5 * len(missing)
                flags.append("fund_missing:" + ",".join(sorted(missing)))
            elif not f.empty:
                newest = pd.to_datetime(f["filed"]).max()
                if (last_date - newest).days > 120:
                    score -= 10
                    flags.append("fund_stale")
        else:
            score -= 20
            flags.append("no_fundamentals_panel")

        rows.append({"ticker": t, "data_confidence": max(0.0, round(score, 1)),
                     "flags": ";".join(flags)})

    per = pd.DataFrame(rows)
    run_ok = len(reasons) == 0
    if not run_ok:
        log.warning("run-level quality issues: %s", reasons)
    return QualityReport(run_ok, reasons, per, sorted(set(quarantined)))
