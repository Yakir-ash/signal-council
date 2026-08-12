"""On-demand deep analysis for a single ticker ("Analyze NVDA") — §16.

Produces a machine-readable analysis pack the Tier-2 Claude session (or CLI user)
renders. Runs the SAME code path as the daily scan — no separate, drift-prone
logic — plus filings retrieval for the qualitative layer.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pandas as pd

from ..config import universe_cfg
from ..features import events, fundamental, technical
from ..logutil import RunLog
from ..models import composite
from ..models.probabilities import TABLES_PATH
from ..paths import REPORTS
from ..providers import build
from ..quality import checks
from ..regime import classifier as regime_mod
from ..scoring import assemble, divergence
from ..store import panels, universe
from . import fetch
from .daily import _sector_returns_63


def analyze(ticker: str, refresh: bool = True) -> dict:
    ticker = ticker.upper()
    run_id = "ondemand-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    rl = RunLog(run_id)
    as_of = pd.Timestamp.utcnow().tz_localize(None).normalize()
    ucfg = universe_cfg()

    need = sorted({ticker, ucfg["benchmark"], *universe.etfs()})
    if refresh:
        # refresh this ticker + market context; reuse panels for the cross-section
        fetch.fetch_all(rl, tickers=need, years=4)

    prices = panels.read_panel(panels.PRICES)
    facts = panels.read_panel(panels.FUNDAMENTALS)
    insiders = panels.read_panel(panels.INSIDERS)
    macro = panels.read_panel(panels.MACRO)
    if prices.empty or ticker not in set(prices["ticker"]):
        return {"ticker": ticker, "error": "no price data available", "run_id": run_id}
    px = panels.primary_prices(prices)

    sectors = universe.sector_map()
    setf = universe.sector_etf_map()

    # cross-section = full scorable universe so percentiles mean something
    tech = technical.compute(px, as_of, benchmark=ucfg["benchmark"],
                             sectors=sectors, sector_etf=setf)
    fund = fundamental.compute(facts, as_of) if not facts.empty else pd.DataFrame(columns=["ticker"])
    ins = events.insider_features(insiders, as_of)
    feat = tech.merge(fund, on="ticker", how="left") if not fund.empty else tech
    feat = feat.merge(ins, on="ticker", how="left") if not ins.empty else feat
    ev = events.earnings_event_features(fund, px, as_of, benchmark=ucfg["benchmark"]) \
        if not fund.empty else pd.DataFrame(columns=["ticker"])
    feat = feat.merge(ev, on="ticker", how="left") if not ev.empty else feat
    val = events.valuation_features(fund, tech, px, as_of) if not fund.empty else pd.DataFrame(columns=["ticker"])
    feat = feat.merge(val, on="ticker", how="left") if not val.empty else feat
    feat["sector"] = feat["ticker"].map(sectors)
    scorable = feat[~feat["ticker"].isin(universe.regime_only())].reset_index(drop=True)

    reg = regime_mod.classify(px, macro, as_of, universe.scorable_stocks(),
                              benchmark=ucfg["benchmark"])
    q = checks.assess(prices[prices["ticker"].isin(set(scorable["ticker"]))], facts, {})
    comps = composite.compute_components(scorable, reg.score)
    div = divergence.compute(scorable, _sector_returns_63(px, setf), sectors)
    scored = assemble.assemble(scorable, comps, div, q.per_ticker, reg.to_dict(),
                               ucfg, q.quarantined, TABLES_PATH.exists())

    row = scored[scored["ticker"] == ticker]
    if row.empty:
        return {"ticker": ticker, "error": "ticker not scorable (not in universe or gated)",
                "run_id": run_id}
    r = row.iloc[0]

    # rank context
    rank = int((scored["opportunity"] > r["opportunity"]).sum()) + 1

    # filings for the qualitative layer
    prov = build()
    filings = prov.filings.recent_filings(ticker, forms=["10-K", "10-Q", "8-K", "4"], limit=15)
    filing_list = [{"form": f.form, "filed_at": f.filed_at, "url": f.url,
                    "period": f.period} for f in filings]
    # cache latest 10-K/10-Q text for LLM tier
    for f in filings:
        if f.form in ("10-K", "10-Q"):
            try:
                prov.filings.filing_text(f)
            except Exception:  # noqa: BLE001
                pass
            break

    raw = scorable[scorable["ticker"] == ticker].iloc[0]
    pack = {
        "kind": "on_demand_analysis",
        "ticker": ticker,
        "run_id": run_id,
        "as_of": str(as_of.date()),
        "sector": sectors.get(ticker),
        "regime": reg.to_dict(),
        "rank_in_universe": f"{rank}/{len(scored)}",
        "scores": {"opportunity": r["opportunity"], "risk": r["risk"],
                   "divergence": r["divergence"], "data_confidence": r["data_confidence"],
                   "confidence": r["confidence"]},
        "components": r["components"],
        "risk_components": r["risk_components"],
        "predictions": r["predictions"],
        "reasons": r["reasons"],
        "divergence_patterns": r["divergence_patterns"],
        "thesis_breakers": r["thesis_breakers"],
        "gate_fail": r["gate_fail"],
        "price": {"last": r["px_last"], "date": r["px_last_date"],
                  "source": "prices_panel(yfinance/stooq) [FACT]"},
        "features_raw": {k: (None if pd.isna(v) else round(float(v), 5))
                         for k, v in raw.items()
                         if isinstance(v, (int, float)) and k != "px_last"},
        "recent_filings": filing_list,
        "evidence": r["evidence"],
        "tags_note": "features_raw derived from FACT sources (SEC XBRL, prices); "
                     "scores/predictions are MODEL ESTIMATES; no AI INTERPRETATION included yet.",
    }
    outdir = REPORTS / "analyzer"
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / f"{ticker}_{str(as_of.date())}.json", "w") as f:
        json.dump(pack, f, indent=1, default=str)
    return pack
