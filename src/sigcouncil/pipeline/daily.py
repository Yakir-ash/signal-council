"""The daily pipeline (DESIGN.md §9): data → quality → features → regime →
predictions → scores → risk → watchlist → ledger → evaluation → report pack.

Failure philosophy: degraded inputs degrade LOUDLY (warnings land on the
dashboard); broken price data aborts scoring entirely (garbage-in guard).
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pandas as pd

from ..config import thresholds_cfg, universe_cfg
from ..features import events, fundamental, technical
from ..ledger import evaluate as eval_mod
from ..ledger import ledger as ledger_mod
from ..logutil import RunLog
from ..models import composite
from ..models.probabilities import TABLES_PATH
from ..paths import REPORTS
from ..quality import checks
from ..regime import classifier as regime_mod
from ..scoring import assemble, divergence
from ..store import panels, universe
from ..watchlist import state as watchlist_mod
from . import fetch


def run_daily(skip_fetch: bool = False, tickers: list[str] | None = None) -> dict:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    rl = RunLog(run_id)
    as_of = pd.Timestamp.utcnow().tz_localize(None).normalize()
    ucfg = universe_cfg()

    # 1-2 ---- data + quality
    if skip_fetch:
        data = {"prices": panels.read_panel(panels.PRICES),
                "fundamentals": panels.read_panel(panels.FUNDAMENTALS),
                "insiders": panels.read_panel(panels.INSIDERS),
                "macro": panels.read_panel(panels.MACRO), "news_counts": {}}
        rl.step("fetch", "ok", "skipped (using existing panels)")
    else:
        data = fetch.fetch_all(rl, tickers=tickers)

    q = checks.assess(data["prices"], data["fundamentals"], data["news_counts"])
    rl.step("quality", "ok" if q.run_ok else "fail",
            f"{len(q.per_ticker)} names, {len(q.quarantined)} quarantined; {q.reasons}")
    if data["prices"].empty:
        rl.step("abort", "fail", "no price data — scoring skipped (garbage-in guard)")
        return _write_pack(None, None, None, None, None, q, rl, as_of, aborted=True)

    px = panels.primary_prices(data["prices"])

    # 3 ---- features
    sectors = universe.sector_map()
    setf = universe.sector_etf_map()
    tech = technical.compute(px, as_of, benchmark=ucfg["benchmark"],
                             sectors=sectors, sector_etf=setf)
    rl.step("features_technical", "ok", f"{len(tech)} tickers")
    fund = fundamental.compute(data["fundamentals"], as_of) \
        if not data["fundamentals"].empty else pd.DataFrame(columns=["ticker"])
    rl.step("features_fundamental", "ok" if len(fund) else "warn", f"{len(fund)} tickers")
    ins = events.insider_features(data["insiders"], as_of)
    feat = tech.merge(fund, on="ticker", how="left") if not fund.empty else tech
    feat = feat.merge(ins, on="ticker", how="left") if not ins.empty else feat
    ev = events.earnings_event_features(fund, px, as_of, benchmark=ucfg["benchmark"]) \
        if not fund.empty else pd.DataFrame(columns=["ticker"])
    feat = feat.merge(ev, on="ticker", how="left") if not ev.empty else feat
    val = events.valuation_features(fund, tech, px, as_of) if not fund.empty else pd.DataFrame(columns=["ticker"])
    feat = feat.merge(val, on="ticker", how="left") if not val.empty else feat
    feat["sector"] = feat["ticker"].map(sectors)

    # regime-only symbols are inputs, not candidates
    scorable = feat[~feat["ticker"].isin(universe.regime_only())].reset_index(drop=True)

    # 4 ---- regime
    reg = regime_mod.classify(px, data["macro"], as_of,
                              universe.scorable_stocks(), benchmark=ucfg["benchmark"])
    rl.step("regime", "ok", f"{reg.label} (score {reg.score})")

    # 5-7 ---- components, divergence, scores, risk
    comps = composite.compute_components(scorable, reg.score)
    sec_ret = _sector_returns_63(px, setf)
    with open(REPORTS / "latest_sectors.json", "w") as sf:
        json.dump(sec_ret, sf, indent=1)
    div = divergence.compute(scorable, sec_ret, sectors)
    calibrated = TABLES_PATH.exists()
    scored = assemble.assemble(scorable, comps, div, q.per_ticker, reg.to_dict(),
                               ucfg, q.quarantined, calibrated)
    rl.step("scoring", "ok", f"{len(scored)} scored; calibrated={calibrated}")

    top = assemble.select_top(scored)
    avoid = assemble.select_avoid(scored)
    rl.step("selection", "ok", f"{len(top)} high-conviction, {len(avoid)} avoid")

    # 8 ---- watchlist
    wl_changes = watchlist_mod.update(scored, str(as_of.date()))
    rl.step("watchlist", "ok", f"{len(wl_changes)} state changes")

    # 9 ---- ledger: record for top + avoid + a scan sample (top50) for calibration breadth
    n_led = 0
    for _, row in scored.head(50).iterrows():
        kind = ("high_conviction" if row["ticker"] in set(top["ticker"])
                else "avoid" if row["ticker"] in set(avoid["ticker"]) else "daily_scan")
        ledger_mod.record_predictions(row.to_dict(), reg.to_dict(), run_id, kind)
        n_led += 1
    rl.step("ledger", "ok", f"{n_led} tickers x horizons recorded")

    # 10 ---- evaluate matured
    n_eval = eval_mod.evaluate_matured(px, benchmark=ucfg["benchmark"])
    calib = eval_mod.calibration_report()
    rl.step("evaluate", "ok", f"{n_eval} newly matured; ledger n={calib.get('n', 0)}")

    # 11 ---- report pack
    return _write_pack(scored, top, avoid, reg, calib, q, rl, as_of,
                       wl_changes=wl_changes)


def _sector_returns_63(px: pd.DataFrame, sector_etf: dict[str, str]) -> dict[str, float]:
    closes = px.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last").sort_index()
    out = {}
    for sector, etf in sector_etf.items():
        if etf in closes.columns:
            c = closes[etf].dropna()
            if len(c) > 64:
                out[sector] = float(c.iloc[-1] / c.iloc[-64] - 1)
    return out


def _row_public(row: pd.Series) -> dict:
    d = {k: row[k] for k in ("ticker", "opportunity", "risk", "divergence",
                             "data_confidence", "confidence", "components",
                             "risk_components", "predictions", "reasons",
                             "divergence_patterns", "thesis_breakers",
                             "px_last", "px_last_date")}
    d["gate_fail"] = row.get("gate_fail", "")
    return d


def _write_pack(scored, top, avoid, reg, calib, q, rl, as_of, aborted=False,
                wl_changes=None) -> dict:
    day = str(as_of.date())
    outdir = REPORTS / day
    outdir.mkdir(parents=True, exist_ok=True)
    pack = {
        "date": day,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "aborted": aborted,
        "regime": reg.to_dict() if reg is not None else None,
        "no_high_conviction": (top is None or len(top) == 0),
        "top_opportunities": [_row_public(r) for _, r in top.iterrows()] if top is not None else [],
        "avoid_list": [_row_public(r) for _, r in avoid.iterrows()] if avoid is not None else [],
        "scan_summary": ([_row_public(r) for _, r in scored.head(25).iterrows()]
                         if scored is not None else []),
        "watchlist_changes": (wl_changes.to_dict("records")
                              if wl_changes is not None and len(wl_changes) else []),
        "calibration": calib,
        "quality": {"run_ok": q.run_ok, "reasons": q.reasons,
                    "quarantined": q.quarantined,
                    "median_data_confidence": (float(q.per_ticker["data_confidence"].median())
                                               if len(q.per_ticker) else None)},
        "runlog": rl.summary(),
        "tags_note": "opportunity/risk/probabilities are MODEL ESTIMATES; prices/fundamentals "
                     "are FACTS from tagged sources; catalyst text (if present) is AI INTERPRETATION.",
    }
    with open(outdir / "pack.json", "w") as f:
        json.dump(pack, f, indent=1, default=str)
    with open(REPORTS / "latest.json", "w") as f:
        json.dump(pack, f, indent=1, default=str)
    rl.step("report_pack", "ok", str(outdir / "pack.json"))
    return pack
