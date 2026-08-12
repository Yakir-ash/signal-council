"""Data acquisition stage. Runs in GitHub Actions (Tier 1) where the network is open."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from ..logutil import RunLog, get_logger
from ..providers import build
from ..store import panels, universe

log = get_logger("fetch")

MACRO_SERIES = ["DGS10", "DGS2", "T10Y2Y", "BAMLH0A0HYM2", "T10YIE", "VIXCLS"]


def fetch_all(runlog: RunLog, years: int = 4, insider_days: int = 120,
              tickers: list[str] | None = None) -> dict:
    prov = build()
    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=int(years * 365.25) + 10)
    syms = tickers or universe.all_symbols()

    # ---- prices: primary + independent cross-check source
    try:
        px = prov.prices.daily_ohlcv(syms, start, end)
        runlog.step("fetch_prices_primary", "ok" if len(px) else "fail",
                    f"{len(px)} rows / {px['ticker'].nunique() if len(px) else 0} tickers")
    except Exception as e:  # noqa: BLE001
        px = pd.DataFrame()
        runlog.step("fetch_prices_primary", "fail", str(e)[:300])

    px2 = pd.DataFrame()
    if prov.prices_fallback is not None:
        try:
            # cross-validation sample: full universe recent window is enough
            check_start = end - timedelta(days=60)
            px2 = prov.prices_fallback.daily_ohlcv(syms, check_start, end)
            runlog.step("fetch_prices_fallback", "ok" if len(px2) else "warn",
                        f"{len(px2)} rows")
        except Exception as e:  # noqa: BLE001
            runlog.step("fetch_prices_fallback", "warn", str(e)[:300])

    merged = pd.concat([p for p in (px, px2) if not p.empty], ignore_index=True) \
        if (not px.empty or not px2.empty) else pd.DataFrame()
    if not merged.empty:
        panels.upsert_prices(merged)

    # ---- fundamentals (EDGAR XBRL)
    stocks = tickers or universe.current_stocks()
    facts_frames, failed = [], 0
    for i, t in enumerate(stocks):
        f = prov.fundamentals.company_facts(t)
        if f.empty:
            failed += 1
        else:
            facts_frames.append(f)
    facts = pd.concat(facts_frames, ignore_index=True) if facts_frames else pd.DataFrame()
    if not facts.empty:
        panels.write_panel(facts, panels.FUNDAMENTALS)
    runlog.step("fetch_fundamentals", "ok" if facts_frames else "fail",
                f"{len(facts_frames)} tickers ok, {failed} failed")

    # ---- insiders (Form 4)
    ins_rows = []
    since = date.today() - timedelta(days=insider_days)
    for t in stocks:
        for tx in prov.insiders.insider_transactions(t, since):
            ins_rows.append(vars(tx))
    insiders = pd.DataFrame(ins_rows)
    if not insiders.empty:
        panels.write_panel(insiders, panels.INSIDERS)
    runlog.step("fetch_insiders", "ok", f"{len(ins_rows)} transactions")

    # ---- macro
    macro_frames = []
    for s in MACRO_SERIES:
        m = prov.macro.series(s, start)
        if not m.empty:
            macro_frames.append(m)
    macro = pd.concat(macro_frames, ignore_index=True) if macro_frames else pd.DataFrame()
    if not macro.empty:
        panels.write_panel(macro, panels.MACRO)
    runlog.step("fetch_macro", "ok" if macro_frames else "warn",
                f"{len(macro_frames)}/{len(MACRO_SERIES)} series")

    # ---- news counts (attention signal + quality-gate corroboration)
    news_counts: dict[str, int] = {}
    for np_ in prov.news:
        for t in stocks:
            try:
                items = np_.recent_news(t, limit=20)
                news_counts[t] = news_counts.get(t, 0) + len(items)
            except Exception:  # noqa: BLE001
                pass
    runlog.step("fetch_news", "ok", f"{sum(news_counts.values())} items")

    return {"prices": panels.read_panel(panels.PRICES), "fundamentals": facts,
            "insiders": insiders, "macro": macro, "news_counts": news_counts}
