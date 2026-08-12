"""Dashboard renderer: gather committed artifacts → embed as JSON → one
self-contained dashboard.html (works offline, persists as a Cowork artifact)."""
from __future__ import annotations

import json
from pathlib import Path

from ..paths import DATA, LEDGER, REPORTS
from ..ledger import ledger as ledger_mod

TEMPLATE = Path(__file__).parent / "template.html"


def _latest_backtest() -> dict | None:
    btdir = REPORTS / "backtests"
    if not btdir.exists():
        return None
    files = sorted(btdir.glob("bt_*.json"))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def _analyzer_packs(limit: int = 8) -> list[dict]:
    adir = REPORTS / "analyzer"
    if not adir.exists():
        return []
    out = []
    for p in sorted(adir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True)[:limit]:
        with open(p) as f:
            out.append(json.load(f))
    return out


def _sector_returns() -> dict:
    p = REPORTS / "latest_sectors.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def render_latest(out_path: Path | None = None) -> str:
    latest = REPORTS / "latest.json"
    pack = json.loads(latest.read_text()) if latest.exists() else None

    preds = ledger_mod.load_predictions()
    outs = ledger_mod.load_outcomes()
    wl_path = DATA / "watchlist.json"
    wl_hist_path = DATA / "watchlist_history.jsonl"
    wl_hist = []
    if wl_hist_path.exists():
        with open(wl_hist_path) as f:
            wl_hist = [json.loads(line) for line in f if line.strip()]

    data = {
        "pack": pack,
        "watchlist": json.loads(wl_path.read_text()) if wl_path.exists() else {},
        "watchlist_history": wl_hist[-200:],
        "ledger_total": int(len(preds)),
        "ledger_preds": (preds.sort_values("ts").tail(600).to_dict("records")
                         if not preds.empty else []),
        "ledger_outcomes": outs.to_dict("records") if not outs.empty else [],
        "backtest": _latest_backtest(),
        "analyzer": _analyzer_packs(),
        "sector_returns": _sector_returns(),
    }
    html = TEMPLATE.read_text().replace(
        "__DATA__", json.dumps(data, default=str).replace("</", "<\\/"))
    out = out_path or (REPORTS / "dashboard.html")
    out.write_text(html)
    if pack:
        day_dir = REPORTS / pack["date"]
        if day_dir.exists():
            (day_dir / "dashboard.html").write_text(html)
    return str(out)
