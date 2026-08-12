"""Watchlist state machine (DESIGN.md §15). States change only through these
rules, every transition is recorded with a reason, and history is kept."""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from ..config import thresholds_cfg
from ..paths import DATA

STATES = ["BUY_CANDIDATE", "WATCH", "WAIT_FOR_PULLBACK", "WAIT_FOR_CONFIRMATION",
          "OVERVALUED", "THESIS_DETERIORATING", "AVOID", "NONE"]

PATH = DATA / "watchlist.json"
HISTORY = DATA / "watchlist_history.jsonl"


def _load() -> dict:
    if PATH.exists():
        return json.loads(PATH.read_text())
    return {}


def classify(row: pd.Series, was: str, th: dict) -> tuple[str, str]:
    opp, risk, conf = row["opportunity"], row["risk"], row["confidence"]
    val = row["components"].get("valuation", 50)
    tech = row["components"].get("technical", 50)
    fund = row["components"].get("fundamental_momentum", 50)
    hc = th["high_conviction"]

    if row.get("gate_fail"):
        return "NONE", f"gates: {row['gate_fail']}"
    if opp >= hc["min_opportunity"] and conf >= hc["min_confidence"] and risk <= hc["max_risk"]:
        return "BUY_CANDIDATE", f"opp {opp:.0f} conf {conf:.0%} risk {risk:.0f}"
    if opp <= th["avoid_list"]["max_opportunity"] and risk >= th["avoid_list"]["min_risk"]:
        return "AVOID", f"opp {opp:.0f} with risk {risk:.0f}"
    if was in ("BUY_CANDIDATE", "WATCH", "WAIT_FOR_PULLBACK", "WAIT_FOR_CONFIRMATION") \
            and fund <= 35:
        return "THESIS_DETERIORATING", f"fundamental momentum fell to {fund:.0f}"
    if opp >= th["watchlist"]["enter_watch"]:
        if val <= 25:
            return "OVERVALUED", f"strong name, valuation component {val:.0f}"
        if tech <= 40:
            return "WAIT_FOR_CONFIRMATION", f"thesis ok, technicals not confirming ({tech:.0f})"
        if row["components"].get("price_momentum", 50) >= 75 and tech >= 70 and opp < hc["min_opportunity"]:
            return "WAIT_FOR_PULLBACK", "extended after strong run"
        return "WATCH", f"opp {opp:.0f}"
    if was != "NONE" and opp > th["watchlist"]["drop_below"]:
        return was, "unchanged"
    return "NONE", f"opp {opp:.0f} below watch threshold"


def update(scored: pd.DataFrame, as_of: str) -> pd.DataFrame:
    th = thresholds_cfg()
    state = _load()
    changes = []
    new_state: dict = {}
    for _, row in scored.iterrows():
        t = row["ticker"]
        was = state.get(t, {}).get("state", "NONE")
        now, reason = classify(row, was, th)
        if now != "NONE":
            new_state[t] = {"state": now, "since": state.get(t, {}).get("since", as_of)
                            if now == was else as_of,
                            "opportunity": row["opportunity"], "risk": row["risk"],
                            "updated": as_of}
        if now != was:
            changes.append({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "date": as_of, "ticker": t, "from": was, "to": now,
                            "reason": reason})
    PATH.write_text(json.dumps(new_state, indent=1, default=str))
    if changes:
        with open(HISTORY, "a") as f:
            for c in changes:
                f.write(json.dumps(c) + "\n")
    return pd.DataFrame(changes)
