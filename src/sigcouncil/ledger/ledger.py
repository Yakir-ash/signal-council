"""The immutable Prediction Ledger (DESIGN.md §18 / red-team Attack 5).

- Append-only JSONL, one file per month, committed to git.
- Every entry: full context (price, scores, probabilities, reasoning, regime,
  model/weights versions) so accountability never depends on memory.
- scripts/check_ledger_immutable.py runs in CI and FAILS THE BUILD if any commit
  modifies or deletes an existing ledger line. History cannot be rewritten quietly.
- Outcomes are written to SEPARATE files; predictions are never edited.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .. import MODEL_VERSION
from ..config import weights_cfg
from ..paths import LEDGER


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _pred_file(ts: str) -> Path:
    return LEDGER / f"predictions-{ts[:7]}.jsonl"


def _outcome_file(ts: str) -> Path:
    return LEDGER / f"outcomes-{ts[:7]}.jsonl"


def prediction_id(ts: str, ticker: str, horizon: str) -> str:
    return hashlib.sha1(f"{ts}|{ticker}|{horizon}".encode()).hexdigest()[:16]


def record_predictions(scored_row: dict, regime: dict, run_id: str,
                       kind: str = "daily_scan") -> list[dict]:
    """Write one ledger entry per horizon for a scored ticker. Returns entries."""
    ts = _now()
    entries = []
    for p in scored_row["predictions"]:
        e = {
            "id": prediction_id(ts, scored_row["ticker"], p["horizon"]),
            "ts": ts,
            "run_id": run_id,
            "kind": kind,                      # daily_scan | high_conviction | avoid | on_demand
            "ticker": scored_row["ticker"],
            "price": scored_row.get("px_last"),
            "price_date": scored_row.get("px_last_date"),
            "price_source": "prices_panel(yfinance/stooq)",
            "model_version": MODEL_VERSION,
            "weights_version": weights_cfg()["version"],
            "opportunity": scored_row["opportunity"],
            "risk": scored_row["risk"],
            "divergence": scored_row["divergence"],
            "data_confidence": scored_row["data_confidence"],
            "confidence": scored_row["confidence"],
            "horizon": p["horizon"],
            "horizon_days": p["horizon_days"],
            "p_positive": p["p_positive"],
            "p_beat_benchmark": p["p_beat_benchmark"],
            "exp_return_low": p["exp_return_low"],
            "exp_return_high": p["exp_return_high"],
            "downside_p5": p["downside_p5"],
            "probability_basis": p["basis"],
            "regime": regime["label"],
            "regime_score": regime["score"],
            "components": scored_row["components"],
            "reasons": scored_row["reasons"],
            "thesis_breakers": scored_row["thesis_breakers"],
        }
        entries.append(e)
    path = _pred_file(ts)
    with open(path, "a") as f:
        for e in entries:
            f.write(json.dumps(e, default=_json_safe) + "\n")
    return entries


def _json_safe(o):
    try:
        import numpy as np
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
    except ImportError:
        pass
    if isinstance(o, float) and (o != o):  # NaN
        return None
    return str(o)


def load_predictions() -> pd.DataFrame:
    rows = []
    for p in sorted(LEDGER.glob("predictions-*.jsonl")):
        with open(p) as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return pd.DataFrame(rows)


def load_outcomes() -> pd.DataFrame:
    rows = []
    for p in sorted(LEDGER.glob("outcomes-*.jsonl")):
        with open(p) as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return pd.DataFrame(rows)


def record_outcomes(outcomes: list[dict]) -> None:
    ts = _now()
    path = _outcome_file(ts)
    existing = set()
    df = load_outcomes()
    if not df.empty:
        existing = set(df["prediction_id"])
    with open(path, "a") as f:
        for o in outcomes:
            if o["prediction_id"] in existing:
                continue                       # an outcome is also written once, ever
            o["evaluated_at"] = ts
            f.write(json.dumps(o, default=_json_safe) + "\n")
