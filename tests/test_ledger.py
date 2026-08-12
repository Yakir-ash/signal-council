"""Ledger behavior: append-only, one outcome per prediction, full context recorded."""
import json

import pandas as pd
import pytest

import sigcouncil.paths as paths_mod


@pytest.fixture()
def tmp_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(paths_mod, "LEDGER", tmp_path)
    import sigcouncil.ledger.ledger as L
    monkeypatch.setattr(L, "LEDGER", tmp_path)
    return L


ROW = {
    "ticker": "TEST", "px_last": 100.0, "px_last_date": "2026-08-11",
    "opportunity": 75.0, "risk": 40.0, "divergence": 55.0,
    "data_confidence": 90.0, "confidence": 0.6,
    "components": {"price_momentum": 80},
    "predictions": [
        {"horizon": "1m", "horizon_days": 21, "p_positive": 0.6, "p_beat_benchmark": 0.58,
         "exp_return_low": -0.05, "exp_return_high": 0.09, "downside_p5": -0.12,
         "expected_vol": 0.08, "basis": "cold_start_clamped"}],
    "reasons": ["r1"], "thesis_breakers": ["b1"],
}
REGIME = {"label": "sideways", "score": 0.0}


def test_record_and_load_roundtrip(tmp_ledger):
    entries = tmp_ledger.record_predictions(ROW, REGIME, "run1", "daily_scan")
    assert len(entries) == 1
    df = tmp_ledger.load_predictions()
    assert len(df) == 1
    e = df.iloc[0]
    assert e["ticker"] == "TEST" and e["price"] == 100.0
    assert e["model_version"] and e["weights_version"]
    assert e["probability_basis"] == "cold_start_clamped"


def test_appends_never_overwrite(tmp_ledger):
    tmp_ledger.record_predictions(ROW, REGIME, "run1")
    tmp_ledger.record_predictions(ROW, REGIME, "run2")
    df = tmp_ledger.load_predictions()
    assert len(df) == 2, "second record must append, not replace"


def test_outcome_written_once(tmp_ledger):
    tmp_ledger.record_predictions(ROW, REGIME, "run1")
    pid = tmp_ledger.load_predictions().iloc[0]["id"]
    o = {"prediction_id": pid, "realized_return": 0.05, "outcome_positive": True}
    tmp_ledger.record_outcomes([dict(o)])
    tmp_ledger.record_outcomes([dict(o, realized_return=0.99)])  # attempted rewrite
    outs = tmp_ledger.load_outcomes()
    assert len(outs) == 1
    assert outs.iloc[0]["realized_return"] == 0.05, "outcome rewrite must be ignored"
