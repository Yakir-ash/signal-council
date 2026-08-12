"""Scoring integrity: gates have no fallback, probabilities are clamped,
components stay in range, empty selection is a legal outcome."""
import numpy as np
import pandas as pd
import pytest

from sigcouncil.models import composite, probabilities
from sigcouncil.scoring import assemble, divergence, risk


def _feat(n=30, seed=1):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "ticker": [f"T{i:03d}" for i in range(n)],
        "px_last": rng.uniform(10, 300, n),
        "px_last_date": "2026-08-11",
        "mom_12_1": rng.normal(0.05, 0.2, n),
        "rs_spy_63": rng.normal(0, 0.08, n),
        "rs_sector_63": rng.normal(0, 0.06, n),
        "above_ma200": rng.normal(0.02, 0.1, n),
        "above_ma50": rng.normal(0.01, 0.06, n),
        "ma50_slope_21": rng.normal(0, 0.02, n),
        "rsi14": rng.uniform(25, 75, n),
        "macd_hist_norm": rng.normal(0, 0.01, n),
        "vol_contraction": rng.uniform(0.6, 1.4, n),
        "realized_vol_63": rng.uniform(0.15, 0.6, n),
        "amihud_63": rng.uniform(0.001, 0.1, n),
        "dollar_vol_med_60": rng.uniform(5e6, 5e9, n),
        "dist_52w_high": rng.uniform(-0.5, 0, n),
        "dist_52w_low": rng.uniform(0, 0.5, n),
        "drawdown": rng.uniform(-0.4, 0, n),
        "gap_freq_63": rng.uniform(0, 0.2, n),
        "gap_max_63": rng.uniform(0, 0.1, n),
        "beta_252": rng.uniform(0.5, 2, n),
        "rev_yoy": rng.normal(0.1, 0.15, n),
        "rev_accel": rng.normal(0, 0.05, n),
        "gross_margin_delta": rng.normal(0, 0.02, n),
        "op_margin_delta": rng.normal(0, 0.02, n),
        "gross_margin": rng.uniform(0.2, 0.7, n),
        "fcf_margin": rng.normal(0.1, 0.1, n),
        "gp_over_assets": rng.uniform(0, 0.8, n),
        "accruals": rng.normal(0, 0.05, n),
        "roe": rng.normal(0.15, 0.2, n),
        "share_change_yoy": rng.normal(0, 0.03, n),
        "sector": rng.choice(["Information Technology", "Financials", "Health Care"], n),
    })


UCFG = {"eligibility": {"min_price": 5.0, "min_median_dollar_volume": 1e7,
                        "min_history_days": 252, "max_missing_recent_days": 3}}
REGIME = {"label": "sideways", "score": 0.0}


def _scored(feat):
    comps = composite.compute_components(feat, 0.0)
    div = divergence.compute(feat, {}, {})
    qual = pd.DataFrame({"ticker": feat["ticker"], "data_confidence": 85.0})
    return assemble.assemble(feat, comps, div, qual, REGIME, UCFG, [], calibrated=False)


def test_scores_in_range_and_decomposed():
    s = _scored(_feat())
    assert s["opportunity"].between(0, 100).all()
    assert s["risk"].between(0, 100).all()
    assert all(isinstance(c, dict) and len(c) >= 8 for c in s["components"])


def test_cold_start_probabilities_clamped():
    s = _scored(_feat())
    for preds in s["predictions"]:
        for p in preds:
            assert 0.35 <= p["p_beat_benchmark"] <= 0.68, "cold-start clamp violated"
            assert p["basis"] == "cold_start_clamped"
            assert p["downside_p5"] < p["exp_return_low"] < p["exp_return_high"]


def test_confidence_capped_cold_start():
    s = _scored(_feat())
    assert (s["confidence"] <= 0.68).all()


def test_no_fallback_when_nothing_qualifies():
    s = _scored(_feat())
    s["opportunity"] = 40.0          # nothing passes the 72 threshold
    top = assemble.select_top(s)
    assert len(top) == 0, "select_top must return empty, never a fallback"


def test_gates_block_illiquid_names():
    f = _feat()
    f.loc[0, "dollar_vol_med_60"] = 1e5
    f.loc[1, "px_last"] = 2.0
    s = _scored(f)
    gf = s.set_index("ticker")["gate_fail"]
    assert "illiquid" in gf["T000"]
    assert "price<" in gf["T001"]


def test_quarantine_blocks_selection():
    f = _feat()
    comps = composite.compute_components(f, 0.0)
    div = divergence.compute(f, {}, {})
    qual = pd.DataFrame({"ticker": f["ticker"], "data_confidence": 85.0})
    s = assemble.assemble(f, comps, div, qual, REGIME, UCFG, ["T000"], calibrated=False)
    assert "data_quarantine" in s.set_index("ticker")["gate_fail"]["T000"]


def test_divergence_patterns_are_named_and_reported():
    f = _feat()
    f.loc[0, ["rev_accel", "gross_margin_delta", "rs_spy_63"]] = [0.05, 0.02, -0.10]
    div = divergence.compute(f, {}, {})
    row = div[div["ticker"] == "T000"].iloc[0]
    names = {p["pattern"] for p in row["divergence_patterns"]}
    assert "fundamentals_up_price_lagging" in names
    assert row["divergence"] > 30


def test_thesis_breakers_generated():
    s = _scored(_feat())
    assert all(len(tb) >= 1 for tb in s["thesis_breakers"])
