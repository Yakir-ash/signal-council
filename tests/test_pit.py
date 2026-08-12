"""Adversarial point-in-time tests (red-team Attack 4): plant leaks, require
the build to fail if they get through."""
import numpy as np
import pandas as pd

from sigcouncil.features import fundamental
from sigcouncil.store.panels import pit_filter


def _facts(rows):
    return pd.DataFrame(rows, columns=["ticker", "concept", "tag", "unit", "period_start",
                                       "period_end", "value", "form", "filed",
                                       "fiscal_frame", "source"])


def test_pit_filter_blocks_future_filings():
    df = _facts([
        ("X", "revenue", "Revenues", "USD", "2024-01-01", "2024-03-31", 100, "10-Q", "2024-05-01", None, "t"),
        ("X", "revenue", "Revenues", "USD", "2024-04-01", "2024-06-30", 999, "10-Q", "2024-08-01", None, "t"),
    ])
    out = pit_filter(df, "2024-06-15", observed_col="filed")
    assert len(out) == 1
    assert out.iloc[0]["value"] == 100, "future-filed fact leaked past the PIT gate"


def test_pit_filter_blocks_same_quarter_early_knowledge():
    # THE trap: a Q2 fact (period ends 2024-06-30) filed 2024-08-01 must NOT be
    # visible on 2024-07-01 even though the period has ended.
    df = _facts([
        ("X", "revenue", "Revenues", "USD", "2024-04-01", "2024-06-30", 999, "10-Q", "2024-08-01", None, "t"),
    ])
    out = pit_filter(df, "2024-07-01", observed_col="filed")
    assert len(out) == 0, "period-ended-but-not-yet-filed fact leaked"


def test_fundamental_features_respect_filing_dates():
    rows = []
    # 8 clean quarters filed ~40d after period end, then a FUTURE quarter with huge growth
    for i, (pe, val) in enumerate([("2022-03-31", 100), ("2022-06-30", 102), ("2022-09-30", 104),
                                   ("2022-12-31", 106), ("2023-03-31", 108), ("2023-06-30", 110),
                                   ("2023-09-30", 112), ("2023-12-31", 114)]):
        ps = str(pd.Timestamp(pe) - pd.Timedelta(days=89))[:10]
        filed = str(pd.Timestamp(pe) + pd.Timedelta(days=40))[:10]
        rows.append(("X", "revenue", "Revenues", "USD", ps, pe, val, "10-Q", filed, None, "t"))
    rows.append(("X", "revenue", "Revenues", "USD", "2024-01-01", "2024-03-31", 500,
                 "10-Q", "2024-05-10", None, "t"))
    facts = _facts(rows)
    feats_before = fundamental.compute_one(facts, pd.Timestamp("2024-04-15"))
    feats_after = fundamental.compute_one(facts, pd.Timestamp("2024-06-01"))
    # before the 500-quarter is filed, growth must be the boring ~5%
    assert feats_before["rev_yoy"] < 0.10, "unfiled monster quarter leaked into rev_yoy"
    assert feats_after["rev_yoy"] > 0.5


def test_pit_filter_drops_unparseable_dates():
    df = _facts([("X", "revenue", "Revenues", "USD", "2024-01-01", "2024-03-31",
                  100, "10-Q", "not-a-date", None, "t")])
    out = pit_filter(df, "2024-06-15", observed_col="filed")
    assert len(out) == 0, "row with unparseable observed_at must be excluded, not included"
