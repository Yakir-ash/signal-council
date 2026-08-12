"""Walk-forward backtest engine (DESIGN.md §8, red-team Attacks 1/2/4).

Integrity properties:
- Point-in-time universe: at each rebalance date, candidates = ACTUAL S&P 500
  membership on that date (historical constituents file). Names without price
  data (typically delisted) are COUNTED and reported as a bias bound.
- Features at date T computed only from bars/facts observable at T (pit_filter;
  fundamentals join on `filed`).
- Costs: configurable bps per side, applied to turnover.
- Outputs: metrics vs SPY + equal-weight + random-portfolio draws, score-bucket
  monotonicity, and the empirical probability tables used by live predictions.
- Fundamentals caveat: EDGAR XBRL history is thin before ~2010 and absent for
  many delisted names; runs report fundamental-coverage per rebalance so
  fundamental-light early periods are visible, not hidden.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..features import technical, fundamental, events
from ..logutil import get_logger
from ..models import composite
from ..paths import CALIBRATION, REPORTS
from ..regime import classifier as regime_mod
from ..scoring import divergence
from ..store import panels, universe
from . import metrics

log = get_logger("backtest")


@dataclass
class BacktestConfig:
    start: str = "2016-01-01"
    end: str | None = None
    rebalance_days: int = 21            # monthly
    top_n: int = 20
    cost_bps_per_side: float = 10.0
    horizon_days: tuple = (5, 21, 63)
    random_draws: int = 100
    use_fundamentals: bool = True
    seed: int = 7                        # fixed: backtests must be reproducible


@dataclass
class BacktestResult:
    config: dict
    portfolio: dict
    benchmark: dict
    equal_weight: dict
    random_beat_frac: float
    bucket_table: list
    missing_data_names: dict            # rebalance date -> count of members w/o data
    fundamental_coverage: dict
    by_regime: dict
    prob_tables: dict = field(default_factory=dict)


def run(cfg: BacktestConfig) -> BacktestResult:
    px_all = panels.primary_prices()
    if px_all.empty:
        raise RuntimeError("price panel empty — run fetch first")
    px_all = px_all[px_all["date"] >= pd.Timestamp(cfg.start) - pd.Timedelta(days=420)]
    end = pd.Timestamp(cfg.end) if cfg.end else px_all["date"].max()

    closes = px_all.pivot_table(index="date", columns="ticker", values="adj_close",
                                aggfunc="last").sort_index()
    dates = closes.index
    rebal_dates = dates[(dates >= pd.Timestamp(cfg.start)) & (dates <= end)][::cfg.rebalance_days]
    facts = panels.read_panel(panels.FUNDAMENTALS) if cfg.use_fundamentals else pd.DataFrame()
    macro = panels.read_panel(panels.MACRO)
    sectors = universe.sector_map()
    setf = universe.sector_etf_map()

    port_rets, ew_rets, spy_rets = [], [], []
    rand_rets = {i: [] for i in range(cfg.random_draws)}
    missing: dict[str, int] = {}
    fund_cov: dict[str, float] = {}
    by_regime_rows = []
    score_rows, fwd_rows = [], []
    rng = np.random.default_rng(cfg.seed)
    prev_hold: set[str] = set()

    for i, t0 in enumerate(rebal_dates[:-1]):
        t1 = rebal_dates[i + 1]
        members = universe.members_as_of(t0.date())
        have = [m for m in members if m in closes.columns
                and closes[m].loc[:t0].notna().sum() > 120]
        missing[str(t0.date())] = len(members) - len(have)

        # ---- features strictly as of t0
        px_slice = px_all[(px_all["ticker"].isin(have + [c for c in setf.values()] + ["SPY"]))
                          & (px_all["date"] <= t0)]
        tech = technical.compute(px_slice, t0, benchmark="SPY",
                                 sectors=sectors, sector_etf=setf)
        feat = tech[tech["ticker"].isin(have)].reset_index(drop=True)
        if not facts.empty:
            fnd = fundamental.compute(facts[facts["ticker"].isin(have)], t0)
            fund_cov[str(t0.date())] = round(len(fnd) / max(1, len(have)), 3)
            if not fnd.empty:
                feat = feat.merge(fnd, on="ticker", how="left")
                ev = events.earnings_event_features(fnd, px_slice, t0)
                if not ev.empty:
                    feat = feat.merge(ev, on="ticker", how="left")
                val = events.valuation_features(fnd, tech, px_slice, t0)
                if not val.empty:
                    feat = feat.merge(val, on="ticker", how="left")
        feat["sector"] = feat["ticker"].map(sectors)

        reg = regime_mod.classify(px_slice, macro, t0, have)
        comps = composite.compute_components(feat, reg.score)
        weights = {"fundamental_momentum": 20, "price_momentum": 18, "quality": 14,
                   "divergence": 14, "valuation": 10, "catalyst": 8, "insider": 6,
                   "technical": 6, "regime_fit": 4}
        div = divergence.compute(feat, {}, sectors)
        m = comps.merge(div[["ticker", "divergence"]], on="ticker", how="left")
        m["divergence"] = m["divergence"].fillna(30)
        total_w = sum(weights.values())
        m["score"] = sum(m.get(k, pd.Series(50, index=m.index)).fillna(50) * w
                         for k, w in weights.items()) / total_w

        # ---- forward returns t0->t1 (entry next bar open ≈ next close conservative)
        window = closes.loc[t0:t1]
        if len(window) < 3:
            continue
        entry = window.iloc[1]          # NEXT bar after signal: no same-bar lookahead
        exit_ = window.iloc[-1]
        fwd = (exit_ / entry - 1)

        ranked = m.sort_values("score", ascending=False)
        hold = list(ranked["ticker"].head(cfg.top_n))
        hold_rets = fwd.reindex(hold).dropna()
        turnover = len(set(hold) - prev_hold) / max(1, len(hold))
        cost = 2 * cfg.cost_bps_per_side / 1e4 * turnover
        port_rets.append({"date": t1, "ret": float(hold_rets.mean()) - cost,
                          "regime": reg.label})
        prev_hold = set(hold)

        univ_rets = fwd.reindex(have).dropna()
        ew_rets.append({"date": t1, "ret": float(univ_rets.mean())})
        if "SPY" in closes.columns:
            spy_win = closes["SPY"].loc[t0:t1]
            spy_rets.append({"date": t1, "ret": float(spy_win.iloc[-1] / spy_win.iloc[1] - 1)})
        for d in range(cfg.random_draws):
            pick = rng.choice(univ_rets.index, size=min(cfg.top_n, len(univ_rets)), replace=False)
            rand_rets[d].append(float(univ_rets.reindex(pick).mean()))

        for _, r in m.iterrows():
            score_rows.append({"date": t0, "ticker": r["ticker"], "score": float(r["score"])})
        # multi-horizon forward returns for probability tables
        for h in cfg.horizon_days:
            idx = dates.searchsorted(t0) + 1
            if idx + h < len(dates):
                e0, e1 = dates[idx], dates[idx + h]
                fh = closes.loc[e1] / closes.loc[e0] - 1
                spy_h = float(fh.get("SPY", np.nan))
                for tkr in have:
                    v = fh.get(tkr, np.nan)
                    if not np.isnan(v):
                        fwd_rows.append({"date": t0, "ticker": tkr, "h": h,
                                         "fwd_ret": float(v), "spy": spy_h,
                                         "regime": reg.label})
        by_regime_rows.append({"date": t1, "regime": reg.label,
                               "ret": port_rets[-1]["ret"],
                               "ew": ew_rets[-1]["ret"]})
        log.info("rebalance %s: %d members, %d held, regime %s",
                 t0.date(), len(have), len(hold), reg.label)

    pr = pd.DataFrame(port_rets).set_index("date")["ret"]
    ew = pd.DataFrame(ew_rets).set_index("date")["ret"]
    spy = pd.DataFrame(spy_rets).set_index("date")["ret"] if spy_rets else None
    periods_per_year = 252 / cfg.rebalance_days

    rand_total = [float(np.prod([1 + x for x in v]) - 1) for v in rand_rets.values() if v]
    port_total = float(np.prod(1 + pr) - 1)
    beat_frac = float(np.mean([port_total > rt for rt in rand_total])) if rand_total else np.nan

    scores_df = pd.DataFrame(score_rows)
    fwd_df = pd.DataFrame(fwd_rows)
    bucket = metrics.bucket_forward_returns(
        scores_df, fwd_df[fwd_df["h"] == 21][["date", "ticker", "fwd_ret"]], 5) \
        if not fwd_df.empty else pd.DataFrame()

    breg = {}
    brdf = pd.DataFrame(by_regime_rows)
    if not brdf.empty:
        for rg, g in brdf.groupby("regime"):
            breg[rg] = {"n": int(len(g)), "mean_ret": round(float(g["ret"].mean()), 4),
                        "mean_excess_vs_ew": round(float((g["ret"] - g["ew"]).mean()), 4)}

    prob_tables = build_prob_tables(scores_df, fwd_df) if not fwd_df.empty else {}

    result = BacktestResult(
        config=vars(cfg) | {"horizon_days": list(cfg.horizon_days)},
        portfolio=metrics.summarize(pr, spy, freq=int(periods_per_year)),
        benchmark=metrics.summarize(spy, freq=int(periods_per_year)) if spy is not None else {},
        equal_weight=metrics.summarize(ew, spy, freq=int(periods_per_year)),
        random_beat_frac=beat_frac,
        bucket_table=bucket.to_dict("records") if not bucket.empty else [],
        missing_data_names=missing,
        fundamental_coverage=fund_cov,
        by_regime=breg,
        prob_tables=prob_tables,
    )
    outdir = REPORTS / "backtests"
    outdir.mkdir(parents=True, exist_ok=True)
    stamp = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%S")
    with open(outdir / f"bt_{stamp}.json", "w") as f:
        json.dump(result.__dict__, f, indent=1, default=str)
    return result


H_NAME = {5: "1w", 21: "1m", 63: "3m", 126: "6-12m", 189: "6-12m"}


def build_prob_tables(scores_df: pd.DataFrame, fwd_df: pd.DataFrame,
                      write: bool = True) -> dict:
    """Empirical conditional distributions: score-decile × regime × horizon →
    quantiles + hit rates. These become the live system's probability source."""
    df = scores_df.merge(fwd_df, on=["date", "ticker"])
    if df.empty:
        return {}
    df["decile"] = "d" + (df["score"] // 10).clip(0, 9).astype(int).astype(str)
    df["hname"] = df["h"].map(H_NAME)
    cells = {}
    for keys, g in df.groupby(["decile", "regime", "hname"]):
        if len(g) < 30:
            continue
        cells["|".join(keys)] = _cell(g)
    for keys, g in df.groupby(["decile", "hname"]):        # regime-agnostic fallback
        cells[f"{keys[0]}|*|{keys[1]}"] = _cell(g)
    tables = {"version": pd.Timestamp.utcnow().strftime("bt%Y%m%d"),
              "n_total": int(len(df)), "cells": cells,
              "note": "built by walk-forward backtest; INDICATIVE until live ledger confirms"}
    if write:
        CALIBRATION.mkdir(parents=True, exist_ok=True)
        with open(CALIBRATION / "prob_tables.json", "w") as f:
            json.dump(tables, f, indent=1)
    return tables


def _cell(g: pd.DataFrame) -> dict:
    return {
        "n": int(len(g)),
        "p_positive": round(float((g["fwd_ret"] > 0).mean()), 4),
        "p_beat": round(float((g["fwd_ret"] > g["spy"]).mean()), 4),
        "q05": round(float(g["fwd_ret"].quantile(0.05)), 4),
        "q25": round(float(g["fwd_ret"].quantile(0.25)), 4),
        "q50": round(float(g["fwd_ret"].quantile(0.50)), 4),
        "q75": round(float(g["fwd_ret"].quantile(0.75)), 4),
    }
