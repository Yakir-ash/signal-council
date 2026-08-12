"""Rule-composite component scores (v1 backbone model).

Each component maps economically-motivated features to a 0-100 cross-sectional
score. Missing data never silently helps a stock: components with insufficient
inputs return 50 (neutral) and the coverage ratio feeds Data Confidence.

Output per ticker: component scores + `component_evidence` explaining exactly
which features drove each score (the transparency requirement, DESIGN.md §7).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def xsec_pct(s: pd.Series) -> pd.Series:
    """Cross-sectional percentile rank, 0-100; NaN stays NaN."""
    return s.rank(pct=True, na_option="keep") * 100


def _blend(parts: list[pd.Series], weights: list[float] | None = None) -> tuple[pd.Series, pd.Series]:
    """Weighted mean of available percentile parts; returns (score, coverage)."""
    df = pd.concat(parts, axis=1)
    w = np.array(weights if weights else [1.0] * df.shape[1])
    vals = df.values
    mask = ~np.isnan(vals)
    wsum = (mask * w).sum(axis=1)
    score = np.where(wsum > 0, np.nansum(vals * w, axis=1) / np.where(wsum == 0, 1, wsum), 50.0)
    coverage = mask.mean(axis=1)
    return (pd.Series(score, index=df.index).fillna(50.0),
            pd.Series(coverage, index=df.index))


def compute_components(feat: pd.DataFrame, regime_score: float,
                       sector_col: str = "sector") -> pd.DataFrame:
    """feat: one row per ticker, merged technical+fundamental+event+valuation features."""
    f = feat.set_index("ticker")
    out = pd.DataFrame(index=f.index)
    evidence: dict[str, dict] = {t: {} for t in f.index}

    def get(col: str) -> pd.Series:
        return f[col] if col in f.columns else pd.Series(np.nan, index=f.index)

    # ---------------- fundamental momentum (PEAD + acceleration)
    parts = {
        "rev_accel": xsec_pct(get("rev_accel")),
        "rev_yoy": xsec_pct(get("rev_yoy")),
        "eps_yoy": xsec_pct(get("eps_yoy")),
        "margin_trend": xsec_pct(get("gross_margin_delta").fillna(0) + get("op_margin_delta").fillna(0)),
        "post_report_drift": xsec_pct(get("post_report_drift")),
        "report_reaction": xsec_pct(get("report_reaction")),
    }
    out["fundamental_momentum"], cov_fm = _blend(list(parts.values()), [1.2, 0.8, 0.8, 1.0, 1.2, 0.8])
    _note(evidence, "fundamental_momentum", parts)

    # ---------------- price momentum
    parts = {
        "mom_12_1": xsec_pct(get("mom_12_1")),
        "rs_spy_63": xsec_pct(get("rs_spy_63")),
        "rs_sector_63": xsec_pct(get("rs_sector_63")),
        "above_ma200": xsec_pct(get("above_ma200")),
    }
    out["price_momentum"], cov_pm = _blend(list(parts.values()), [1.2, 1.0, 1.0, 0.8])
    _note(evidence, "price_momentum", parts)

    # ---------------- quality
    parts = {
        "gp_over_assets": xsec_pct(get("gp_over_assets")),
        "fcf_margin": xsec_pct(get("fcf_margin")),
        "low_accruals": xsec_pct(-get("accruals")),
        "buybacks": xsec_pct(-get("share_change_yoy")),
        "roe": xsec_pct(get("roe").clip(-1, 1)),
    }
    out["quality"], cov_q = _blend(list(parts.values()), [1.2, 1.0, 1.0, 0.6, 0.8])
    _note(evidence, "quality", parts)

    # ---------------- valuation (sector-relative + own history; never raw cheap)
    ev_sector_z = pd.Series(np.nan, index=f.index)
    if "ev_sales" in f.columns and sector_col in f.columns:
        grp = f.groupby(sector_col)["ev_sales"]
        ev_sector_z = (f["ev_sales"] - grp.transform("median")) / grp.transform("std").replace(0, np.nan)
    parts = {
        "fcf_yield": xsec_pct(get("fcf_yield")),
        "ev_sales_vs_sector": xsec_pct(-ev_sector_z),
        "vs_own_3y_range": xsec_pct(-get("ev_sales_pctile_3y_est")),
    }
    out["valuation"], cov_v = _blend(list(parts.values()), [1.2, 1.0, 0.8])
    _note(evidence, "valuation", parts)

    # ---------------- insider (sparse: direct mapping, not percentile)
    net = get("insider_net_value_90").fillna(0)
    cluster = get("insider_cluster").fillna(0)
    buyers = get("insider_buyers_90").fillna(0)
    ins = 50 + np.clip(np.log10(np.abs(net.values) + 1) * np.sign(net.values) * 4, -25, 25) \
        + cluster.values * 15 + np.clip(buyers.values - 1, 0, 3) * 3
    out["insider"] = pd.Series(np.clip(ins, 0, 100), index=f.index)
    for t in f.index:
        if net.get(t, 0) != 0:
            evidence[t]["insider"] = {"net_value_90d": float(net.get(t, 0)),
                                      "cluster": bool(cluster.get(t, 0)),
                                      "distinct_buyers": int(buyers.get(t, 0))}

    # ---------------- technical confirmation (timing overlay)
    rsi = get("rsi14")
    rsi_sweet = 100 - (rsi - 57.5).abs() * 4          # peak at RSI 57.5, falls to 0 at extremes
    parts = {
        "trend_confirm": xsec_pct(get("above_ma50")),
        "macd": xsec_pct(get("macd_hist_norm")),
        "rsi_zone": rsi_sweet.clip(0, 100),
        "vol_contraction": xsec_pct(-get("vol_contraction")),
        "near_high_base": xsec_pct(get("dist_52w_high").clip(-0.35, 0)),
    }
    out["technical"], cov_t = _blend(list(parts.values()), [1.0, 0.8, 0.6, 0.8, 0.8])
    _note(evidence, "technical", parts)

    # ---------------- regime fit
    beta = get("beta_252")
    rv = get("realized_vol_63")
    if regime_score >= 0.15:      # risk-on: higher-beta momentum is rewarded
        fit = 50 + 20 * regime_score * ((xsec_pct(beta) - 50) / 50).fillna(0)
    else:                          # risk-off: defensiveness is rewarded
        fit = 50 + 20 * (-min(regime_score, 0)) * ((50 - xsec_pct(rv)) / 50).fillna(0)
    out["regime_fit"] = fit.clip(0, 100)

    # ---------------- catalyst: neutral until the Tier-2 LLM overlay fills it
    out["catalyst"] = 50.0

    out["component_coverage"] = pd.concat([cov_fm, cov_pm, cov_q, cov_v, cov_t], axis=1).mean(axis=1)
    out["evidence"] = pd.Series(evidence)
    return out.reset_index().rename(columns={"index": "ticker"})


def _note(evidence: dict, comp: str, parts: dict[str, pd.Series]) -> None:
    for t in evidence:
        vals = {k: round(float(v.get(t)), 1) for k, v in parts.items()
                if v.get(t) is not None and not np.isnan(v.get(t, np.nan))}
        if vals:
            evidence[t][comp] = vals
