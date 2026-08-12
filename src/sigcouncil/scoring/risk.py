"""Risk Score 0-100 (higher = riskier) with full decomposition + Thesis Breakers.

Risk is not the inverse of opportunity — it is the cost of being wrong:
volatility, illiquidity, leverage, event proximity, valuation asymmetry,
gap/drawdown behavior. Every recommendation ships with this decomposition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.composite import xsec_pct


def compute(feat: pd.DataFrame) -> pd.DataFrame:
    f = feat.set_index("ticker")

    def get(col: str) -> pd.Series:
        return f[col] if col in f.columns else pd.Series(np.nan, index=f.index)

    comp = pd.DataFrame(index=f.index)
    comp["volatility"] = xsec_pct(get("realized_vol_63")).fillna(50)
    comp["illiquidity"] = xsec_pct(get("amihud_63")).fillna(50)

    lev = pd.Series(0.0, index=f.index)
    nd_op = get("net_debt_to_op")
    cov = get("interest_coverage")
    lev += np.clip((nd_op.fillna(0) - 2) * 12, 0, 60)
    lev += np.where(cov.notna() & (cov < 3), 25, 0)
    lev += np.where(cov.notna() & (cov < 1.5), 15, 0)
    comp["leverage"] = lev.clip(0, 100)

    d2r = get("days_to_report_est")
    comp["event_proximity"] = np.where(d2r.notna() & (d2r <= 7), 80,
                                np.where(d2r.notna() & (d2r <= 15), 55, 25))

    vp = get("ev_sales_pctile_3y_est")
    comp["valuation_extreme"] = np.where(vp.notna() & (vp > 0.9), 85,
                                  np.where(vp.notna() & (vp > 0.75), 60, 35))

    comp["tail_behavior"] = (0.5 * xsec_pct(-get("drawdown")).fillna(50)
                             + 0.5 * xsec_pct(get("gap_max_63")).fillna(50))
    comp["market_sensitivity"] = xsec_pct(get("beta_252").abs()).fillna(50)

    w = {"volatility": 0.25, "illiquidity": 0.13, "leverage": 0.20,
         "event_proximity": 0.12, "valuation_extreme": 0.12,
         "tail_behavior": 0.10, "market_sensitivity": 0.08}
    risk = sum(comp[k] * v for k, v in w.items())

    out = pd.DataFrame({"ticker": f.index, "risk": risk.round(1).values})
    out["risk_components"] = [
        {k: round(float(comp.loc[t, k]), 1) for k in w} for t in f.index]
    return out


def thesis_breakers(row: pd.Series, components: dict, divergence_patterns: list[dict]) -> list[str]:
    """Concrete falsifiers generated from what the thesis actually rests on.
    A triggered breaker forces a visible watchlist state change (§12)."""
    br: list[str] = []
    pats = {p["pattern"] for p in (divergence_patterns or [])}

    def g(col, default=np.nan):
        v = row.get(col, default)
        return float(v) if v is not None and not pd.isna(v) else default

    if components.get("fundamental_momentum", 0) >= 60:
        ra = g("rev_accel")
        if not np.isnan(ra):
            br.append(f"Next 10-Q shows revenue growth decelerating (YoY accel {ra:+.1%} turning negative)")
        gm = g("gross_margin")
        if not np.isnan(gm):
            br.append(f"TTM gross margin drops below {gm - 0.03:.0%} (currently {gm:.0%})")
    if components.get("price_momentum", 0) >= 60:
        br.append("Closes below its 200-day average and stays there for 15+ sessions")
        br.append("Relative strength vs sector turns negative over a rolling 20-session window")
    if "insider_cluster_into_drawdown" in pats:
        br.append("A new cluster of insider selling appears (2+ officers/directors, open-market)")
    if "post_earnings_drift_window" in pats or "strong_report_sold_off" in pats:
        br.append("Price gives back the full post-earnings move (drift thesis invalidated)")
    lev = g("net_debt_to_op")
    if not np.isnan(lev) and lev > 2:
        br.append(f"Interest coverage falls below 3x or net debt/operating income rises above {lev + 1:.1f}x")
    br.append("An 8-K discloses a material adverse event (guidance cut, exec departure, investigation)")
    return br[:6]
