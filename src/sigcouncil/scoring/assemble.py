"""Final score assembly: components → Opportunity Score → gates → explanations.

Hard rules (red-team Attacks 3/6):
- Gates run BEFORE ranking; there is no "best available" fallback path.
- Every score ships with its decomposition, reasons, and thesis breakers.
- An empty high-conviction list is a normal, expected outcome.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import thresholds_cfg, weights_cfg
from ..models import probabilities as prob
from . import risk as risk_mod


def opportunity_score(components: pd.DataFrame) -> pd.Series:
    w = weights_cfg()["components"]
    total = sum(w.values())
    s = pd.Series(0.0, index=components.index)
    for name, weight in w.items():
        col = components[name] if name in components.columns else pd.Series(50.0, index=components.index)
        s += col.fillna(50.0) * weight
    return (s / total).round(1)


def eligibility_gates(feat: pd.DataFrame, universe_cfg: dict,
                      quarantined: list[str]) -> pd.Series:
    """Returns a Series of gate-failure reason strings ('' = eligible)."""
    e = universe_cfg["eligibility"]
    f = feat.set_index("ticker")
    reasons = pd.Series("", index=f.index)

    px = f.get("px_last", pd.Series(np.nan, index=f.index))
    reasons[px < e["min_price"]] += f"price<${e['min_price']};"
    dv = f.get("dollar_vol_med_60", pd.Series(np.nan, index=f.index))
    reasons[dv < e["min_median_dollar_volume"]] += "illiquid;"
    reasons[dv.isna()] += "no_liquidity_data;"
    for t in quarantined:
        if t in reasons.index:
            reasons[t] += "data_quarantine;"
    return reasons


def reasons_for(row: pd.Series, components: dict, divergence_patterns: list[dict],
                max_reasons: int = 5) -> list[str]:
    """The 3-5 strongest evidence-backed reasons, drawn from actual drivers."""
    cands: list[tuple[float, str]] = []

    def g(col, default=np.nan):
        v = row.get(col, default)
        return float(v) if v is not None and not pd.isna(v) else default

    if components.get("fundamental_momentum", 50) >= 60:
        ra, gm_d = g("rev_accel"), g("gross_margin_delta")
        msg = "Business momentum: "
        bits = []
        if not np.isnan(g("rev_yoy")):
            bits.append(f"revenue {g('rev_yoy'):+.0%} YoY")
        if not np.isnan(ra) and ra > 0:
            bits.append(f"growth accelerating ({ra:+.1%} q/q)")
        if not np.isnan(gm_d) and gm_d > 0:
            bits.append(f"gross margin +{gm_d:.1%} vs last year")
        if bits:
            cands.append((components["fundamental_momentum"], msg + ", ".join(bits) + " [FACT: SEC XBRL]"))
    if components.get("price_momentum", 50) >= 60:
        cands.append((components["price_momentum"],
                      f"Market confirmation: 12-1 momentum {g('mom_12_1'):+.0%}, "
                      f"RS vs S&P {g('rs_spy_63'):+.1%} (63d) [FACT: prices]"))
    if components.get("quality", 50) >= 60:
        cands.append((components["quality"],
                      f"Quality: FCF margin {g('fcf_margin'):.0%}, "
                      f"accruals {g('accruals'):+.2f} [FACT: SEC XBRL]"))
    if components.get("valuation", 50) >= 60:
        vp = g("ev_sales_pctile_3y_est")
        if not np.isnan(vp):
            cands.append((components["valuation"],
                          f"Valuation context: sits at the {vp:.0%} pctile of its own 3y range "
                          f"despite improving economics [MODEL ESTIMATE]"))
    for p in (divergence_patterns or []):
        cands.append((60 + p["points"],
                      f"Divergence: {p['pattern'].replace('_', ' ')} "
                      f"({', '.join(f'{k}={v:+.2%}' if isinstance(v, float) and abs(v) < 1 else f'{k}={v}' for k, v in p['inputs'].items())}) "
                      f"[MODEL ESTIMATE]"))
    if components.get("insider", 50) >= 65:
        cands.append((components["insider"], "Insider signal: cluster open-market buying "
                      "in the last 90 days [FACT: SEC Form 4]"))
    cands.sort(key=lambda x: -x[0])
    return [c[1] for c in cands[:max_reasons]]


def assemble(feat: pd.DataFrame, components: pd.DataFrame, divergence: pd.DataFrame,
             quality_per_ticker: pd.DataFrame, regime: dict, universe_cfg: dict,
             quarantined: list[str], calibrated: bool) -> pd.DataFrame:
    """Produce the final scored frame, one row per eligible ticker."""
    th = thresholds_cfg()
    f = feat.merge(components, on="ticker", how="left") \
            .merge(divergence, on="ticker", how="left") \
            .merge(quality_per_ticker[["ticker", "data_confidence"]], on="ticker", how="left")
    f["data_confidence"] = f["data_confidence"].fillna(40.0)
    f["divergence"] = f["divergence"].fillna(30.0)

    comp_cols = list(weights_cfg()["components"].keys())
    for c in comp_cols:
        if c not in f.columns:
            f[c] = 50.0
    f["opportunity"] = opportunity_score(f[comp_cols])

    risk_df = risk_mod.compute(feat)
    f = f.merge(risk_df, on="ticker", how="left")

    gates = eligibility_gates(feat, universe_cfg, quarantined)
    f["gate_fail"] = f["ticker"].map(gates).fillna("")

    rows = []
    for _, r in f.iterrows():
        comps = {c: float(r[c]) for c in comp_cols}
        comps["divergence"] = float(r["divergence"])
        conf = prob.confidence(float(r["data_confidence"]), comps,
                               regime["label"], calibrated)
        preds = prob.predict(float(r["opportunity"]), r.get("realized_vol_63", np.nan),
                             regime["label"], regime["score"])
        pats = r.get("divergence_patterns") or []
        rows.append({
            "ticker": r["ticker"],
            "opportunity": float(r["opportunity"]),
            "risk": float(r["risk"]) if not pd.isna(r["risk"]) else 50.0,
            "divergence": float(r["divergence"]),
            "data_confidence": float(r["data_confidence"]),
            "confidence": round(conf, 3),
            "components": comps,
            "risk_components": r.get("risk_components", {}),
            "predictions": [p.to_dict() for p in preds],
            "reasons": reasons_for(r, comps, pats),
            "divergence_patterns": pats,
            "thesis_breakers": risk_mod.thesis_breakers(r, comps, pats),
            "gate_fail": r["gate_fail"],
            "px_last": r.get("px_last", np.nan),
            "px_last_date": r.get("px_last_date", ""),
            "evidence": r.get("evidence", {}),
        })
    out = pd.DataFrame(rows).sort_values("opportunity", ascending=False).reset_index(drop=True)
    return out


def select_top(scored: pd.DataFrame) -> pd.DataFrame:
    """High-conviction selection. No fallback: zero rows is a valid result."""
    th = thresholds_cfg()["high_conviction"]
    ok = scored[(scored["gate_fail"] == "")
                & (scored["opportunity"] >= th["min_opportunity"])
                & (scored["confidence"] >= th["min_confidence"])
                & (scored["risk"] <= th["max_risk"])
                & (scored["data_confidence"] >= th["min_data_confidence"])]
    return ok.head(th["max_shown"])


def select_avoid(scored: pd.DataFrame) -> pd.DataFrame:
    th = thresholds_cfg()["avoid_list"]
    bad = scored[(scored["gate_fail"] == "")
                 & (scored["opportunity"] <= th["max_opportunity"])
                 & (scored["risk"] >= th["min_risk"])]
    return bad.sort_values(["risk", "opportunity"], ascending=[False, True]).head(th["max_shown"])
