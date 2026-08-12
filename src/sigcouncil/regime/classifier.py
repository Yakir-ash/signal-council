"""Market regime engine (DESIGN.md §8 / red-team Attack 9).

Deliberately SIMPLE: five standard macro/market dimensions, transparent rules,
no fitted parameters to overfit. Regime acts primarily as a conviction scaler
and signal re-weighter, not a timing oracle. Identified with lag, by design.

Dimensions (each -1..+1):
  trend    SPY vs 200dma + 50dma slope
  vol      VIX level & direction (fallback: SPY realized vol)
  breadth  % of universe above 200dma
  credit   HY OAS level & 3m change
  curve    10y-2y spread (inversion = late-cycle warning)
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class RegimeResult:
    label: str
    score: float                    # -1 (deep risk-off) .. +1 (healthy expansion)
    subscores: dict
    inputs: dict                    # raw values with sources, for the dashboard
    as_of: str

    def to_dict(self) -> dict:
        return asdict(self)


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return float(np.clip(x, lo, hi))


def classify(prices: pd.DataFrame, macro: pd.DataFrame, as_of: pd.Timestamp,
             universe: list[str], benchmark: str = "SPY") -> RegimeResult:
    closes = prices.pivot_table(index="date", columns="ticker", values="adj_close",
                                aggfunc="last").sort_index()
    closes = closes[closes.index <= as_of]
    inputs: dict = {}
    sub: dict[str, float] = {}

    # --- trend
    spy = closes.get(benchmark, pd.Series(dtype=float)).dropna()
    if len(spy) > 210:
        ma200 = spy.rolling(200).mean()
        ma50 = spy.rolling(50).mean()
        above = spy.iloc[-1] / ma200.iloc[-1] - 1
        slope = ma50.iloc[-1] / ma50.iloc[-22] - 1
        sub["trend"] = _clip(6 * above + 12 * slope)
        inputs["spy_vs_200dma"] = {"value": round(float(above), 4), "source": "prices:SPY"}
        inputs["ma50_slope_1m"] = {"value": round(float(slope), 4), "source": "prices:SPY"}
    else:
        sub["trend"] = 0.0

    # --- volatility
    m = macro.copy()
    if not m.empty:
        m["date"] = pd.to_datetime(m["date"])
        m = m[m["date"] <= as_of]
    vix = m[m["series"] == "VIXCLS"]["value"] if not m.empty else pd.Series(dtype=float)
    if len(vix) > 21:
        v_now = float(vix.iloc[-1])
        v_prev = float(vix.tail(21).mean())
        sub["vol"] = _clip((18 - v_now) / 10 - 0.5 * np.sign(v_now - v_prev) * min(1, abs(v_now - v_prev) / 5))
        inputs["vix"] = {"value": v_now, "source": "FRED:VIXCLS"}
    elif len(spy) > 63:
        rv = float(spy.pct_change().tail(21).std() * np.sqrt(252))
        sub["vol"] = _clip((0.15 - rv) / 0.10)
        inputs["spy_realized_vol_21d"] = {"value": round(rv, 4), "source": "prices:SPY"}
    else:
        sub["vol"] = 0.0

    # --- breadth
    stocks = [t for t in universe if t in closes.columns]
    if stocks:
        above200 = []
        for t in stocks:
            c = closes[t].dropna()
            if len(c) > 200:
                above200.append(float(c.iloc[-1] > c.rolling(200).mean().iloc[-1]))
        if above200:
            frac = float(np.mean(above200))
            sub["breadth"] = _clip((frac - 0.5) * 2.5)
            inputs["pct_above_200dma"] = {"value": round(frac, 3),
                                          "source": f"prices:universe({len(above200)})"}
    sub.setdefault("breadth", 0.0)

    # --- credit
    hy = m[m["series"] == "BAMLH0A0HYM2"]["value"] if not m.empty else pd.Series(dtype=float)
    if len(hy) > 63:
        oas = float(hy.iloc[-1])
        chg = oas - float(hy.iloc[-63])
        sub["credit"] = _clip((4.5 - oas) / 2.0 - chg / 1.0)
        inputs["hy_oas"] = {"value": oas, "source": "FRED:BAMLH0A0HYM2"}
        inputs["hy_oas_3m_chg"] = {"value": round(chg, 2), "source": "FRED:BAMLH0A0HYM2"}
    else:
        sub["credit"] = 0.0

    # --- curve
    curve = m[m["series"] == "T10Y2Y"]["value"] if not m.empty else pd.Series(dtype=float)
    if len(curve) > 0:
        cv = float(curve.iloc[-1])
        sub["curve"] = _clip(cv / 1.0, -1, 0.5)
        inputs["t10y2y"] = {"value": cv, "source": "FRED:T10Y2Y"}
    else:
        sub["curve"] = 0.0

    weights = {"trend": 0.30, "vol": 0.20, "breadth": 0.20, "credit": 0.20, "curve": 0.10}
    score = sum(sub[k] * w for k, w in weights.items())

    # label rules — coarse on purpose
    trend, vol, credit = sub["trend"], sub["vol"], sub["credit"]
    if score > 0.45 and trend > 0.3:
        label = "bullish_expansion"
    elif score > 0.15:
        label = "cautious_bullish"
    elif vol < -0.6 and trend < 0:
        label = "high_volatility"
    elif credit < -0.5 and trend < -0.3:
        label = "risk_off"
    elif trend < -0.4 and score < -0.3:
        label = "recessionary" if sub["curve"] < -0.3 else "risk_off"
    elif trend > 0.2 and score <= 0.15:
        label = "recovery"
    else:
        label = "sideways"

    return RegimeResult(label=label, score=round(float(score), 3),
                        subscores={k: round(v, 3) for k, v in sub.items()},
                        inputs=inputs, as_of=str(as_of.date()))


# conviction scaling per regime: multiplies confidence & tightens thresholds
REGIME_CONVICTION = {
    "bullish_expansion": 1.00,
    "cautious_bullish": 0.90,
    "recovery": 0.85,
    "sideways": 0.75,
    "high_volatility": 0.60,
    "risk_off": 0.50,
    "recessionary": 0.45,
}
