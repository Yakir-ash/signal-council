"""Multi-horizon probabilistic predictions (DESIGN.md §5 / red-team Attack 3).

The integrity rule: a probability shown to the user traces to one of exactly two
places, and the output says which:

1. EMPIRICAL tables (data/calibration/prob_tables.json), built by the walk-forward
   backtest from score-decile × regime × horizon conditional forward-return
   distributions, later corrected by live-ledger isotonic recalibration.
2. COLD_START mapping — a deliberately conservative linear map, hard-clamped to
   the configured band (default [0.35, 0.68]). The system cannot claim
   confidence it has not earned; there is no third path and no free parameter
   an LLM can inflate.

Return ranges/downside always come from the name's own realized volatility and
the empirical distribution when available — never from a point-estimate fantasy.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import numpy as np

from ..config import thresholds_cfg
from ..paths import CALIBRATION

TABLES_PATH = CALIBRATION / "prob_tables.json"


@dataclass
class HorizonPrediction:
    horizon: str
    horizon_days: int
    p_positive: float
    p_beat_benchmark: float
    exp_return_low: float          # 25th pct
    exp_return_high: float         # 75th pct
    downside_p5: float
    expected_vol: float
    basis: str                     # 'empirical:<table_version>' | 'cold_start_clamped'

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


def _load_tables() -> dict | None:
    if TABLES_PATH.exists():
        with open(TABLES_PATH) as f:
            return json.load(f)
    return None


def predict(opportunity: float, realized_vol: float, regime_label: str,
            regime_score: float) -> list[HorizonPrediction]:
    cfg = thresholds_cfg()
    tables = _load_tables()
    out: list[HorizonPrediction] = []
    vol = realized_vol if realized_vol and not np.isnan(realized_vol) else 0.30

    for h in cfg["horizons"]:
        name, days = h["name"], h["days"]
        t_frac = days / 252.0
        h_vol = vol * np.sqrt(t_frac)

        if tables:
            decile = f"d{min(9, int(opportunity // 10))}"
            key_opts = [f"{decile}|{regime_label}|{name}", f"{decile}|*|{name}"]
            cell = next((tables["cells"][k] for k in key_opts if k in tables["cells"]), None)
            if cell and cell.get("n", 0) >= 30:
                lo, hi = cfg["probability_clamp"]["calibrated"]
                out.append(HorizonPrediction(
                    horizon=name, horizon_days=days,
                    p_positive=float(np.clip(cell["p_positive"], lo, hi)),
                    p_beat_benchmark=float(np.clip(cell["p_beat"], lo, hi)),
                    exp_return_low=cell["q25"], exp_return_high=cell["q75"],
                    downside_p5=cell["q05"], expected_vol=h_vol,
                    basis=f"empirical:{tables.get('version', '?')}(n={cell['n']})"))
                continue

        # ---- cold start: conservative, clamped, vol-anchored
        lo, hi = cfg["probability_clamp"]["cold_start"]
        edge = (opportunity - 50.0) / 50.0                    # -1..1
        p_beat = float(np.clip(0.5 + 0.16 * edge, lo, hi))
        # market base rate ~54-56% positive per quarter historically; tilt by regime
        base_pos = 0.5 + 0.10 * np.sqrt(t_frac) + 0.05 * regime_score
        p_pos = float(np.clip(base_pos + 0.12 * edge, lo, hi))
        drift = 0.06 * t_frac * (1 + regime_score) / 2 + 0.10 * edge * t_frac
        out.append(HorizonPrediction(
            horizon=name, horizon_days=days,
            p_positive=p_pos, p_beat_benchmark=p_beat,
            exp_return_low=float(drift - 0.674 * h_vol),
            exp_return_high=float(drift + 0.674 * h_vol),
            downside_p5=float(drift - 1.645 * h_vol),
            expected_vol=float(h_vol),
            basis="cold_start_clamped"))
    return out


def confidence(data_confidence: float, component_scores: dict, regime_label: str,
               calibrated: bool) -> float:
    """Confidence = evidence agreement × data quality × regime conviction.
    Capped hard in cold start. NOT a vibe — every term is computed."""
    from ..regime.classifier import REGIME_CONVICTION

    comps = [v for k, v in component_scores.items()
             if k in ("fundamental_momentum", "price_momentum", "quality",
                      "valuation", "technical", "divergence")]
    strong = sum(1 for c in comps if c >= 60)
    weak = sum(1 for c in comps if c <= 40)
    agreement = (strong - weak) / max(1, len(comps))          # -1..1
    c = 0.40 + 0.25 * max(0.0, agreement)
    c *= (0.5 + 0.5 * data_confidence / 100.0)
    c *= REGIME_CONVICTION.get(regime_label, 0.75)
    cap = 0.68 if not calibrated else 0.90
    return float(np.clip(c, 0.05, cap))
