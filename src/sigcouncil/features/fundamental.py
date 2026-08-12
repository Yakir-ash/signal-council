"""Fundamental features from EDGAR XBRL facts — strictly point-in-time.

Every input row carries `filed` (the date the market could first know it).
`compute(facts, as_of)` passes facts through the PIT gate before ANY math.

The question asked is never "is it cheap" but "is the trajectory of the
business improving faster than the valuation implies" (DESIGN.md §2/§7):
levels matter less than trends and accelerations.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..store.panels import pit_filter

FLOW = {"revenue", "cost_of_revenue", "gross_profit", "operating_income", "net_income",
        "operating_cf", "capex", "interest_expense", "eps_diluted"}
STOCK = {"assets", "equity", "cash", "long_term_debt", "short_term_debt", "inventory",
         "shares_diluted", "shares_outstanding_dei"}


def _quarterly_series(f: pd.DataFrame, concept: str) -> pd.Series:
    """Latest-filed value per fiscal quarter for a flow concept (duration ≈ 1 quarter)."""
    g = f[f["concept"] == concept].copy()
    if g.empty:
        return pd.Series(dtype=float)
    g["period_start"] = pd.to_datetime(g["period_start"], errors="coerce")
    g["period_end"] = pd.to_datetime(g["period_end"], errors="coerce")
    g["dur"] = (g["period_end"] - g["period_start"]).dt.days
    q = g[(g["dur"] >= 70) & (g["dur"] <= 100)]
    if q.empty:
        return pd.Series(dtype=float)
    q = q.sort_values("filed").drop_duplicates(subset=["period_end"], keep="last")
    return q.set_index("period_end")["value"].astype(float).sort_index()


def _latest_stock(f: pd.DataFrame, concept: str) -> tuple[float, str | None]:
    g = f[f["concept"] == concept].copy()
    if g.empty:
        return np.nan, None
    g["period_end"] = pd.to_datetime(g["period_end"], errors="coerce")
    g = g.sort_values(["period_end", "filed"])
    row = g.iloc[-1]
    return float(row["value"]), str(row["period_end"].date())


def _ttm(s: pd.Series) -> float:
    return float(s.tail(4).sum()) if len(s) >= 4 else np.nan


def _yoy_growth(s: pd.Series, lag_q: int = 4) -> float:
    if len(s) < lag_q + 4:
        return np.nan
    now, prev = s.tail(4).sum(), s.iloc[-(lag_q + 4):-lag_q].sum()
    if prev == 0:
        return np.nan
    return float(now / abs(prev) - 1)


def compute_one(facts: pd.DataFrame, as_of: pd.Timestamp) -> dict:
    """Fundamental feature dict for one ticker's facts frame, as of a date."""
    f = pit_filter(facts, as_of, observed_col="filed")
    out: dict = {}
    if f.empty:
        return out
    rev = _quarterly_series(f, "revenue")
    gp = _quarterly_series(f, "gross_profit")
    if gp.empty:
        cor = _quarterly_series(f, "cost_of_revenue")
        if not rev.empty and not cor.empty:
            gp = (rev - cor.reindex(rev.index)).dropna()
    ni = _quarterly_series(f, "net_income")
    op = _quarterly_series(f, "operating_income")
    ocf = _quarterly_series(f, "operating_cf")
    capex = _quarterly_series(f, "capex")
    eps = _quarterly_series(f, "eps_diluted")
    interest = _quarterly_series(f, "interest_expense")

    assets, _ = _latest_stock(f, "assets")
    equity, _ = _latest_stock(f, "equity")
    cash, _ = _latest_stock(f, "cash")
    ltd, _ = _latest_stock(f, "long_term_debt")
    std_, _ = _latest_stock(f, "short_term_debt")
    inv, _ = _latest_stock(f, "inventory")
    sh_dil = _quarterly_series(f, "shares_diluted")
    sh_out, _ = _latest_stock(f, "shares_outstanding_dei")

    rev_ttm = _ttm(rev)
    out["rev_ttm"] = rev_ttm
    out["rev_yoy"] = _yoy_growth(rev)
    # acceleration: latest-quarter YoY vs previous-quarter YoY
    if len(rev) >= 9:
        q_yoy_now = rev.iloc[-1] / abs(rev.iloc[-5]) - 1 if rev.iloc[-5] != 0 else np.nan
        q_yoy_prev = rev.iloc[-2] / abs(rev.iloc[-6]) - 1 if rev.iloc[-6] != 0 else np.nan
        out["rev_q_yoy"] = float(q_yoy_now)
        out["rev_accel"] = float(q_yoy_now - q_yoy_prev)
    out["eps_yoy"] = _yoy_growth(eps)

    gp_ttm, ni_ttm, op_ttm, ocf_ttm = _ttm(gp), _ttm(ni), _ttm(op), _ttm(ocf)
    capex_ttm = _ttm(capex)
    if rev_ttm and not np.isnan(rev_ttm) and rev_ttm != 0:
        out["gross_margin"] = gp_ttm / rev_ttm if not np.isnan(gp_ttm) else np.nan
        out["op_margin"] = op_ttm / rev_ttm if not np.isnan(op_ttm) else np.nan
        fcf_ttm = (ocf_ttm - capex_ttm) if not np.isnan(ocf_ttm) and not np.isnan(capex_ttm) else np.nan
        out["fcf_ttm"] = fcf_ttm
        out["fcf_margin"] = fcf_ttm / rev_ttm if not np.isnan(fcf_ttm) else np.nan
        out["capex_intensity"] = capex_ttm / rev_ttm if not np.isnan(capex_ttm) else np.nan
    # margin trend: TTM gross margin now vs 4 quarters ago
    if len(gp) >= 8 and len(rev) >= 8:
        gm_now = gp.tail(4).sum() / rev.tail(4).sum()
        gm_prev = gp.iloc[-8:-4].sum() / rev.iloc[-8:-4].sum()
        out["gross_margin_delta"] = float(gm_now - gm_prev)
    if len(op) >= 8 and len(rev) >= 8:
        out["op_margin_delta"] = float(op.tail(4).sum() / rev.tail(4).sum()
                                       - op.iloc[-8:-4].sum() / rev.iloc[-8:-4].sum())

    # quality / balance sheet
    if not np.isnan(assets) and assets > 0:
        out["gp_over_assets"] = gp_ttm / assets if not np.isnan(gp_ttm) else np.nan
        if not np.isnan(ni_ttm) and not np.isnan(ocf_ttm):
            out["accruals"] = (ni_ttm - ocf_ttm) / assets   # high accruals = red flag
    if not np.isnan(equity) and equity > 0 and not np.isnan(ni_ttm):
        out["roe"] = ni_ttm / equity
    debt = (0 if np.isnan(ltd) else ltd) + (0 if np.isnan(std_) else std_)
    out["total_debt"] = debt
    out["cash"] = cash
    out["net_debt"] = debt - (0 if np.isnan(cash) else cash)
    if not np.isnan(op_ttm) and op_ttm > 0:
        out["net_debt_to_op"] = out["net_debt"] / op_ttm
        it = _ttm(interest)
        out["interest_coverage"] = op_ttm / it if it and not np.isnan(it) and it > 0 else np.nan
    if not np.isnan(equity) and equity > 0:
        out["debt_to_equity"] = debt / equity

    # dilution / buybacks
    if len(sh_dil) >= 8:
        prev = sh_dil.iloc[-5]
        out["share_change_yoy"] = float(sh_dil.iloc[-1] / prev - 1) if prev else np.nan
    out["shares_outstanding"] = sh_out
    out["shares_diluted_last"] = float(sh_dil.iloc[-1]) if len(sh_dil) else np.nan
    if not np.isnan(inv) and rev_ttm and rev_ttm > 0:
        out["inventory_to_rev"] = inv / rev_ttm

    # last report event (for PEAD features in events.py)
    forms = f[f["form"].isin(["10-Q", "10-K", "8-K"])]
    if not forms.empty:
        out["last_report_filed"] = str(pd.to_datetime(forms["filed"]).max().date())
    return out


def compute(facts_all: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for t, g in facts_all.groupby("ticker"):
        d = compute_one(g, as_of)
        if d:
            d["ticker"] = t
            rows.append(d)
    return pd.DataFrame(rows)
