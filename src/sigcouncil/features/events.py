"""Event & flow features: insider clusters, earnings proximity, announcement drift,
valuation context. All point-in-time via filed/observed dates."""
from __future__ import annotations

import numpy as np
import pandas as pd


def insider_features(insider_df: pd.DataFrame, as_of: pd.Timestamp,
                     window_days: int = 90) -> pd.DataFrame:
    """Cluster-weighted insider signal. Open-market purchases ('P') by multiple
    distinct insiders are the strongest documented variant of this signal;
    routine plan sales get little weight."""
    cols = ["ticker", "insider_net_value_90", "insider_buyers_90", "insider_sellers_90",
            "insider_buy_value_90", "insider_cluster"]
    if insider_df is None or insider_df.empty:
        return pd.DataFrame(columns=cols)
    df = insider_df.copy()
    df["filed_at"] = pd.to_datetime(df["filed_at"], errors="coerce")
    df = df[(df["filed_at"] <= as_of)
            & (df["filed_at"] >= as_of - pd.Timedelta(days=window_days))]
    rows = []
    for t, g in df.groupby("ticker"):
        buys = g[(g["kind"] == "P") & (g["value"].notna()) & (g["value"] > 0)]
        sells = g[(g["kind"] == "S") & (g["value"].notna())]
        buy_val = float(buys["value"].sum())
        sell_val = float(sells["value"].abs().sum())
        rows.append({
            "ticker": t,
            "insider_net_value_90": buy_val - sell_val,
            "insider_buy_value_90": buy_val,
            "insider_buyers_90": int(buys["filer"].nunique()),
            "insider_sellers_90": int(sells["filer"].nunique()),
            "insider_cluster": int(buys["filer"].nunique() >= 2 and buy_val > 250_000),
        })
    return pd.DataFrame(rows, columns=cols)


def earnings_event_features(fund_feats: pd.DataFrame, prices: pd.DataFrame,
                            as_of: pd.Timestamp, benchmark: str = "SPY") -> pd.DataFrame:
    """Post-earnings drift inputs: reaction to the last report and time since it.
    Next-earnings proximity is ESTIMATED (last fiscal report + ~91d cycle) and
    tagged as such — free data has no reliable forward calendar."""
    if fund_feats.empty or "last_report_filed" not in fund_feats.columns:
        return pd.DataFrame(columns=["ticker", "days_since_report", "days_to_report_est",
                                     "report_reaction", "post_report_drift"])
    closes = prices.pivot_table(index="date", columns="ticker", values="adj_close",
                                aggfunc="last").sort_index()
    closes = closes[closes.index <= as_of]
    bench = closes.get(benchmark)
    rows = []
    for _, r in fund_feats.iterrows():
        t = r["ticker"]
        filed = pd.to_datetime(r.get("last_report_filed"), errors="coerce")
        if pd.isna(filed) or t not in closes.columns:
            continue
        c = closes[t].dropna()
        days_since = int((as_of - filed).days)
        f = {"ticker": t, "days_since_report": days_since,
             "days_to_report_est": max(0, 91 - days_since)}
        pre = c[c.index < filed]
        post = c[c.index >= filed]
        if len(pre) > 0 and len(post) > 0:
            base = float(pre.iloc[-1])
            # reaction: first 2 sessions after filing
            k = min(2, len(post))
            f["report_reaction"] = float(post.iloc[k - 1]) / base - 1
            # drift since reaction, benchmark-relative
            drift = float(c.iloc[-1]) / float(post.iloc[k - 1]) - 1
            if bench is not None:
                b = bench.dropna()
                b_post = b[b.index >= filed]
                if len(b_post) >= k:
                    drift -= float(b.iloc[-1]) / float(b_post.iloc[k - 1]) - 1
            f["post_report_drift"] = drift
        rows.append(f)
    return pd.DataFrame(rows)


def valuation_features(fund_feats: pd.DataFrame, tech: pd.DataFrame,
                       prices: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Valuation multiples + own-history percentile (3y of price/TTM-rev).
    History percentile uses CURRENT share count across the window — an
    approximation, so it is emitted as *_est and treated as MODEL ESTIMATE."""
    if fund_feats.empty:
        return pd.DataFrame(columns=["ticker"])
    px = tech.set_index("ticker")["px_last"] if not tech.empty else pd.Series(dtype=float)
    closes = prices.pivot_table(index="date", columns="ticker", values="close",
                                aggfunc="last").sort_index()
    closes = closes[closes.index <= as_of]
    rows = []
    for _, r in fund_feats.iterrows():
        t = r["ticker"]
        shares = r.get("shares_outstanding") or r.get("shares_diluted_last")
        price = px.get(t, np.nan)
        f: dict = {"ticker": t}
        if shares and not np.isnan(shares) and price and not np.isnan(price):
            mcap = shares * price
            f["mcap"] = mcap
            rev = r.get("rev_ttm", np.nan)
            fcf = r.get("fcf_ttm", np.nan)
            nd = r.get("net_debt", 0) or 0
            if rev and not np.isnan(rev) and rev > 0:
                f["ev_sales"] = (mcap + nd) / rev
                # Own-history valuation percentile over ~3y. Free data has no
                # historical share counts or PIT revenue snapshots per day, so we
                # approximate P/S_t ∝ price_t / rev-per-share-growth-adjusted; since
                # revenue moves far slower than price, price relative to a
                # revenue-growth-adjusted trend is a serviceable estimate. Emitted
                # as *_est => rendered as MODEL ESTIMATE, never FACT.
                if t in closes.columns:
                    c3 = closes[t].dropna().tail(756)
                    if len(c3) > 252:
                        g = r.get("rev_yoy", np.nan)
                        g = 0.0 if g is None or np.isnan(g) else float(np.clip(g, -0.5, 1.0))
                        n = len(c3)
                        # de-trend price path by revenue growth so the percentile
                        # compares valuation, not just price level
                        growth_path = (1 + g) ** (np.arange(n)[::-1] / 252.0)
                        ps_proxy = c3.values * growth_path
                        f["ev_sales_pctile_3y_est"] = float((ps_proxy <= ps_proxy[-1]).mean())
            if fcf and not np.isnan(fcf) and mcap > 0:
                f["fcf_yield"] = fcf / mcap
        rows.append(f)
    return pd.DataFrame(rows)
