"""Free news: yfinance per-ticker news + EDGAR 8-K stream as the reliable spine.

Design stance (DESIGN.md §3): free news feeds are thin. Material corporate events
are legally required to appear in 8-Ks, so the filings stream is the *primary*
catalyst source; headline feeds add color and an attention-anomaly signal only.
"""
from __future__ import annotations

from ..logutil import get_logger
from .base import NewsItem, NewsProvider

log = get_logger("news")


class YFinanceNews(NewsProvider):
    name = "yfinance_news"

    def recent_news(self, ticker: str, limit: int = 25) -> list[NewsItem]:
        import yfinance as yf

        try:
            raw = yf.Ticker(ticker).news or []
        except Exception as e:  # noqa: BLE001
            log.warning("%s news failed: %s", ticker, e)
            return []
        out = []
        for it in raw[:limit]:
            c = it.get("content", it)  # yfinance changed shape across versions
            title = c.get("title") or ""
            ts = (c.get("pubDate") or c.get("displayTime")
                  or str(it.get("providerPublishTime", "")))
            url = ((c.get("canonicalUrl") or {}).get("url")
                   if isinstance(c.get("canonicalUrl"), dict) else c.get("link", "")) or ""
            src = (c.get("provider") or {}).get("displayName", "") if isinstance(
                c.get("provider"), dict) else str(c.get("publisher", ""))
            if title:
                out.append(NewsItem(ticker=ticker, published_at=str(ts), source=src or "yahoo",
                                    title=title, url=url,
                                    summary=(c.get("summary") or "")[:500]))
        return out
