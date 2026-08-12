"""Provider abstractions. The intelligence engine imports ONLY these interfaces.

Swapping yfinance for Polygon (or adding FMP estimates) means writing one new
class here and flipping a config entry — zero changes to features/models/scoring.

Every provider returns data with explicit `source` and observation timestamps so
the point-in-time store can enforce the observed_at discipline.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd


@dataclass
class Filing:
    ticker: str
    cik: str
    accession: str
    form: str
    filed_at: str          # ISO date — this IS observed_at for everything inside it
    period: str | None
    url: str
    local_text_path: str | None = None


@dataclass
class NewsItem:
    ticker: str
    published_at: str
    source: str
    title: str
    url: str
    summary: str = ""


@dataclass
class InsiderTx:
    ticker: str
    filer: str
    role: str
    tx_date: str
    filed_at: str
    kind: str              # 'P' open-market purchase, 'S' sale, other codes passed through
    shares: float | None
    price: float | None
    value: float | None


class PriceProvider(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    def daily_ohlcv(self, tickers: list[str], start: date, end: date) -> pd.DataFrame:
        """Long frame: ticker, date, open, high, low, close, adj_close, volume, source."""


class FundamentalsProvider(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    def company_facts(self, ticker: str) -> pd.DataFrame:
        """Long frame: ticker, concept, unit, period_start, period_end, value, form,
        filed (ISO date = observed_at), fiscal_frame."""


class FilingsProvider(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    def recent_filings(self, ticker: str, forms: list[str], limit: int = 40) -> list[Filing]: ...

    @abc.abstractmethod
    def filing_text(self, filing: Filing, max_chars: int = 400_000) -> str: ...


class InsiderProvider(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    def insider_transactions(self, ticker: str, since: date) -> list[InsiderTx]: ...


class MacroProvider(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    def series(self, series_id: str, start: date) -> pd.DataFrame:
        """Frame: date, value, series, source."""


class NewsProvider(abc.ABC):
    name: str = "abstract"

    @abc.abstractmethod
    def recent_news(self, ticker: str, limit: int = 25) -> list[NewsItem]: ...


@dataclass
class ProviderSet:
    """The resolved bundle the pipeline uses. Built by providers.registry.build()."""
    prices: PriceProvider
    prices_fallback: PriceProvider | None
    fundamentals: FundamentalsProvider
    filings: FilingsProvider
    insiders: InsiderProvider
    macro: MacroProvider
    news: list[NewsProvider] = field(default_factory=list)
