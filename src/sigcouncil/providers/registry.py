"""Build the ProviderSet from config. This is the ONLY place concrete providers
are instantiated — swapping providers is a change here + config, nowhere else."""
from __future__ import annotations

from .base import ProviderSet
from .edgar import Edgar
from .fred import Fred
from .rss_news import YFinanceNews
from .stooq import StooqPrices
from .yf_prices import YFinancePrices


def build() -> ProviderSet:
    edgar = Edgar()
    return ProviderSet(
        prices=YFinancePrices(),
        prices_fallback=StooqPrices(),
        fundamentals=edgar,
        filings=edgar,
        insiders=edgar,
        macro=Fred(),
        news=[YFinanceNews()],
    )
