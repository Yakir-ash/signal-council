"""SEC EDGAR — the system's source of truth for fundamentals, filings and insiders.

Why EDGAR is the backbone of the free-data stack:
- XBRL company facts carry the exact `filed` date for every value → point-in-time
  by construction (the observed_at discipline gets ground truth for free).
- Filings (10-K/10-Q/8-K/Form 4) are legally mandated — more reliable than news.
Rate limit: SEC allows ~10 req/s with a descriptive User-Agent; we stay well under.
"""
from __future__ import annotations

import json
import re
from datetime import date

import pandas as pd

from ..logutil import get_logger
from ..paths import CACHE, FILINGS
from .base import Filing, FilingsProvider, FundamentalsProvider, InsiderProvider, InsiderTx
from .http import Http

log = get_logger("edgar")

# Curated XBRL concepts: everything the fundamental feature engine needs, nothing more.
# key = canonical name used downstream; value = list of us-gaap/dei tags tried in order
# (issuers use different tags for the same economic quantity).
CONCEPTS: dict[str, list[str]] = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "cost_of_revenue": ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": ["NetIncomeLoss"],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "assets": ["Assets"],
    "equity": ["StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "cash": ["CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents"],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "short_term_debt": ["LongTermDebtCurrent", "DebtCurrent"],
    "operating_cf": ["NetCashProvidedByUsedInOperatingActivities",
                     "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
    "inventory": ["InventoryNet"],
    "interest_expense": ["InterestExpense", "InterestExpenseDebt", "InterestExpenseNonoperating"],
    "shares_diluted": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
    "shares_outstanding_dei": ["EntityCommonStockSharesOutstanding"],  # dei namespace
}


class Edgar(FundamentalsProvider, FilingsProvider, InsiderProvider):
    name = "sec_edgar"

    def __init__(self):
        self.http = Http(min_interval=0.13)  # ≈7.5 req/s max, under SEC's 10/s
        self._cik_map: dict[str, str] | None = None

    # ------------------------------------------------------------------ CIK map
    def cik_for(self, ticker: str) -> str | None:
        if self._cik_map is None:
            r = self.http.get("https://www.sec.gov/files/company_tickers.json",
                              cache_ttl=7 * 24 * 3600)
            data = json.loads(r.content)
            self._cik_map = {v["ticker"].upper(): str(v["cik_str"]).zfill(10)
                             for v in data.values()}
        # EDGAR uses '-' where exchanges use '.' (BRK.B -> BRK-B)
        return self._cik_map.get(ticker.upper()) or self._cik_map.get(
            ticker.upper().replace(".", "-"))

    # ------------------------------------------------------- XBRL company facts
    def company_facts(self, ticker: str) -> pd.DataFrame:
        cols = ["ticker", "concept", "tag", "unit", "period_start", "period_end",
                "value", "form", "filed", "fiscal_frame", "source"]
        cik = self.cik_for(ticker)
        if not cik:
            log.warning("no CIK for %s", ticker)
            return pd.DataFrame(columns=cols)
        try:
            r = self.http.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json",
                              cache_ttl=20 * 3600)
            facts = json.loads(r.content)
        except Exception as e:  # noqa: BLE001
            log.warning("companyfacts %s failed: %s", ticker, e)
            return pd.DataFrame(columns=cols)

        rows: list[dict] = []
        gaap = facts.get("facts", {}).get("us-gaap", {})
        dei = facts.get("facts", {}).get("dei", {})
        for concept, tags in CONCEPTS.items():
            ns = dei if concept.endswith("_dei") else gaap
            for tag in tags:
                node = ns.get(tag)
                if not node:
                    continue
                for unit, items in node.get("units", {}).items():
                    for it in items:
                        rows.append({
                            "ticker": ticker, "concept": concept, "tag": tag, "unit": unit,
                            "period_start": it.get("start"), "period_end": it.get("end"),
                            "value": it.get("val"), "form": it.get("form"),
                            "filed": it.get("filed"),        # <- observed_at, exact
                            "fiscal_frame": it.get("frame"), "source": self.name,
                        })
                break  # first tag that exists wins; avoids double counting
        return pd.DataFrame(rows, columns=cols)

    # ----------------------------------------------------------------- filings
    def _submissions(self, cik: str) -> dict:
        r = self.http.get(f"https://data.sec.gov/submissions/CIK{cik}.json", cache_ttl=6 * 3600)
        return json.loads(r.content)

    def recent_filings(self, ticker: str, forms: list[str], limit: int = 40) -> list[Filing]:
        cik = self.cik_for(ticker)
        if not cik:
            return []
        try:
            sub = self._submissions(cik)
        except Exception as e:  # noqa: BLE001
            log.warning("submissions %s failed: %s", ticker, e)
            return []
        recent = sub.get("filings", {}).get("recent", {})
        out: list[Filing] = []
        for i in range(len(recent.get("form", []))):
            form = recent["form"][i]
            if forms and form not in forms:
                continue
            acc = recent["accessionNumber"][i]
            acc_nodash = acc.replace("-", "")
            # Form 4 (and some others) prefix primaryDocument with an XSL renderer
            # path like "xslF345X06/form4.xml" — the RAW document lives at the
            # accession root, so strip any directory prefix.
            doc = recent["primaryDocument"][i].split("/")[-1]
            out.append(Filing(
                ticker=ticker, cik=cik, accession=acc, form=form,
                filed_at=recent["filingDate"][i],
                period=recent.get("reportDate", [None] * 10_000)[i] or None,
                url=f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_nodash}/{doc}",
            ))
            if len(out) >= limit:
                break
        return out

    def filing_text(self, filing: Filing, max_chars: int = 400_000) -> str:
        """Fetch primary document, strip HTML to text, cache to FILINGS dir so the
        Tier-2 LLM session reads committed text (analysis is reproducible)."""
        safe = f"{filing.ticker}_{filing.form.replace('/', '-')}_{filing.filed_at}_{filing.accession.replace('-', '')}.txt"
        path = FILINGS / safe
        if path.exists():
            return path.read_text()[:max_chars]
        r = self.http.get(filing.url, cache_ttl=None)
        text = _html_to_text(r.text)[:max_chars]
        path.write_text(text)
        filing.local_text_path = str(path)
        return text

    # ---------------------------------------------------------------- insiders
    def insider_transactions(self, ticker: str, since: date) -> list[InsiderTx]:
        """Parse Form 4 XML for non-derivative open-market transactions.

        The primary document (prefix-stripped in recent_filings) IS the ownership
        XML — fetch it directly, cached for 60d (filings are immutable), so daily
        runs only ever download new Form 4s. Falls back to the accession index
        if the primary document isn't XML.
        """
        out: list[InsiderTx] = []
        for f in self.recent_filings(ticker, forms=["4"], limit=60):
            if f.filed_at < since.isoformat():
                continue
            acc_dir = f"https://www.sec.gov/Archives/edgar/data/{int(f.cik)}/{f.accession.replace('-', '')}"
            try:
                doc_name = f.url.rsplit("/", 1)[-1]
                if not doc_name.endswith(".xml"):
                    r = self.http.get(acc_dir + "/index.json", cache_ttl=60 * 24 * 3600)
                    items = json.loads(r.content)["directory"]["item"]
                    doc_name = next((i["name"] for i in items
                                     if i["name"].endswith(".xml")), None)
                    if not doc_name:
                        continue
                rx = self.http.get(f"{acc_dir}/{doc_name}", cache_ttl=60 * 24 * 3600)
                out.extend(_parse_form4(rx.content, ticker, f.filed_at))
            except Exception as e:  # noqa: BLE001
                log.warning("form4 %s %s failed: %s", ticker, f.accession, e)
        return out


def _parse_form4(xml_bytes: bytes, ticker: str, filed_at: str) -> list[InsiderTx]:
    from lxml import etree

    try:
        root = etree.fromstring(xml_bytes)
    except Exception:  # noqa: BLE001
        return []
    txt = lambda el, p: (el.findtext(p) or "").strip()  # noqa: E731
    owner = txt(root, ".//reportingOwner/reportingOwnerId/rptOwnerName")
    is_dir = txt(root, ".//reportingOwner/reportingOwnerRelationship/isDirector")
    is_off = txt(root, ".//reportingOwner/reportingOwnerRelationship/isOfficer")
    title = txt(root, ".//reportingOwner/reportingOwnerRelationship/officerTitle")
    role = title or ("Director" if is_dir == "1" else "Officer" if is_off == "1" else "Owner")
    out = []
    for tr in root.findall(".//nonDerivativeTable/nonDerivativeTransaction"):
        code = txt(tr, ".//transactionCoding/transactionCode")
        shares = txt(tr, ".//transactionAmounts/transactionShares/value")
        price = txt(tr, ".//transactionAmounts/transactionPricePerShare/value")
        ad = txt(tr, ".//transactionAmounts/transactionAcquiredDisposedCode/value")
        tdate = txt(tr, ".//transactionDate/value")
        try:
            sh = float(shares) if shares else None
            px = float(price) if price else None
        except ValueError:
            sh, px = None, None
        val = sh * px if sh and px else None
        if ad == "D" and sh:
            sh = -sh
        out.append(InsiderTx(ticker=ticker, filer=owner, role=role, tx_date=tdate,
                             filed_at=filed_at, kind=code, shares=sh, price=px, value=val))
    return out


_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t\r\f\v]+")


def _html_to_text(html: str) -> str:
    html = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    html = re.sub(r"(?i)</(p|div|tr|table|h[1-6]|li|br)[^>]*>", "\n", html)
    text = _TAG_RE.sub(" ", html)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = _WS_RE.sub(" ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()
