# Tier-2 LLM analysis contracts

This directory defines how the scheduled Claude (Cowork) session adds qualitative
analysis on top of the quantitative pipeline. The rules exist to keep LLM output
useful and *contained* (DESIGN.md red-team Attack 8):

## Hard rules

1. **LLM output is always AI INTERPRETATION.** It can never populate a FACT field
   or override a quantitative number (prices, fundamentals, probabilities, scores
   other than `catalyst`).
2. **Every claim must cite committed source text** — the filing excerpt in
   `data/filings_cache/` or a stored headline — with a short quote. Claims without
   citations are dropped during merge.
3. **Structured output only.** Each task has a JSON schema in `schemas/`. Output
   that fails validation is rejected and retried once, then skipped (logged).
4. **The catalyst component is bounded**: 8/100 weight in the Opportunity Score,
   and it moves scores only through the documented merge rule below.

## Tasks (prompts in prompts/, schemas in schemas/)

| Task | Input | Output |
|---|---|---|
| `filing_diff` | current vs previous 10-K/10-Q text for a ticker | new/removed risks, tone shifts, margin/demand/liquidity commentary, accounting changes — each with quotes |
| `catalyst_scan` | recent 8-Ks + headlines for top-ranked tickers | catalysts with direction, magnitude (S/M/L), confidence, expected duration, priced-in assessment |
| `earnings_language` | latest results-related filing sections | guidance direction, management tone, newly introduced concerns |
| `daily_brief` | pack.json + macro inputs | the MARKET TODAY narrative (clearly AI-tagged) |

## Merge rule (applied by the Tier-2 session, committed as overlay.json)

- `catalyst` component per ticker = 50 + 10·Σ(direction × magnitude × confidence),
  clipped to [20, 80]. magnitude S=0.5 M=1 L=2; confidence in [0,1].
- Overlay lives in `data/reports/<date>/overlay.json`; the dashboard renders it
  as AI INTERPRETATION cards with quotes. The ledger entry for that day already
  exists and is NOT rewritten — overlays are additive context, recorded with
  their own timestamp.
