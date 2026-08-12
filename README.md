# Signal Council

Evidence-first stock research, ranking and prediction system.
**Philosophy:** combine many weak, independent, economically sensible signals into one
calibrated thesis — and stay silent when the evidence isn't there.

> `NO HIGH-CONVICTION OPPORTUNITIES TODAY` is a first-class output, not a failure.

Full design rationale, methodology, red-team analysis: **[docs/DESIGN.md](docs/DESIGN.md)**.

## Architecture in one breath

Two tiers. **Tier 1 (GitHub Actions, scheduled):** fetch prices (yfinance + Stooq
cross-validation), SEC EDGAR XBRL fundamentals/filings/Form-4 insiders, FRED macro →
quality gate → point-in-time features → regime → rule-composite scoring → risk /
divergence / confidence → append to the immutable prediction ledger → evaluate matured
predictions → commit the daily pack + dashboard. **Tier 2 (scheduled Claude session):**
reads committed filing texts, adds cited, schema-validated qualitative analysis
(AI INTERPRETATION, bounded influence), delivers the dashboard.

## Integrity invariants (enforced in code + CI)

1. **Point-in-time**: features only use data with `observed_at <= prediction time`
   (`tests/test_pit.py` plants leaks; CI fails if they pass).
2. **Append-only ledger**: `scripts/check_ledger_immutable.py` fails any commit that
   edits a historical prediction line.
3. **No invented probabilities**: probabilities come from empirical walk-forward tables
   or are hard-clamped cold-start estimates, labeled as such.
4. **No fallback recommendations**: gates have no "best available" path.
5. **Every displayed datum** is tagged FACT / MODEL ESTIMATE / AI INTERPRETATION
   with source + timestamp.

## CLI

```bash
pip install -e .
sigc daily              # full pipeline (network needed)
sigc analyze NVDA       # deep on-demand pack
sigc compare NVDA AMD
sigc backtest --start 2016-01-01 --top-n 20
sigc calibration        # how honest are our probabilities?
sigc render             # rebuild dashboard.html
pytest tests/ -q        # integrity tests
```

## Not investment advice

This is a research tool. Its outputs are model estimates with documented
uncertainty, built partly on free data with documented gaps. Nothing here is a
recommendation to buy or sell securities.
