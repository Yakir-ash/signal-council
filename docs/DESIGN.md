# Signal Council — System Design Document

**An evidence-first stock research, ranking, and prediction system.**
Version 1.0 — 2026-08-12 — Author: Claude (with Yakir)

> Philosophy: combine many weak, independent, economically sensible signals into one
> calibrated thesis. Prefer silence over manufactured conviction. Prediction integrity
> beats feature count, every time.

---

## 0. Constraints discovered before designing anything

Before proposing an architecture I probed the actual execution environment. Findings that shape everything below:

1. **The Cowork cloud sandbox cannot reach financial APIs.** Yahoo Finance, Stooq, SEC EDGAR, and FRED are all blocked by the sandbox's network allowlist (verified empirically: proxy 403s). GitHub (git, REST API, raw content) and package registries **are** reachable.
2. **The sandbox is ephemeral.** Code and the prediction ledger need a permanent home (decided: GitHub).
3. **Chosen configuration** (Yakir's decisions): GitHub repo + scheduled cloud runs; free-tier data providers to start; S&P 500 + major ETFs universe.

Consequence: a **two-tier execution architecture**. The quantitative pipeline cannot run where Claude runs, and Claude's analysis cannot run inside a dumb cron job. Each tier does what it is uniquely good at.

---

## 1. Proposed architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  TIER 1 — THE MUSCLE: GitHub Actions (scheduled, unrestricted net)   │
│                                                                      │
│  daily cron (post-close ET)                                          │
│   1. fetch: prices (yfinance→Stooq fallback), EDGAR filings+XBRL,    │
│      FRED macro, news RSS                                            │
│   2. data-quality gate (freshness, splits, outliers, cross-source)   │
│   3. feature engine (technical, fundamental, event features)         │
│   4. market regime classifier                                        │
│   5. prediction models (rule composite + GBM ensemble, 4 horizons)   │
│   6. opportunity / risk / divergence / data-confidence scoring       │
│   7. watchlist state machine update                                  │
│   8. append predictions to immutable ledger (JSONL, git history)     │
│   9. evaluate matured predictions, update calibration                │
│  10. emit machine-readable daily pack (JSON) + raw filing texts      │
│  11. commit everything to the repo                                   │
└───────────────────────────┬──────────────────────────────────────────┘
                            │  git (the only interface between tiers)
┌───────────────────────────▼──────────────────────────────────────────┐
│  TIER 2 — THE ANALYST: scheduled Claude (Cowork) session             │
│                                                                      │
│   1. pull the day's pack + filing/news texts from the repo           │
│   2. LLM analysis with STRICT JSON contracts:                        │
│      filing diffs, transcript tone, catalyst extraction,             │
│      thesis-breaker articulation — every claim must quote source     │
│   3. merge as a clearly-tagged qualitative overlay                   │
│      (AI INTERPRETATION never overwrites a quant number)             │
│   4. render final dashboard (self-contained HTML)                    │
│   5. commit overlay + dashboard back to repo                         │
│   6. deliver the dashboard to Yakir (chat + persistent artifact)     │
└──────────────────────────────────────────────────────────────────────┘

On-demand ("Analyze NVDA"): Claude triggers a workflow_dispatch for the
ticker → Actions builds a fresh data pack (~2-3 min) → Claude runs the
full analysis pipeline on it and responds. Same code path as daily runs.
```

**Repo layout (monorepo, Python):**

```
signal-council/
├── src/sigcouncil/
│   ├── providers/        # abstract interfaces + implementations (swappable)
│   │   ├── base.py       #   PriceProvider, FundamentalsProvider, FilingsProvider,
│   │   │                 #   MacroProvider, NewsProvider (ABCs)
│   │   ├── yf_prices.py  #   yfinance (primary prices)
│   │   ├── stooq.py      #   Stooq CSV (fallback prices, cross-validation)
│   │   ├── edgar.py      #   SEC EDGAR: filings index, full text, XBRL company facts
│   │   ├── fred.py       #   FRED macro series (no key needed via fredgraph.csv)
│   │   └── rss_news.py   #   free news RSS feeds
│   ├── store/            # Parquet OHLCV panels + SQLite build artifact + PIT rules
│   ├── quality/          # data-quality checks → Data Confidence Score
│   ├── features/         # technical / fundamental / event feature computation
│   ├── regime/           # market regime classifier
│   ├── models/           # rule composite, GBM ranker, calibration, ensemble
│   ├── scoring/          # Opportunity, Risk, Divergence, final assembly + gates
│   ├── ledger/           # append-only prediction ledger + evaluation
│   ├── backtest/         # walk-forward engine + metrics + anti-lookahead guards
│   ├── watchlist/        # state machine (BUY CANDIDATE … AVOID)
│   ├── report/           # daily JSON pack + HTML dashboard renderer
│   └── pipeline/         # orchestration, logging, failure handling
├── analysis/             # Tier-2 LLM contracts: prompts + JSON schemas + merge rules
├── config/               # universe.yaml, weights.yaml, thresholds.yaml, regimes.yaml
├── data/                 # committed: ledger/, reports/, calibration/, filings_cache/
├── .github/workflows/    # daily.yml, ondemand.yml, backtest.yml, ci.yml
└── tests/                # unit + integrity tests (lookahead guards are tested)
```

Everything the intelligence engine consumes goes through the provider ABCs. Swapping yfinance for Polygon later = one new file + one config line, zero changes to features/models/scoring.

---

## 2. Recommended technology stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11 | the entire quant/data ecosystem |
| Data frames | pandas + numpy | universal; universe of ~530 names doesn't need Spark |
| Storage | Parquet (price panels, features) + JSONL (ledger, committed) + SQLite (local query cache, rebuilt, not committed) | git-friendly, diffable, immutable-by-history |
| ML | scikit-learn + LightGBM | GBMs are the workhorse of cross-sectional equity ranking; sklearn for isotonic calibration |
| Scheduling (quant) | GitHub Actions cron | free, reliable, unrestricted network, logs kept |
| Scheduling (analysis) | Cowork scheduled task | Claude does the LLM work natively — no separate LLM API key or cost |
| Dashboard | self-contained static HTML (vanilla JS + inline data) | zero hosting, renders in Cowork artifact gallery + any browser, survives forever |
| Config | YAML, versioned in git | weights/thresholds changes are code-reviewed diffs, not silent edits |
| Secrets | GitHub Actions secrets | keys never touch the repo or frontend |

Explicitly rejected: a hosted web app + database server (cost/ops burden with zero benefit at this scale); LLM-as-price-predictor (banned by design, see §20); real-time streaming (this is a daily system; pretending otherwise adds noise).

---

## 3. Data providers — free vs paid

### Free (v1, chosen)

| Provider | Data | Reliability notes |
|---|---|---|
| **SEC EDGAR** (free, official, excellent) | 10-K/10-Q/8-K/Form 4/S-1 full text; **XBRL company facts** = structured fundamentals with exact filing timestamps | The best free source in the entire stack. Point-in-time by construction (each fact carries its filing date). Rate limit 10 req/s with User-Agent. |
| **yfinance** (Yahoo, unofficial) | OHLCV, some estimates/targets, profile, holders | Unofficial API; can break or rate-limit. Treated as replaceable; cross-validated against Stooq. Estimate data marked LOW CONFIDENCE. |
| **Stooq** | daily OHLCV CSV | independent second price source → cross-validation + fallback |
| **FRED** | rates, yield curve, credit spreads, inflation expectations, financial conditions | official, free, no key needed for CSV endpoint |
| **News RSS** (Yahoo Finance per-ticker RSS, GlobeNewswire/PRNewswire feeds, SEC 8-K stream) | headlines + timestamps | thin vs paid news APIs; catalysts mostly detected via 8-Ks (which are legally mandated and therefore *more* reliable than news) |

### What free data honestly cannot give us (v1 gaps, stated up front)

- **Analyst estimate revisions history** — the single most valuable missing signal. Partially proxied by: earnings surprise + XBRL fundamental acceleration + guidance language from filings. Yahoo's current-snapshot estimates are used as weak/low-confidence input only.
- **Earnings call transcripts** — not freely licensable in bulk. v1 substitutes the MD&A / risk-factor / results sections of 10-Q/10-K and 8-K earnings releases (which contain guidance language). Paid tier (FMP ~$30/mo) adds transcripts later behind the existing `FilingsProvider` interface.
- **Options flow** — no free reliable source; sentiment engine runs without it in v1.
- **Delisted-stock price history** — the survivorship problem; see §9 for how we handle it honestly.

### Paid upgrade path (designed-in, not built)

Financial Modeling Prep Starter (~$30/mo): estimates + revisions, transcripts, better fundamentals history. Polygon ($30–200/mo): institutional price data + options. Each maps to an existing provider interface.

---

## 4. Database schema (logical)

Point-in-time discipline is enforced at the schema level: **every row carries `as_of` (when the fact was true) and `observed_at` (when we could first have known it)**. Features may only read rows where `observed_at <= prediction_time`. This single rule kills most look-ahead bugs, and it is unit-tested.

```
prices(ticker, date, open, high, low, close, adj_close, volume,
       source, ingested_at)                          # Parquet panel
corporate_actions(ticker, date, kind, ratio, source, observed_at)
fundamentals(ticker, concept, period_end, value, unit, fiscal_frame,
             form, filed_at)                         # filed_at == observed_at (EDGAR gives us this exactly)
filings(ticker, accession, form, filed_at, period, url, local_text_path)
insider_tx(ticker, filer, role, tx_date, filed_at, kind, shares, price, value)
macro(series, date, value, observed_at)
news(id, ticker, published_at, source, title, url, ingested_at)
features(date, ticker, feature, value, version)      # Parquet, regenerable
regime(date, label, subscores_json, version)
scores(date, ticker, opportunity, risk, divergence, data_confidence,
       components_json, model_version)
predictions(ledger — see §18 schema below; JSONL, append-only)
outcomes(prediction_id, evaluated_at, realized_return, benchmark_return,
         outcome_binary, notes)
watchlist(date, ticker, state, prev_state, reason)
quality_log(date, check, target, status, detail)
```

The SQLite file is a **build artifact** compiled from the committed Parquet/JSONL — rebuildable from scratch, never the source of truth, never committed.

---

## 5. Prediction methodology

### The frame: cross-sectional ranking, not market prophecy

Predicting absolute market direction is mostly noise. The tractable question is: **which names are likely to do unusually well or badly relative to the universe, and is the environment friendly to that bet?** So:

- Stock-level models predict **relative** forward returns vs the universe/benchmark.
- The **regime engine** (§8 of your spec) supplies the market-level tilt: it scales conviction and exposure, and re-weights which signals count.
- Absolute-return predictions shown to you = relative prediction + regime-conditional benchmark distribution, always as a **range**, never a point.

### Horizons

5d (~1w), 21d (~1m), 63d (~3m), 126–252d (~6–12m) trading days. Each horizon has its own model and its own ledger entries. Signals decay differently: earnings drift works at 21–63d; momentum at 63–252d; volatility-contraction breakouts at 5–21d; valuation/quality only at 126d+.

### Models (ensemble of genuinely different kinds)

1. **Rule composite (v1 backbone, always on).** A transparent weighted sum of subscores built only from signals with decades of published, economically-motivated evidence:
   - momentum 12-1 (skip last month), sector-relative
   - post-earnings-announcement drift (surprise + XBRL fundamental acceleration)
   - profitability/quality (gross-profit/assets, FCF margin trend, accrual quality)
   - value-vs-quality within sector (never raw "cheap")
   - insider-buying clusters (multiple insiders, open-market, meaningful size)
   - low-vol/quality adjustment; volatility-contraction timing overlay
2. **GBM ranker (v2, after backtest data exists).** LightGBM per horizon on the full point-in-time feature set, walk-forward retrained quarterly, never trained on data it will be evaluated on.
3. **Ensemble.** Rank-average; **disagreement between models lowers confidence** rather than being averaged away — a stock the rules love and the GBM hates is flagged, not smoothed.

### Probabilities and distributions

Probabilities come from **historical empirical distributions, never from vibes**: score-decile × regime × horizon buckets are mapped to their realized forward-return distributions in the walk-forward backtest. From that conditional distribution we report P(positive), P(beat SPY), expected range (25th–75th pct), downside (5th pct), expected vol. Once the live ledger matures, an **isotonic recalibration layer** trained only on out-of-sample matured predictions corrects any bias. Until enough live data exists, confidence is explicitly capped (no ">75%" claims from a cold-start model) — the system is forbidden from displaying uncalibrated high confidence.

---

## 6. Feature set (v1, ~90 features; every one economically motivated)

**Price/technical** (per §7): returns over {5,21,63,126,252}d; 12-1 momentum; MA(20/50/200) positions and crosses; RSI(14); MACD; ATR%; realized vol 21/63d; vol-of-vol; volatility contraction ratio; relative volume 5/21d; distance from 52w high/low; drawdown; gap frequency/size; support/resistance proximity (swing levels); RS vs SPY and vs sector ETF (21/63d); beta; Amihud illiquidity; dollar volume.

**Fundamental (XBRL, point-in-time)**: revenue growth YoY/QoQ-annualized and its **acceleration**; gross/operating/FCF margins and trends; EPS growth; accruals (NI–OCF gap); debt/EBITDA; interest coverage; cash ratio; share-count trend (dilution/buyback); ROIC, ROE, gross-profit/assets; capex trend; inventory-vs-revenue divergence.

**Valuation**: EV/S, EV/EBITDA, P/E, FCF yield — each as (a) sector-relative z-score and (b) percentile vs the name's own 3y history. Never absolute "cheap".

**Event/flow**: days to/from earnings; last surprise; announcement-day return and drift; insider net buying 90d (cluster-weighted); 8-K event flags; institutional 13F change (quarterly, lagged as observed).

**Sentiment/attention (v1: thin, honest)**: news volume anomaly vs trailing baseline; headline polarity (lexicon-based, cheap and stable); RS-vs-news-flow divergence. Social sentiment deferred until a reliable free source exists — a bad sentiment feed is worse than none.

**Macro/regime**: regime label + subscores joined to every row.

---

## 7. Scoring methodology

### Opportunity Score (0–100, fully decomposable)

Weighted composite of component scores, each 0–100, each with its own evidence trail:

| Component | v1 weight | Rationale for weight |
|---|---|---|
| Earnings/Fundamental Momentum (PEAD, acceleration) | 20 | strongest documented anomaly reachable with free data |
| Price Momentum & Trend | 18 | most robust cross-sectional factor in the literature |
| Quality/Profitability | 14 | strong, slow, stabilizing |
| Divergence/Mispricing | 14 | the system's specialty; see below |
| Valuation (context-adjusted) | 10 | weak alone at short horizons; matters at 6–12m |
| Catalyst Strength (8-K/filing/news, LLM-extracted) | 8 | high value but noisy → modest weight, confidence-scaled |
| Insider/Institutional Signal | 6 | real but sparse |
| Technical Confirmation (timing overlay) | 6 | timing, not thesis |
| Regime Compatibility | 4 (plus gating role) | regime mostly acts as a gate/scaler, not points |

Weights live in `config/weights.yaml`, are validated (not discovered) by walk-forward backtest, and changing them requires a versioned commit — the ledger records which weight-version made every prediction. **These weights are a prior from published factor research, and are labeled as such until our own out-of-sample evidence accumulates.**

**Gates (hard, before any score is shown):** liquidity gate (min $10M median daily dollar volume, price ≥ $5, listed ≥ 1y); data-confidence gate (score < 60 → ineligible for recommendation, shown only with a warning); risk cap (Risk Score > 80 → cannot be a Top Opportunity regardless of upside).

**High-conviction threshold:** Opportunity ≥ 72 AND calibrated confidence ≥ floor AND passes all gates. At most 5 shown daily. **Zero passing → the dashboard prints `NO HIGH-CONVICTION OPPORTUNITIES TODAY` — a first-class, expected, honest output.** There is no code path that back-fills "best available" names.

### Risk Score (0–100)

Composite of realized vol percentile, Amihud illiquidity, leverage (debt/EBITDA, coverage), earnings proximity (event risk), valuation extremeness (blow-up asymmetry), historical max drawdown & gap behavior, macro/rate sensitivity, and concentration flags from filings. Every Top Opportunity ships with its risk decomposition and **Thesis Breakers** — concrete falsifiers ("gross margin < X% next quarter", "loses relative strength vs sector for 20 sessions", "CFO departs") generated per-name and tracked: a triggered breaker forces a watchlist state change, visibly.

### Divergence / Mispricing Score (0–100, dedicated)

Measures *disagreement between evidence streams*, the setups you specifically asked for: fundamental acceleration vs price stagnation; estimates/guidance direction vs sentiment; insider buying into drawdown + pessimism; margin expansion vs compressed sector-relative valuation; post-earnings positive surprise + initial selloff; RS strength despite weak sector. Each pattern is a rule with a defined trigger; the score is a capped sum, and the triggered patterns are listed by name in the explanation.

### Data Confidence Score (0–100)

Freshness, cross-source price agreement, fundamental completeness, filing recency, corporate-action consistency. Shown on every card; low confidence blocks recommendations rather than silently degrading them.

### FACT / MODEL ESTIMATE / AI INTERPRETATION

Every displayed datum carries one of these three tags plus source + timestamp. Enforced by the report schema itself (the renderer refuses untagged values). Prices/fundamentals = FACT(source,ts). Probabilities/scores = MODEL ESTIMATE(version). Filing-tone/catalyst readings = AI INTERPRETATION(with quoted source text). LLM outputs can never populate FACT fields.

---

## 8. Backtesting methodology (§17) — and its honest limits

**Walk-forward, point-in-time, cost-aware:**

- Expanding-window walk-forward: train/fit on data through T, predict T+1 cohort, roll. GBM retrained quarterly inside the loop; rule-composite weights held fixed within each validation fold.
- Features computed **only** from rows with `observed_at <= T` (the PIT rule, unit-tested with synthetic traps — e.g., a fundamental fact filed late must not leak into earlier features; the test fails the build if it does).
- Transaction costs: 10 bps/side + spread estimate; slippage stress at 2×. Signals that die at 2× costs are reported as fragile.
- Benchmarks: SPY total return, equal-weight universe, and random-portfolio Monte Carlo (200 draws) — a strategy must beat *random selection from its own universe*, not just the index.
- Metrics: CAGR, total return, Sharpe, Sortino, max DD, vol, hit rate, avg win/loss, profit factor, turnover, **calibration (Brier + reliability curves)**, and performance sliced by regime × score-decile × horizon.

**Honest limits (free data):**

1. **Survivorship bias.** We use historical S&P 500 constituent lists (public datasets) so the backtest universe at time T is the *actual* index membership at T — but free price sources lack full delisted-name history. Names that left the index and delisted with no data are logged, counted, and the backtest reports a **bias bound**: results shown as a range [pessimistic: missing names assumed at -50% ... optimistic: as-is], not a single flattering number.
2. **Backtest depth**: ~15 years of dailies is feasible; XBRL fundamentals are reliable from ~2010+. Earlier periods excluded rather than filled with lower-quality data.
3. Any strategy result is labeled **INDICATIVE** until it also survives the live ledger. The live, out-of-sample prediction ledger — not the backtest — is the system's real report card, and the UI says so.

---

## 9. Daily automation architecture (§25)

`daily.yml` (GitHub Actions, cron ~22:30 UTC = after US close + data settle):
steps 1–12 exactly as your spec, each step logs to `quality_log`, each step can fail independently: a failed news fetch degrades (with a visible warning on the dashboard) rather than killing the run; a failed **price** fetch aborts scoring (garbage-in guard) and the dashboard for that day says so explicitly. Retries with exponential backoff at the provider layer; response caching (ETag/If-Modified-Since for EDGAR). A `RUN ANALYSIS NOW` = `workflow_dispatch` trigger — you can press it in GitHub's UI, or ask me in chat and I trigger it via API.

Tier-2 Cowork scheduled task fires after the Actions run (with a repo-freshness check + retry window), does the LLM overlay + dashboard delivery. If Tier 1 hasn't landed, Tier 2 reports *that* instead of analyzing stale data.

## 10. UI architecture (§24)

One self-contained `dashboard.html` per day (plus `latest`), committed and delivered as a persistent Cowork artifact — seven tabs matching your spec: **Today** (regime brief → top opportunities → avoid list, cards exactly per your §13 layout), **Stock Analyzer** (per-ticker deep packs; on-demand via chat), **Watchlist** (state machine + state-change history), **Prediction Ledger** (every prediction ever, filterable, with outcomes), **Performance** (30/90/365d/all-time accuracy, calibration plots, vs benchmarks), **Backtest Lab** (results of committed experiments with their configs), **Market Intelligence** (regime dial, sector RS matrix, macro panel, upcoming earnings/events). Professional-terminal aesthetic, plain-language everywhere, every number tagged FACT/ESTIMATE/INTERPRETATION with source+timestamp on hover.

## 11. Expected operating costs

| Item | Cost |
|---|---|
| Data (v1) | $0 |
| GitHub private repo + Actions | $0 (daily run ~10–20 min ⇒ ~300–600 min/mo, within the 2,000 free) |
| Tier-2 Claude runs | included in your Claude subscription usage |
| **Total v1** | **$0/month** |
| Optional later: FMP Starter | ~$30/mo (adds estimates/transcripts — biggest single upgrade) |

## 12. Limitations & failure modes (declared before writing code)

Free-data gaps (§3); survivorship bounds (§8); yfinance fragility (mitigated: Stooq cross-check, provider abstraction, quality gate); cold-start calibration (mitigated: confidence caps until evidence accumulates); small live-sample overinterpretation (mitigated: dashboards show N and confidence intervals on accuracy stats, and refuse to celebrate N<30); regime classifier lag (regimes are identified with delay; treated as a slow tilt, not a timing signal); LLM failure modes (hallucination/overconfidence — mitigated: JSON contracts, mandatory source quotes, no numeric authority, tagged output); **alpha decay** — published anomalies are partly arbitraged; expected edge is modest, and the system's honesty about that is a feature.

---

## 13. Red team: how this system could produce impressive-looking but useless predictions

*(the self-challenge you asked for — each attack has a design answer)*

**Attack 1 — "The backtest is a survivorship story."** Backtesting today's S&P 500 members over the past decade bakes in that they survived; +2–3%/yr of fake alpha is typical. → *Fix: historical constituents, missing-name accounting, results as bias-bounded ranges (§8). Never one flattering number.*

**Attack 2 — "You tuned 30 weights on one history."** Iterating weights until the equity curve looks great = curve-fitting; out-of-sample it's noise. → *Fix: weights are priors from literature, validated not optimized; coarse grid only; any tuning happens inside walk-forward folds; weight changes are versioned commits; the ledger records which version made each call. And the live ledger is the arbiter of record.*

**Attack 3 — "Confidence percentages are decorations."** LLMs (and quants) love saying "78% confident" with no basis. → *Fix: probabilities only from empirical conditional distributions; cold-start caps; Brier/reliability tracked publicly on the Performance tab; the renderer literally has no field for a probability without a distribution source.*

**Attack 4 — "Look-ahead leaks in quietly."** Fundamentals joined by period-end instead of filing date; adjusted prices leaking future splits; features 'as of today' using today's close to predict today. → *Fix: `observed_at` discipline in the schema, EDGAR's filed_at used as ground truth, prediction timestamps at next-day-open, and adversarial unit tests that plant leaks and require the build to fail.*

**Attack 5 — "The ledger gets quietly rewritten."** The strongest temptation in any prediction system. → *Fix: append-only JSONL in git — history is public within the repo; the evaluation job writes to a separate outcomes file; CI **fails if any commit modifies an existing ledger line** (tested by a dedicated guard script).*

**Attack 6 — "It recommends something every day because that feels valuable."** Silent pressure toward output inflation. → *Fix: hard threshold + gates with no fallback path; the empty state is designed, first-class UI; and the Performance tab tracks 'days with zero recommendations' as a health metric, not a failure.*

**Attack 7 — "One data source lies and the model believes it."** A bad Yahoo tick or missed split creates a fake -40% 'opportunity'. → *Fix: two independent price sources cross-validated daily; corporate-action checks; outlier quarantine (a >25% move without a corroborating filing/news item freezes the name pending review rather than scoring it).*

**Attack 8 — "The LLM invents a catalyst."** → *Fix: catalyst extraction must cite the committed source document + quote; un-cited claims are dropped in the merge step; LLM output feeds only Catalyst/Interpretation components (≤8 weight points + confidence scaling), never prices, fundamentals, or probabilities.*

**Attack 9 — "Regime overfit: seven regimes fit on one bull market."** → *Fix: regime rules are simple, few, and macro-economically standard (trend/vol/credit/curve); regime is a conviction scaler and signal re-weighter, not an oracle; regime-conditional performance is reported with its (small) N.*

**Attack 10 — "It works until it doesn't, and nobody notices."** Signal decay is certain, not possible. → *Fix: per-signal live IC tracking with drift alarms (§21); a signal whose rolling live contribution goes negative gets flagged on the dashboard and down-weighted only through a versioned, logged change.*

---

## 14. Development phases

- **Phase 0 (now):** repo scaffold, provider layer, PIT store, quality gate. *Exit: real data flows for full universe, cross-validated.*
- **Phase 1:** features + regime + rule composite + risk/divergence/confidence scoring + ledger + daily pipeline + dashboard + GitHub Actions + Tier-2 scheduled task. *Exit: honest daily runs land automatically; predictions accumulate.*
- **Phase 2:** walk-forward backtest framework + historical constituents + empirical probability tables + calibration machinery. *Exit: every probability shown traces to a conditional distribution; weights validated.*
- **Phase 3:** LLM overlay contracts (filing diffs, catalysts, thesis breakers) in the Tier-2 run; watchlist state machine maturity; avoid-list.
- **Phase 4:** GBM ranker + ensemble + feature-importance/decay tracking.
- **Phase 5:** alerts (new high-conviction entry, thesis-breaker triggered, score jumps), paid-data upgrades if desired, universe expansion (Israeli/European stocks, crypto) via config.

Ordering rationale: the ledger exists from Phase 1 — every day of operation grows the out-of-sample evidence base that everything later (calibration, ML, trust) depends on. The ML model arrives last, not first: it needs the PIT feature store and walk-forward harness to be trained *honestly*, and the rule composite gives an interpretable, literature-grounded baseline it must beat to earn its place.
