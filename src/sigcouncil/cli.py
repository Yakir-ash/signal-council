"""Signal Council CLI.

  sigc daily [--skip-fetch]        full daily pipeline
  sigc fetch [--tickers A,B]       data acquisition only
  sigc analyze NVDA [--no-refresh] deep on-demand analysis pack
  sigc compare NVDA AMD            side-by-side packs
  sigc backtest [--start 2016-01-01] [--top-n 20]
  sigc evaluate                    evaluate matured predictions only
  sigc calibration                 print calibration report
  sigc render                      rebuild dashboard.html from latest pack
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    ap = argparse.ArgumentParser(prog="sigc")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("daily")
    p.add_argument("--skip-fetch", action="store_true")

    p = sub.add_parser("fetch")
    p.add_argument("--tickers", default=None)
    p.add_argument("--years", type=int, default=4)

    p = sub.add_parser("analyze")
    p.add_argument("ticker")
    p.add_argument("--no-refresh", action="store_true")

    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")

    p = sub.add_parser("backtest")
    p.add_argument("--start", default="2016-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--top-n", type=int, default=20)
    p.add_argument("--rebalance-days", type=int, default=21)
    p.add_argument("--cost-bps", type=float, default=10.0)
    p.add_argument("--no-fundamentals", action="store_true")

    sub.add_parser("evaluate")
    sub.add_parser("calibration")
    sub.add_parser("render")

    a = ap.parse_args()

    if a.cmd == "daily":
        from .pipeline.daily import run_daily
        pack = run_daily(skip_fetch=a.skip_fetch)
        _summary(pack)
    elif a.cmd == "fetch":
        from .logutil import RunLog
        from .pipeline.fetch import fetch_all
        tickers = a.tickers.split(",") if a.tickers else None
        fetch_all(RunLog("fetch-manual"), years=a.years, tickers=tickers)
    elif a.cmd == "analyze":
        from .pipeline.analyze import analyze
        pack = analyze(a.ticker, refresh=not a.no_refresh)
        print(json.dumps(pack, indent=1, default=str))
    elif a.cmd == "compare":
        from .pipeline.analyze import analyze
        pa = analyze(a.a, refresh=True)
        pb = analyze(a.b, refresh=False)   # panels already refreshed
        print(json.dumps({"comparison": [pa, pb]}, indent=1, default=str))
    elif a.cmd == "backtest":
        from .backtest.engine import BacktestConfig, run
        res = run(BacktestConfig(start=a.start, end=a.end, top_n=a.top_n,
                                 rebalance_days=a.rebalance_days,
                                 cost_bps_per_side=a.cost_bps,
                                 use_fundamentals=not a.no_fundamentals))
        print(json.dumps({"portfolio": res.portfolio, "benchmark": res.benchmark,
                          "equal_weight": res.equal_weight,
                          "random_beat_frac": res.random_beat_frac,
                          "bucket_table": res.bucket_table,
                          "by_regime": res.by_regime}, indent=1))
    elif a.cmd == "evaluate":
        from .ledger.evaluate import evaluate_matured
        from .store import panels
        n = evaluate_matured(panels.primary_prices())
        print(f"evaluated {n} matured predictions")
    elif a.cmd == "calibration":
        from .ledger.evaluate import calibration_report
        print(json.dumps(calibration_report(), indent=1))
    elif a.cmd == "render":
        from .report.html import render_latest
        print(render_latest())
    return 0


def _summary(pack: dict) -> None:
    print(f"\n=== {pack['date']} | regime: {pack['regime']['label'] if pack['regime'] else '?'} ===")
    if pack.get("aborted"):
        print("RUN ABORTED — see quality reasons:", pack["quality"]["reasons"])
        return
    if pack["no_high_conviction"]:
        print("NO HIGH-CONVICTION OPPORTUNITIES TODAY")
    for t in pack["top_opportunities"]:
        print(f"  {t['ticker']:6s} opp {t['opportunity']:5.1f}  risk {t['risk']:5.1f}  "
              f"conf {t['confidence']:.0%}  ${t['px_last']:.2f}")
    if pack["avoid_list"]:
        print("AVOID:", ", ".join(x["ticker"] for x in pack["avoid_list"]))


if __name__ == "__main__":
    sys.exit(main())
