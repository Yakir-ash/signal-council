"""Repo-root-aware paths. Everything under data/ splits into:
- committed (ledger, reports, calibration, universe): small, diffable, permanent
- panels/cache/build (gitignored): heavy, regenerable, restored from CI cache
"""
from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    env = os.environ.get("SIGC_ROOT")
    if env:
        return Path(env)
    p = Path(__file__).resolve()
    for parent in p.parents:
        if (parent / "pyproject.toml").exists() and (parent / "config").exists():
            return parent
    return Path.cwd()


ROOT = repo_root()
CONFIG = ROOT / "config"
DATA = ROOT / "data"
UNIVERSE = DATA / "universe"
PANELS = DATA / "panels"          # gitignored: prices.parquet, features.parquet, ...
CACHE = DATA / "cache"            # gitignored: http cache, edgar json
BUILD = DATA / "build"            # gitignored: intermediate artifacts
LEDGER = DATA / "ledger"          # committed: predictions-YYYY-MM.jsonl, outcomes-*.jsonl
REPORTS = DATA / "reports"        # committed: YYYY-MM-DD/pack.json, dashboard.html
CALIBRATION = DATA / "calibration"  # committed: empirical probability tables
FILINGS = DATA / "filings_cache"  # committed (capped): filing text excerpts for LLM tier

for _p in (PANELS, CACHE, BUILD, LEDGER, REPORTS, CALIBRATION, FILINGS):
    _p.mkdir(parents=True, exist_ok=True)
