from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

from .paths import BUILD

_LOG = logging.getLogger("sigcouncil")
if not _LOG.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    _LOG.addHandler(h)
    _LOG.setLevel(logging.INFO)


def get_logger(name: str) -> logging.Logger:
    return _LOG.getChild(name)


class RunLog:
    """Structured step log for pipeline runs; written to build dir and summarized
    into the daily pack so failures/anomalies are visible on the dashboard."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.steps: list[dict] = []
        self.path = Path(BUILD) / f"runlog-{run_id}.jsonl"

    def step(self, name: str, status: str, detail: str = "", **kw) -> None:
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "run_id": self.run_id, "step": name, "status": status, "detail": detail, **kw}
        self.steps.append(rec)
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        lg = get_logger("run")
        (lg.warning if status in ("warn", "fail") else lg.info)("%s: %s %s", name, status, detail)

    def summary(self) -> dict:
        return {
            "run_id": self.run_id,
            "steps": self.steps,
            "failures": [s for s in self.steps if s["status"] == "fail"],
            "warnings": [s for s in self.steps if s["status"] == "warn"],
        }
