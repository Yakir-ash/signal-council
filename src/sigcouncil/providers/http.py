"""Shared HTTP plumbing: retries with backoff, polite rate limiting, on-disk cache.

Every provider goes through this so retry/caching/logging behavior is uniform
and testable, and so a single misbehaving endpoint can't hammer anyone.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import requests

from ..logutil import get_logger
from ..paths import CACHE

log = get_logger("http")

DEFAULT_HEADERS = {
    # SEC requires a descriptive User-Agent with contact info; harmless elsewhere.
    "User-Agent": "SignalCouncil research (contact: yakir10101@gmail.com)",
    "Accept-Encoding": "gzip, deflate",
}


class Http:
    def __init__(self, min_interval: float = 0.12, timeout: int = 30, retries: int = 3):
        self.sess = requests.Session()
        self.sess.headers.update(DEFAULT_HEADERS)
        self.min_interval = min_interval
        self.timeout = timeout
        self.retries = retries
        self._last = 0.0

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.monotonic()

    def get(self, url: str, params: dict | None = None, cache_ttl: int | None = None) -> requests.Response:
        cache_path = None
        if cache_ttl:
            key = hashlib.sha1((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()
            cache_path = Path(CACHE) / "http" / f"{key}.bin"
            if cache_path.exists() and (time.time() - cache_path.stat().st_mtime) < cache_ttl:
                resp = requests.Response()
                resp.status_code = 200
                resp._content = cache_path.read_bytes()
                resp.url = url
                return resp
        last_exc: Exception | None = None
        for attempt in range(self.retries):
            try:
                self._throttle()
                r = self.sess.get(url, params=params, timeout=self.timeout)
                if r.status_code in (429, 500, 502, 503, 504):
                    raise requests.HTTPError(f"status {r.status_code}", response=r)
                if 400 <= r.status_code < 500:
                    # permanent client error: retrying a 404 wastes seconds per call
                    # and never helps — fail fast, caller degrades gracefully
                    raise _PermanentHTTPError(f"GET {url} -> {r.status_code}")
                r.raise_for_status()
                if cache_path:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    cache_path.write_bytes(r.content)
                return r
            except _PermanentHTTPError:
                raise
            except Exception as e:  # noqa: BLE001 — provider layer converts to warnings upstream
                last_exc = e
                sleep = 2.0**attempt
                log.warning("GET %s failed (%s), retry in %.1fs", url, type(e).__name__, sleep)
                time.sleep(sleep)
        raise RuntimeError(f"GET {url} failed after {self.retries} attempts") from last_exc


class _PermanentHTTPError(RuntimeError):
    pass
