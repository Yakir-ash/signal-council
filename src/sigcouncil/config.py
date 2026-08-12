from __future__ import annotations

import functools
from typing import Any

import yaml

from .paths import CONFIG


@functools.lru_cache(maxsize=None)
def load(name: str) -> dict[str, Any]:
    with open(CONFIG / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def universe_cfg() -> dict[str, Any]:
    return load("universe")


def weights_cfg() -> dict[str, Any]:
    return load("weights")


def thresholds_cfg() -> dict[str, Any]:
    return load("thresholds")


def horizons() -> list[dict[str, Any]]:
    return thresholds_cfg()["horizons"]
