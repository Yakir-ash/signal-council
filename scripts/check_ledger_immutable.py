#!/usr/bin/env python3
"""CI guard (red-team Attack 5): FAIL if any commit in the range modifies or
deletes an existing prediction-ledger line. Appends are the only legal change.

Usage: check_ledger_immutable.py [base_ref] [head_ref]   (defaults origin/main HEAD)
"""
from __future__ import annotations

import subprocess
import sys


def sh(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True, check=False).stdout


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "origin/main"
    head = sys.argv[2] if len(sys.argv) > 2 else "HEAD"
    # ensure base exists (first push / shallow clone)
    if subprocess.run(["git", "rev-parse", "--verify", base],
                      capture_output=True).returncode != 0:
        print(f"base {base} not found — skipping (first push)")
        return 0
    diff = sh("git", "diff", "--numstat", f"{base}...{head}", "--", "data/ledger/")
    bad = False
    for line in diff.strip().splitlines():
        added, deleted, path = line.split("\t")
        if deleted != "0":
            print(f"IMMUTABILITY VIOLATION: {path} has {deleted} deleted/modified lines")
            bad = True
        else:
            print(f"ok: {path} (+{added} appended)")
    # renames/deletions of ledger files themselves
    status = sh("git", "diff", "--name-status", f"{base}...{head}", "--", "data/ledger/")
    for line in status.strip().splitlines():
        code = line.split("\t")[0]
        if code.startswith(("D", "R")):
            print(f"IMMUTABILITY VIOLATION: ledger file deleted/renamed: {line}")
            bad = True
    if bad:
        print("\nThe prediction ledger is append-only. History is the product.")
        return 1
    print("ledger immutability: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
