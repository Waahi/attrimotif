# -*- coding: utf-8 -*-
"""Run every illustrative example end to end.

Used both as a user entry point and as a CI smoke test: examples 1-3 are fully
synthetic and must succeed; example 4 (MovieLens) is skipped cleanly when the
public data has not been downloaded.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import example1_planted_tail
import example2_null_identifiability
import example3_synthetic_panel
import example4_movielens


def main() -> int:
    for name, mod in [
        ("example1", example1_planted_tail),
        ("example2", example2_null_identifiability),
        ("example3", example3_synthetic_panel),
        ("example4", example4_movielens),
    ]:
        print(f"\n=== {name} ===")
        mod.main()
    print("\nAll examples completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
