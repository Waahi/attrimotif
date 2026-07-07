# -*- coding: utf-8 -*-
"""Example 2 - the null-identifiability diagnostic.

The degree-preserving null cannot identify the fan counts (they are exact
degree-sequence statistics, null sd ~ 0). The 2x2 overlap count is not a pure
degree statistic, so the same null has power and returns a real z-score.
"""
from __future__ import annotations

import numpy as np

import attrimotif as am


def _graph_with_planted_overlap(seed=0):
    rng = np.random.default_rng(seed)
    edges = []
    # a clustered core: agents co-select the same objects -> genuine overlap
    core = ["o1", "o2", "o3", "o4"]
    for i in range(25):
        for o in rng.choice(core, size=3, replace=False):
            edges.append((f"c{i}", o))
    # a dispersed rim: adds degree without co-occurrence
    for i in range(40):
        edges.append((f"r{i}", f"p{i}"))
    return edges


def main():
    edges = _graph_with_planted_overlap()
    print("Example 2 - null-identifiability diagnostic")
    for stat in ("fan-out", "fan-in", "overlap"):
        r = am.null_test(edges, stat, n_samples=300, seed=0)
        tag = "identifiable" if r["identifiable"] else "NOT identifiable"
        print(
            f"  {stat:8s}: obs={r['observed']:.0f}  null={r['null_mean']:.1f}"
            f"±{r['null_sd']:.2f}  z={r['z']:.2f}  p={r['perm_p']:.3f}  [{tag}]"
        )
    print("  -> fan counts have null sd ~ 0 (unidentifiable); overlap carries real signal.")
    return True


if __name__ == "__main__":
    main()
