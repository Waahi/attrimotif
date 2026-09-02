# -*- coding: utf-8 -*-
"""Example 2 - the null-identifiability diagnostic.

The degree-preserving null cannot identify the fan counts (they are exact
degree-sequence statistics, null sd ~ 0). The 2x2 overlap count is not a pure
degree statistic, so the same null has power and returns a real z-score.
"""
from __future__ import annotations

import attrimotif as am

# Number of degree-preserving null replicates. The attainable p-value floor is
# 1 / (R + 1), so R = 300 resolves p down to 0.0033.
R = 300


def main():
    # The graph is a shipped generator, so this example is reproducible from
    # the distributed package alone (25 core agents of degree 3 over 4 core
    # objects, plus a 40-agent degree-1 rim: 115 arcs).
    g = am.datasets.clustered_core_rim(seed=0)
    print("Example 2 - null-identifiability diagnostic")
    print(f"  graph: {g}  (generator: datasets.clustered_core_rim, seed=0)")
    for stat in ("fan-out", "fan-in", "overlap"):
        r = am.null_test(g, stat, n_samples=R, seed=0)
        tag = "identifiable" if r["identifiable"] else "NOT identifiable"
        print(
            f"  {stat:8s}: obs={r['observed']:.0f}  null={r['null_mean']:.1f}"
            f"±{r['null_sd']:.2f}  z={r['z']:.2f}  p={r['perm_p']:.4f}"
            f"  [{tag} via {r['identifiability_route']}]"
        )
    print(f"  (R = {R} replicates, so the p-value floor is 1/{R + 1} = {1 / (R + 1):.4f})")
    print("  -> fan counts have zero null variance (unidentifiable); overlap carries real signal.")
    return True


if __name__ == "__main__":
    main()
