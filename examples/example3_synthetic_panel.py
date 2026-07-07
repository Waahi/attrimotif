# -*- coding: utf-8 -*-
"""Example 3 - the full workflow on a synthetic attributed panel.

Runs all four modules on a multi-network panel of attributed directed bipartite
graphs: census, category stratification, operator Phi, and a cross-panel
Portrait Divergence permutation test.
"""
from __future__ import annotations

import os

import attrimotif as am

OUT = os.path.join(os.path.dirname(__file__), "output")


def main():
    d = am.datasets.synthetic_panel(n_panels=4, agents_per_panel=40, seed=0)
    g = d["graph"]

    print("Example 3 - full attributed-panel workflow")
    print("  census:", am.census(g))
    strat = am.stratified_census(g)
    print("  fan-in by object category:", strat["fan-in"])
    phi = am.phi_distributions(g)
    print("  Phi fan-out tail: max =", round(am.tail_summary(phi["fan-out"])["max"], 2))

    res = am.panel_permutation_test(g, d["agent_panel"], n_samples=200, seed=0)
    print("  cross-panel Portrait Divergence (observed / perm-p):")
    for (a, b), r in res.items():
        print(f"    {a} vs {b}: PD={r['observed']:.3f}  perm_p={r['perm_p']:.3f}")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        labels = d["panels"]
        n = len(labels)
        mat = np.zeros((n, n))
        for i, a in enumerate(labels):
            for j, b in enumerate(labels):
                if i < j:
                    mat[i, j] = mat[j, i] = res[(a, b)]["observed"]
        os.makedirs(OUT, exist_ok=True)
        fig, ax = plt.subplots(figsize=(4.5, 4))
        im = ax.imshow(mat, cmap="viridis")
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticks(range(n)); ax.set_yticklabels(labels)
        fig.colorbar(im, ax=ax, label="Portrait Divergence")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "example3_panel_pd.png"), dpi=150)
        plt.close(fig)
        print(f"  figure -> {os.path.join(OUT, 'example3_panel_pd.png')}")
    except ImportError:
        print("  (matplotlib not installed; skipping figure)")

    return res


if __name__ == "__main__":
    main()
