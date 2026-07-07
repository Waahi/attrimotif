# -*- coding: utf-8 -*-
"""Example 1 - operator Phi recovers a planted tail a count conflates.

Two size-3 motif classes are built with matched size and central tendency, but
a single high-value instance is planted in ``fan-out``. A count (or count +
mean + a percentile) cannot tell the classes apart; Phi's per-class
distribution exposes the planted maximum.
"""
from __future__ import annotations

import os

import attrimotif as am

OUT = os.path.join(os.path.dirname(__file__), "output")


def main():
    g = am.datasets.planted_tail_example(seed=0)
    counts = am.size3_counts(g.edges)
    phi = am.phi_distributions(g)
    summ = {c: am.tail_summary(v) for c, v in phi.items()}

    print("Example 1 - operator Phi vs a count baseline")
    print(f"  counts:            fan-out={counts['fan-out']}  fan-in={counts['fan-in']}")
    for c in ("fan-out", "fan-in"):
        s = summ[c]
        print(f"  {c:8s} Phi:  mean={s['mean']:.3f}  p99={s['p99']:.3f}  max={s['max']:.3f}")
    print("  -> counts and mean/p99 match; only Phi's max exposes the fan-out tail.")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(OUT, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(phi["fan-in"], bins=40, alpha=0.7, label="fan-in Phi")
        ax.hist(phi["fan-out"], bins=40, alpha=0.7, label="fan-out Phi")
        ax.set_xlabel("re-attached instance value")
        ax.set_ylabel("count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, "example1_planted_tail.png"), dpi=150)
        plt.close(fig)
        print(f"  figure -> {os.path.join(OUT, 'example1_planted_tail.png')}")
    except ImportError:
        print("  (matplotlib not installed; skipping figure)")

    return summ


if __name__ == "__main__":
    main()
