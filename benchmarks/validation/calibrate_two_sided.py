# -*- coding: utf-8 -*-
r"""Calibrate the two-sided permutation p-value under a true null.

Codex's second round argued that the two-sided branch breaks the finite-R
Type-I guarantee because it centres on the mean of the permutation samples
alone, so the observed value does not participate symmetrically and the pooled
set is no longer exchangeable under the ranking.

This checks the claim independently rather than accepting it. Under the null,
obs and the R replicates are drawn iid from the same law, so an exact test must
reject at rate at most alpha. Two centrings are compared:

  current : |x - mean(samp)|          asymmetric, obs excluded from the centre
  pooled  : |x - mean(samp + [obs])|  symmetric in the pooled set
"""
from __future__ import annotations

import sys

import numpy as np

ALPHA = 0.05
TRIALS = 200_000
RS = (19, 99, 199)


def rates(rng, n_samples, dist):
    """Rejection rate at ALPHA for both centrings, over TRIALS null draws."""
    obs = dist(rng, TRIALS)
    samp = dist(rng, (TRIALS, n_samples))

    # current: centre on the permutation samples only
    c_cur = samp.mean(axis=1, keepdims=True)
    r_cur = (np.abs(samp - c_cur) >= np.abs(obs[:, None] - c_cur)).sum(axis=1)
    p_cur = (r_cur + 1) / (n_samples + 1)

    # pooled: centre on the pooled set, which is symmetric in obs and samples
    pooled = np.concatenate([samp, obs[:, None]], axis=1)
    c_pool = pooled.mean(axis=1, keepdims=True)
    r_pool = (np.abs(samp - c_pool) >= np.abs(obs[:, None] - c_pool)).sum(axis=1)
    p_pool = (r_pool + 1) / (n_samples + 1)

    return float((p_cur <= ALPHA).mean()), float((p_pool <= ALPHA).mean())


def main() -> int:
    rng = np.random.default_rng(20260902)
    dists = {
        "normal": lambda g, size: g.normal(0.0, 1.0, size),
        "skewed (lognormal)": lambda g, size: g.lognormal(0.0, 1.0, size),
        "counts (poisson 50)": lambda g, size: g.poisson(50.0, size).astype(float),
    }
    print(f"  rejection rate at alpha={ALPHA}, {TRIALS:,} null trials per cell")
    print(f"  {'distribution':<22}{'R':>6}{'current':>12}{'pooled':>10}")
    bad = 0
    for name, dist in dists.items():
        for r in RS:
            cur, pool = rates(rng, r, dist)
            flag = "  <-- exceeds alpha" if cur > ALPHA + 0.004 else ""
            if flag:
                bad += 1
            print(f"  {name:<22}{r:>6}{cur:>12.4f}{pool:>10.4f}{flag}")
    print()
    if bad:
        print(f"  {bad} cells exceed the nominal level under the current centring;")
        print("  the pooled centring holds it. The claim is confirmed.")
    else:
        print("  no cell exceeds the nominal level: the claim is NOT reproduced.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
