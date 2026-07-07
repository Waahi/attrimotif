# -*- coding: utf-8 -*-
"""Degree-preserving null model, significance test, and the
null-identifiability diagnostic.

The null is a bipartite double-edge swap that preserves **both** agent
out-degrees and object in-degrees exactly. :func:`null_test` scores an observed
motif statistic against this null (z-score and a permutation p-value).

The :func:`is_degree_determined` diagnostic reports which statistics are exact
functions of the two degree sequences. For those, the degree-preserving null
reproduces the observed value with (near-)zero variance, so the null is
**not identifiable** for them -- a large observed count is a property of the
degree sequence, not evidence of beyond-degree structure. The diagnostic exists
to stop that value from being read as a significance result; it makes no claim
about whether such motifs are substantively interesting.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple, Union

import numpy as np

from .census import STATISTICS
from .graph import BipartiteDiGraph, Edge

# statistics that are exact functions of the degree sequence(s)
_DEGREE_DETERMINED = {
    "fan-out": "= sum C(agent_degree, 2)",
    "fan-in": "= sum C(object_degree, 2)",
    "fan-out4": "= sum C(agent_degree, 3)",
    "fan-in4": "= sum C(object_degree, 3)",
}


def is_degree_determined(statistic: str) -> Tuple[bool, str]:
    """Return ``(is_degree_determined, explanation)`` for a statistic name."""
    if statistic in _DEGREE_DETERMINED:
        return True, (
            f"'{statistic}' is an exact degree-sequence statistic "
            f"({_DEGREE_DETERMINED[statistic]}); a degree-preserving null "
            f"cannot identify it (null variance ~ 0)."
        )
    return False, ""


def degree_swap(
    edges: Sequence[Edge],
    n_swaps: int,
    rng: np.random.Generator,
    return_count: bool = False,
):
    """Bipartite double-edge swap preserving both-side degrees:
    ``(a1,o1),(a2,o2) -> (a1,o2),(a2,o1)`` when both targets are free.

    Produces a simple graph (no multi-edges or self-loops by construction). With
    ``return_count=True`` also returns the number of swaps actually realized,
    which :func:`null_test` uses to report mixing.
    """
    E = [tuple(e) for e in dict.fromkeys((a, o) for a, o in edges)]
    S = set(E)
    m = len(E)
    if m < 2:
        return (E, 0) if return_count else E
    done = tries = 0
    while done < n_swaps and tries < 50 * n_swaps:
        tries += 1
        i, j = int(rng.integers(0, m)), int(rng.integers(0, m))
        if i == j:
            continue
        a1, o1 = E[i]
        a2, o2 = E[j]
        if a1 == a2 or o1 == o2:
            continue
        if (a1, o2) in S or (a2, o1) in S:
            continue
        S.discard((a1, o1)); S.discard((a2, o2))
        S.add((a1, o2)); S.add((a2, o1))
        E[i] = (a1, o2); E[j] = (a2, o1)
        done += 1
    if return_count:
        return E, done
    return E


def null_test(
    g_or_edges: Union[BipartiteDiGraph, Sequence[Edge]],
    statistic: str,
    n_samples: int = 500,
    swaps_per: int = None,
    seed: int = 0,
    alternative: str = "greater",
    stat_func: Callable = None,
) -> Dict:
    """Score a motif statistic against the degree-preserving null.

    Parameters
    ----------
    statistic : str
        Key in :data:`attrimotif.census.STATISTICS` (or supply ``stat_func``).
    n_samples : int
        Number of null replicates.
    alternative : {'greater', 'less', 'two-sided'}
        Direction for the permutation p-value.

    Returns a dict with the observed value, null mean/sd, z-score, the
    permutation p-value ``(r + 1) / (n_samples + 1)``, and an ``identifiable``
    flag (``False`` when the statistic is degree-determined).
    """
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    if stat_func is None and statistic not in STATISTICS:
        raise ValueError(
            f"unknown statistic {statistic!r}; choose from {sorted(STATISTICS)} or pass stat_func"
        )
    edges = g_or_edges.edges if isinstance(g_or_edges, BipartiteDiGraph) else list(g_or_edges)
    edges = list(dict.fromkeys((a, o) for a, o in edges))
    fn = stat_func or STATISTICS[statistic]
    obs = fn(edges)
    m = len(edges)
    swaps_per = swaps_per if swaps_per is not None else 12 * max(m, 1)
    rng = np.random.default_rng(seed)
    samp_vals, ratios = [], []
    for _ in range(n_samples):
        swapped, done = degree_swap(edges, swaps_per, rng, return_count=True)
        samp_vals.append(fn(swapped))
        ratios.append(done / max(swaps_per, 1))
    samp = np.asarray(samp_vals, float)
    mean, sd = float(samp.mean()), float(samp.std())
    mean_swap_ratio = float(np.mean(ratios))
    z = (obs - mean) / (sd + 1e-12)
    if alternative == "greater":
        r = int(np.sum(samp >= obs))
    elif alternative == "less":
        r = int(np.sum(samp <= obs))
    else:
        r = int(np.sum(np.abs(samp - mean) >= abs(obs - mean)))
    perm_p = (r + 1) / (n_samples + 1)
    dd, why = is_degree_determined(statistic)
    return {
        "statistic": statistic,
        "observed": float(obs),
        "null_mean": mean,
        "null_sd": sd,
        "z": float(z),
        "perm_p": float(perm_p),
        "identifiable": not dd,
        "mean_swap_ratio": mean_swap_ratio,
        "note": why,
    }


# -- multiplicity correction ---------------------------------------------------
def holm_bonferroni(pvals: Sequence[float]) -> np.ndarray:
    """Holm-Bonferroni step-down adjusted p-values (monotone, clipped to 1)."""
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


def benjamini_hochberg(pvals: Sequence[float]) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (monotone, clipped to 1)."""
    p = np.asarray(pvals, float)
    n = p.size
    order = np.argsort(p)
    adj = np.empty(n)
    running = 1.0
    for rank in range(n - 1, -1, -1):
        idx = order[rank]
        val = p[idx] * n / (rank + 1)
        running = min(running, val)
        adj[idx] = min(running, 1.0)
    return adj
