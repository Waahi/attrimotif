# -*- coding: utf-8 -*-
"""Degree-preserving null model, significance test, and the
null-identifiability diagnostic.

The null is a bipartite double-edge swap that preserves **both** agent
out-degrees and object in-degrees exactly. :func:`null_test` scores an observed
motif statistic against this null (z-score and a permutation p-value).

The identifiability guard reports when the null cannot discriminate the observed
value, so that such a value is not read as a significance result. It reaches its
verdict by one of four clearly separated routes, and **only the first is a
proof**:

``registry-proof``
    The statistic is one of the four built-in fan counts, which are exact
    algebraic functions of the two degree sequences, so a degree-preserving null
    reproduces them with exactly zero variance. This route is taken only when
    the caller supplies no ``stat_func`` *and* the registry entry is still the
    canonical function shipped with the package, since
    :data:`attrimotif.census.STATISTICS` is a mutable public mapping and the
    proof is a property of the function, not of its name.

``empirical-degenerate``
    Every null replicate returned exactly the same value. This describes only
    the finite sample drawn in this run: it flags that the run cannot estimate
    null variability. It does **not** prove that the full null distribution is
    degenerate, nor that the statistic is a function of the degree sequence,
    since too few replicates, a degree sequence with (almost) no alternative
    realisation, and a chain that could not move all produce it as well.

``empirical-variable``
    At least two replicates differed. For a deterministic, isomorphism-invariant
    statistic this rules out degree-determinacy. It does not establish that the
    swap chain has mixed, nor that the reported p-value is valid.

``undetermined``
    Fewer than two replicates, or non-finite values, so empirical variability
    cannot be assessed at all. ``identifiable`` is then ``None`` rather than a
    bool, because neither answer is supported.

Neither empirical route establishes swap-chain mixing; see
:func:`swap_convergence` for chain-length diagnostics.

The budget given to :func:`degree_swap` counts **proposed** swaps, so a rejected
proposal is a step of the chain and the rejection self-loops are preserved.
That is what makes the stationary distribution uniform over the graphs sharing
the observed degree sequences. Version 1.0.0 instead ran until a fixed number of
swaps had been *accepted*, which samples the embedded jump chain and
oversamples states with more available swaps; on an exhaustively enumerated
3x3 fibre that rule reproduced the valid-move counts with correlation 0.997 and
a chi-square of 535 on 4 degrees of freedom, against 8.5 for the present rule.
The reported ``mean_acceptance_rate`` is therefore a genuine acceptance rate,
accepted proposals divided by proposals made. A low rate means the chain is
rejection-limited and the budget should be raised; it is not by itself evidence
that the chain has mixed.
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

# The registry proof is a property of the FUNCTION, not of its name, and
# STATISTICS is a mutable public mapping. Snapshot the canonical callables at
# import so a replaced entry cannot inherit the proof.
_CANONICAL_STATISTICS = dict(STATISTICS)


def is_degree_determined(statistic: str) -> Tuple[bool, str]:
    """Return ``(is_degree_determined, explanation)`` for a **statistic name**.

    This is an exact registry lookup over the four built-in fan statistics, and
    it is meaningful only when that name describes the function actually being
    computed. :func:`null_test` therefore consults it only when no ``stat_func``
    override is supplied; a user-supplied statistic is assessed empirically
    instead (see the module docstring).
    """
    if statistic in _DEGREE_DETERMINED:
        return True, (
            f"'{statistic}' is an exact degree-sequence statistic "
            f"({_DEGREE_DETERMINED[statistic]}); a degree-preserving null "
            f"cannot identify it (null variance ~ 0)."
        )
    return False, ""


def degree_swap(
    edges: Sequence[Edge],
    n_steps: int,
    rng: np.random.Generator,
    return_count: bool = False,
    count: str = "proposals",
) -> Union[List[Edge], Tuple[List[Edge], int]]:
    """Bipartite double-edge swap preserving both-side degrees:
    ``(a1,o1),(a2,o2) -> (a1,o2),(a2,o1)`` when both targets are free.

    Produces a simple graph (no multi-edges or self-loops by construction).

    Parameters
    ----------
    n_steps : int
        Budget, interpreted according to ``count``.
    count : {'proposals', 'successes'}
        ``'proposals'`` (the default, and the only setting that samples
        correctly) spends the budget on **proposed** swaps, so a rejected
        proposal counts as a step and the chain stays where it is. That keeps
        the rejection self-loops, and with them the uniform stationary
        distribution over the graphs sharing the observed degree sequences.

        ``'successes'`` reproduces the behaviour of attrimotif 1.0.0, which ran
        until a fixed number of swaps had been **accepted**. Removing the
        self-loops that way samples the embedded jump chain, whose stationary
        weight is proportional to the number of valid moves out of each state
        rather than uniform, so states with more available swaps are
        oversampled. It is retained only for reproducing v1.0.0 numbers and
        should not be used for inference.

    With ``return_count=True`` also returns the number of swaps **accepted**,
    which :func:`null_test` divides by the budget to report the acceptance rate.
    """
    if count not in {"proposals", "successes"}:
        raise ValueError("count must be 'proposals' or 'successes'")
    E = [tuple(e) for e in dict.fromkeys((a, o) for a, o in edges)]
    S = set(E)
    m = len(E)
    if m < 2:
        return (E, 0) if return_count else E
    done = steps = 0
    limit = n_steps if count == "proposals" else 50 * n_steps
    while steps < limit and (count == "proposals" or done < n_steps):
        steps += 1
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
    return_samples: bool = False,
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
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    if swaps_per is not None and (
        isinstance(swaps_per, bool)
        or not isinstance(swaps_per, (int, np.integer))
        or swaps_per < 0
    ):
        raise ValueError("swaps_per must be a non-negative integer or None")
    if stat_func is None and statistic not in STATISTICS:
        raise ValueError(
            f"unknown statistic {statistic!r}; choose from {sorted(STATISTICS)} or pass stat_func"
        )
    edges = g_or_edges.edges if isinstance(g_or_edges, BipartiteDiGraph) else list(g_or_edges)
    edges = list(dict.fromkeys((a, o) for a, o in edges))
    # explicit None-check: a valid callable may be falsy under __bool__
    fn = STATISTICS[statistic] if stat_func is None else stat_func
    obs = fn(edges)
    m = len(edges)
    swaps_per = swaps_per if swaps_per is not None else 12 * max(m, 1)
    rng = np.random.default_rng(seed)
    samp_vals, ratios = [], []
    for _ in range(n_samples):
        swapped, done = degree_swap(edges, swaps_per, rng, return_count=True)
        samp_vals.append(fn(swapped))
        ratios.append(done / max(swaps_per, 1))  # accepted / proposed
    samp = np.asarray(samp_vals, float)
    mean, sd = float(samp.mean()), float(samp.std())
    mean_acceptance_rate = float(np.mean(ratios))
    z = (obs - mean) / (sd + 1e-12)
    if alternative == "greater":
        r = int(np.sum(samp >= obs))
    elif alternative == "less":
        r = int(np.sum(samp <= obs))
    else:
        # Centre on the POOLED set, observed value included. Centring on the
        # replicate mean alone makes the ranking asymmetric in obs, which breaks
        # the exchangeability the (r+1)/(R+1) estimator rests on: calibrated
        # under a true null at R=19 that version rejects at 0.062 against a
        # nominal 0.05, while this one holds the level.
        centre = float((samp.sum() + obs) / (n_samples + 1))
        r = int(np.sum(np.abs(samp - centre) >= abs(obs - centre)))
    perm_p = (r + 1) / (n_samples + 1)

    # -- identifiability guard: proof first, then the empirical routes ---------
    # The registry proof belongs to the canonical FUNCTION. It is withheld when
    # the caller overrides stat_func, and also when the public STATISTICS entry
    # has been replaced, since the name would then no longer describe what ran.
    if stat_func is None and STATISTICS.get(statistic) is _CANONICAL_STATISTICS.get(statistic):
        dd, why = is_degree_determined(statistic)
    else:
        dd, why = False, ""
    # Degeneracy is tested on the RAW returned values, before the float coercion
    # used for the moments: two distinct integers above 2**53 collapse to one
    # float and would otherwise be misread as a degenerate null.
    finite = bool(np.all(np.isfinite(samp))) and np.isfinite(float(obs))
    if n_samples < 2:
        null_degenerate = None
        undetermined_why = f"only {n_samples} replicate(s): empirical variability cannot be assessed."
    elif not finite:
        null_degenerate = None
        undetermined_why = ("the null produced non-finite values, so empirical "
                            "variability cannot be assessed.")
    else:
        first = samp_vals[0]
        null_degenerate = all(v == first for v in samp_vals)
        undetermined_why = ""
    if dd:
        route = "registry-proof"
    elif null_degenerate is None:
        route = "undetermined"
        why = undetermined_why
    elif null_degenerate:
        route = "empirical-degenerate"
        why = (
            f"all {n_samples} null replicates returned exactly the same value, so "
            f"this run cannot estimate null variability for {statistic!r}. That is "
            "expected for a degree-determined statistic, but it does not prove one: "
            "too few replicates, a degree sequence with (almost) no alternative "
            "realisation, and a chain that could not move produce it as well. Check "
            "mean_swap_ratio and swap_convergence() before interpreting this."
        )
    else:
        route = "empirical-variable"
    # Tri-state: None when neither answer is supported by the evidence.
    identifiable = None if null_degenerate is None and not dd else not (dd or null_degenerate)
    out = {
        "statistic": statistic,
        "observed": float(obs),
        "null_mean": mean,
        "null_sd": sd,
        "z": float(z),
        "perm_p": float(perm_p),
        "p_resolution": 1.0 / (n_samples + 1),
        "identifiable": identifiable,
        "degree_determined": bool(dd),
        "null_degenerate": null_degenerate,
        "identifiability_route": route,
        "n_samples": int(n_samples),
        "proposals_per_replicate": int(swaps_per),
        "swaps_requested": int(swaps_per),
        "mean_acceptance_rate": mean_acceptance_rate,
        # retained under its 1.0.0 name; the value is now the
        # acceptance rate, since the budget counts proposals
        "mean_swap_ratio": mean_acceptance_rate,
        "note": why,
    }
    if return_samples:
        # The replicate values themselves, so that a caller plotting the null
        # (a histogram, a QQ plot) does not have to re-implement the swap loop.
        # Re-implementing it is how the manuscript figure kept drawing the
        # pre-1.1.0 biased null after the package had been corrected.
        out["null_samples"] = samp.copy()
    return out


def swap_convergence(
    g_or_edges: Union[BipartiteDiGraph, Sequence[Edge]],
    statistic: str = "overlap",
    multipliers: Sequence[float] = (0.25, 0.5, 1, 2, 3, 6, 12, 24),
    n_samples: int = 20,
    seed: int = 0,
    stat_func: Callable = None,
) -> List[Dict]:
    """Chain-length sensitivity of the null, as a function of swaps per arc.

    Runs the degree-preserving null at several chain lengths, each expressed as
    a multiple of the number of arcs, and reports the null mean and sd plus the
    realized swap ratio at each. The default chain length used by
    :func:`null_test` is ``12 * |E|``; a flat trace across multipliers spanning
    that value is the practical evidence that the reported null is not sensitive
    to the chain length. The acceptance rate is not itself such a criterion:
    it falls with density because fewer swaps are legal, not because anything
    is wrong. What matters is how many accepted moves the budget buys per arc,
    ``budget * acceptance / |E|``.

    This is a diagnostic, not a convergence proof: no mixing-time or
    spectral-gap bound is claimed for the bipartite swap chain on an arbitrary
    degree sequence, and a flat trace cannot rule out a slow mode the statistic
    does not see.
    """
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    multipliers = list(multipliers)
    if not multipliers or any(not np.isfinite(x) or x <= 0 for x in multipliers):
        raise ValueError("multipliers must be a non-empty sequence of positive finite numbers")
    edges = g_or_edges.edges if isinstance(g_or_edges, BipartiteDiGraph) else list(g_or_edges)
    edges = list(dict.fromkeys((a, o) for a, o in edges))
    if stat_func is None and statistic not in STATISTICS:
        raise ValueError(
            f"unknown statistic {statistic!r}; choose from {sorted(STATISTICS)} or pass stat_func"
        )
    fn = STATISTICS[statistic] if stat_func is None else stat_func
    m = max(len(edges), 1)
    obs = float(fn(edges))
    trace: List[Dict] = []
    for mult in multipliers:
        swaps = max(int(round(mult * m)), 1)
        rng = np.random.default_rng(seed)
        vals, ratios = [], []
        for _ in range(n_samples):
            swapped, done = degree_swap(edges, swaps, rng, return_count=True)
            vals.append(fn(swapped))
            ratios.append(done / swaps)
        arr = np.asarray(vals, float)
        trace.append({
            "multiplier": float(mult),
            "swaps_requested": int(swaps),
            "observed": obs,
            "null_mean": float(arr.mean()),
            "null_sd": float(arr.std()),
            "mean_acceptance_rate": float(np.mean(ratios)),
            "mean_swap_ratio": float(np.mean(ratios)),
            "min_acceptance_rate": float(np.min(ratios)),
            "min_swap_ratio": float(np.min(ratios)),
        })
    return trace


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
