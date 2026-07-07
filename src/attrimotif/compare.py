# -*- coding: utf-8 -*-
"""Portrait Divergence and panel-level comparison.

A network *portrait* (Bagrow & Bollt 2019) is the matrix ``B[l][k]`` counting
nodes that have ``k`` nodes at shortest-path distance ``l``. Portrait Divergence
is the Jensen-Shannon divergence between two portraits' pair-distance
distributions, an all-scales, alignment-free comparison in ``[0, 1]``.

This module ships a dependency-light implementation (numpy + networkx +
scipy.stats.entropy). Passing ``backend="netrd"`` delegates to
``netrd.distance.PortraitDivergence`` when that optional package is installed,
so results can be cross-checked against the reference library.

Limitations (see the methodological-specification docs): portraits use BFS
shortest-path shells and ignore edge weights; disconnected graphs contribute
only finite shells; two empty graphs score 0 and an empty-vs-non-empty pair
scores 1; the comparison is graph-level and is not a causal test. The bundled
implementation uses the Bagrow-Bollt pair weighting as in the author's prior
work; ``backend="netrd"`` gives the reference implementation, which may differ
slightly in how the k-axis is binned across graphs of different sizes.
"""
from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Dict, List, Mapping, Tuple

import numpy as np
from scipy.stats import entropy

try:
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None

from .graph import BipartiteDiGraph


def portrait_matrix(g) -> np.ndarray:
    """Network portrait B-matrix of a ``networkx`` graph."""
    if nx is None:  # pragma: no cover
        raise ImportError("networkx is required for portrait_matrix()")
    nodes = list(g.nodes())
    n = len(nodes)
    if n == 0:
        return np.zeros((1, 1))
    shells = []
    max_d = 0
    for node in nodes:
        d = nx.single_source_shortest_path_length(g, node)
        c = Counter(d.values())
        shells.append(c)
        if d:
            max_d = max(max_d, max(d.values()))
    b = np.zeros((max_d + 1, n + 1))
    for c in shells:
        for l in range(max_d + 1):
            b[l, c.get(l, 0)] += 1
    return b


def _portrait_distribution(b: np.ndarray) -> np.ndarray:
    k = np.arange(b.shape[1])
    w = (b * k).ravel().astype(float)
    s = w.sum()
    return w / s if s > 0 else w


def _js_from_portraits(b1: np.ndarray, b2: np.ndarray) -> float:
    empty1, empty2 = b1.sum() == 0, b2.sum() == 0
    if empty1 and empty2:
        return 0.0  # two empty graphs are identical
    if empty1 or empty2:
        return 1.0  # one empty, one non-empty: maximally different
    rows = max(b1.shape[0], b2.shape[0])
    cols = max(b1.shape[1], b2.shape[1])
    p_mat = np.zeros((rows, cols))
    q_mat = np.zeros((rows, cols))
    p_mat[: b1.shape[0], : b1.shape[1]] = b1
    q_mat[: b2.shape[0], : b2.shape[1]] = b2
    p = _portrait_distribution(p_mat)
    q = _portrait_distribution(q_mat)
    m = 0.5 * (p + q)
    mask = (p + q) > 0
    return float(
        0.5 * entropy(p[mask], m[mask], base=2)
        + 0.5 * entropy(q[mask], m[mask], base=2)
    )


def portrait_divergence(g1, g2, backend: str = "builtin") -> float:
    """Portrait Divergence between two ``networkx`` graphs, in ``[0, 1]``.

    ``backend="builtin"`` (default) uses the bundled implementation;
    ``backend="netrd"`` delegates to ``netrd.distance.PortraitDivergence``.
    """
    if backend == "netrd":
        from netrd.distance import PortraitDivergence  # optional dependency

        return float(PortraitDivergence().dist(g1, g2))
    if backend != "builtin":
        raise ValueError("backend must be 'builtin' or 'netrd'")
    return _js_from_portraits(portrait_matrix(g1), portrait_matrix(g2))


def panel_divergence_matrix(graphs: Mapping) -> Dict[Tuple, float]:
    """Pairwise Portrait Divergence over a mapping ``{label: networkx graph}``."""
    labels = list(graphs.keys())
    return {
        (a, b): portrait_divergence(graphs[a], graphs[b])
        for a, b in combinations(labels, 2)
    }


def panel_permutation_test(
    g: BipartiteDiGraph,
    agent_panel: Mapping,
    n_samples: int = 500,
    min_shared: int = 2,
    seed: int = 0,
    alternative: str = "greater",
) -> Dict[Tuple, Dict]:
    """Random-partition permutation test for cross-panel Portrait Divergence.

    Each panel's object-object projection is built from the arcs of the agents
    assigned to it. The null repeatedly reassigns agents to panels (preserving
    panel sizes) and rebuilds the projections, giving a per-pair permutation
    p-value ``(r + 1) / (n_samples + 1)``.
    """
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative must be 'greater', 'less', or 'two-sided'")
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")
    labels = sorted(set(agent_panel.values()), key=repr)
    agents = [a for a in g.agents() if a in agent_panel]
    lab_arr = np.array([agent_panel[a] for a in agents], dtype=object)

    def build(labels_for_agents):
        out = {}
        for lab in labels:
            keep = [a for a, l in zip(agents, labels_for_agents) if l == lab]
            out[lab] = g.subgraph_agents(keep).project_objects(min_shared)
        return out

    observed_graphs = build(lab_arr)
    pairs = list(combinations(labels, 2))
    obs = {p: portrait_divergence(observed_graphs[p[0]], observed_graphs[p[1]]) for p in pairs}

    rng = np.random.default_rng(seed)
    null = {p: [] for p in pairs}
    for _ in range(n_samples):
        perm = rng.permutation(lab_arr)
        gn = build(perm)
        for p in pairs:
            null[p].append(portrait_divergence(gn[p[0]], gn[p[1]]))

    result = {}
    for p in pairs:
        arr = np.asarray(null[p], float)
        o = obs[p]
        if alternative == "greater":
            r = int(np.sum(arr >= o))
        elif alternative == "less":
            r = int(np.sum(arr <= o))
        else:
            r = int(np.sum(np.abs(arr - arr.mean()) >= abs(o - arr.mean())))
        result[p] = {
            "observed": float(o),
            "null_mean": float(arr.mean()),
            "null_p95": float(np.percentile(arr, 95)),
            "perm_p": float((r + 1) / (n_samples + 1)),
        }
    return result
