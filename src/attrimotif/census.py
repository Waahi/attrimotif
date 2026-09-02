# -*- coding: utf-8 -*-
"""Directed bipartite motif census (size-3 and size-4) + a directed unipartite
size-3 passthrough.

Motif classes enumerated on the agent -> object bipartite graph:

===========  =================================  ==============================
class        description                        degree relationship
===========  =================================  ==============================
``fan-out``  one agent -> two objects (3-A)     = sum C(agent_degree, 2)
``fan-in``   two agents -> one object  (3-B)     = sum C(object_degree, 2)
``fan-out4`` one agent -> three objects (4-A)    = sum C(agent_degree, 3)
``fan-in4``  three agents -> one object (4-D)    = sum C(object_degree, 3)
``overlap``  two agents both -> two objects      NOT a pure degree statistic
             (2x2 biclique / bipartite 4-cycle)
===========  =================================  ==============================

All counts are of **induced** instances. The fan classes are exact functions of
the two degree sequences; the ``overlap`` class is not (see
:mod:`attrimotif.nulls` for why this matters for significance testing).
"""
from __future__ import annotations

from itertools import combinations
from math import comb
from typing import Dict, List, Sequence, Tuple

from .graph import BipartiteDiGraph, Edge

Instance = Tuple[Edge, Edge]


def _adjacency(edges):
    out, inn = {}, {}
    for a, o in edges:
        out.setdefault(a, set()).add(o)
        inn.setdefault(o, set()).add(a)
    return out, inn


# -- size 3 --------------------------------------------------------------------
def enumerate_size3(edges: Sequence[Edge]) -> Dict[str, List[Instance]]:
    """Induced size-3 bipartite motif instances, each a pair of arcs.

    ``fan-out`` = a pair of an agent's arcs; ``fan-in`` = a pair of an object's
    arcs. Edges are de-duplicated first.
    """
    uniq = list(dict.fromkeys((a, o) for a, o in edges))
    out, inn = _adjacency(uniq)
    fanout = [
        ((a, o1), (a, o2))
        for a, objs in out.items()
        for o1, o2 in combinations(sorted(objs, key=repr), 2)
    ]
    fanin = [
        ((a1, o), (a2, o))
        for o, ags in inn.items()
        for a1, a2 in combinations(sorted(ags, key=repr), 2)
    ]
    return {"fan-out": fanout, "fan-in": fanin}


def size3_counts(edges: Sequence[Edge]) -> Dict[str, int]:
    """Size-3 fan counts in closed form: ``fan-out`` = sum C(agent_degree, 2),
    ``fan-in`` = sum C(object_degree, 2).

    Equivalent to ``{c: len(v) for c, v in enumerate_size3(edges).items()}`` by
    the bijection between a hub's size-3 fans and the unordered pairs of its
    distinct neighbours, but it never materialises the instance lists. Degrees
    are taken over **de-duplicated** arcs, matching :func:`enumerate_size3`.
    Use :func:`enumerate_size3` when the instances themselves are needed (the
    stratified census and operator Phi consume them).
    """
    uniq = list(dict.fromkeys((a, o) for a, o in edges))
    out, inn = _adjacency(uniq)
    return {
        "fan-out": sum(comb(len(objs), 2) for objs in out.values()),
        "fan-in": sum(comb(len(ags), 2) for ags in inn.values()),
    }


# -- size 4 --------------------------------------------------------------------
def size4_fan_counts(edges: Sequence[Edge]) -> Dict[str, int]:
    """Induced size-4 FAN counts: ``fan-out4`` = sum C(agent_deg, 3),
    ``fan-in4`` = sum C(object_deg, 3). Like the size-3 fans these are exact
    degree-sequence statistics."""
    uniq = list(dict.fromkeys((a, o) for a, o in edges))
    out, inn = _adjacency(uniq)
    return {
        "fan-out4": sum(comb(len(objs), 3) for objs in out.values()),
        "fan-in4": sum(comb(len(ags), 3) for ags in inn.values()),
    }


#: Above this many agents the dense route's ``n_A**2`` co-occurrence matrix is
#: judged too large to allocate on a commodity machine, and ``backend="auto"``
#: stays on the sparse route. 20000 agents is about 1.6 GiB in float32.
MATRIX_AGENT_LIMIT = 20000


def _overlap_sparse(out, agents) -> int:
    total = 0
    for i in range(len(agents)):
        si = out[agents[i]]
        for j in range(i + 1, len(agents)):
            common = len(si & out[agents[j]])
            if common >= 2:
                total += common * (common - 1) // 2
    return total


#: float32 represents integers exactly only up to 2**24. Co-occurrence entries
#: are bounded by the number of objects, so the matmul is exact below this; the
#: *sum* of binomial terms is not, and is accumulated in int64 instead.
_F32_EXACT_INT = 1 << 24


def _overlap_matrix(uniq, agents) -> int:
    import numpy as np

    # insertion order, not sorted(): node labels of mixed types are legal here
    # and sorting them raises TypeError, which the sparse route never does
    objects = list(dict.fromkeys(o for _, o in uniq))
    if len(objects) >= _F32_EXACT_INT:
        raise ValueError(
            "the dense overlap route needs fewer than 2**24 objects for the "
            "co-occurrence matrix to be exact in float32; use backend='sparse'")
    ai = {a: i for i, a in enumerate(agents)}
    oi = {o: i for i, o in enumerate(objects)}
    b = np.zeros((len(agents), len(objects)), dtype=np.float32)
    for a, o in uniq:
        b[ai[a], oi[o]] = 1.0
    # Each product entry is a count of shared objects, at most len(objects), so
    # it is exact in float32. The reduction is not: summing C(m,2) over
    # O(n_A^2) pairs reaches values far beyond 2**24, where float32 starts
    # rounding. Measured on a 3162-agent graph the float32 reduction returned
    # 2,497,105,408 against the true 2,497,105,317. Cast to int64 first.
    m = (b @ b.T)[np.triu_indices(len(agents), k=1)].astype(np.int64)
    return int(np.sum(m * (m - 1) // 2, dtype=np.int64))


def overlap_count(edges: Sequence[Edge], backend: str = "auto") -> int:
    """Number of 2x2 bicliques (a1,a2 both linked to o1,o2): induced size-4
    'overlap' motifs.

    Unlike the fan classes this is not a pure degree statistic, so a
    degree-preserving null *can* be informative for it. Whether it is remains
    graph-specific: a degree sequence admitting (almost) no alternative
    realisation makes the null degenerate here too, which
    :func:`attrimotif.nulls.null_test` reports as ``empirical-degenerate``.

    Two routes compute the same integer. ``"sparse"`` intersects adjacency sets
    pairwise, costing ``O(sum_{a<a'} min(d_a, d_a'))`` time and
    ``O(|E| + n_A + n_O)`` memory. ``"matrix"`` forms the dense co-occurrence
    matrix ``B B^T``, costing ``O(n_A^2 n_O)`` operations and ``O(n_A^2)``
    memory, but spends them inside BLAS. Measured on this package's benchmark
    the dense route is the faster of the two from roughly a thousand agents
    upward at every density tested, by more than an order of magnitude at ten
    thousand, and pays for it in memory. ``"auto"`` therefore takes the dense
    route once there are at least 1000 agents and the matrix is small enough to
    allocate, and the sparse route otherwise.

    The two routes are pinned to each other by an equality test, so a
    disagreement is a test failure rather than a silently different number.
    """
    if backend not in ("auto", "sparse", "matrix"):
        raise ValueError("backend must be 'auto', 'sparse' or 'matrix'")
    uniq = list(dict.fromkeys((a, o) for a, o in edges))
    out, _ = _adjacency(uniq)
    agents = list(out)
    if backend == "sparse":
        return _overlap_sparse(out, agents)
    if backend == "matrix":
        return _overlap_matrix(uniq, agents)
    if 1000 <= len(agents) <= MATRIX_AGENT_LIMIT:
        return _overlap_matrix(uniq, agents)
    return _overlap_sparse(out, agents)


# -- public statistic registry (edges -> scalar) -------------------------------
STATISTICS = {
    "fan-out": lambda e: size3_counts(e)["fan-out"],
    "fan-in": lambda e: size3_counts(e)["fan-in"],
    "fan-out4": lambda e: size4_fan_counts(e)["fan-out4"],
    "fan-in4": lambda e: size4_fan_counts(e)["fan-in4"],
    "overlap": overlap_count,
}


def census(g: BipartiteDiGraph) -> Dict[str, int]:
    """Full size-3 + size-4 bipartite motif census for a graph."""
    counts = dict(size3_counts(g.edges))
    counts.update(size4_fan_counts(g.edges))
    counts["overlap"] = overlap_count(g.edges)
    return counts


def directed_triadic_census(graph: "nx.DiGraph") -> Dict[str, int]:
    """Baseline passthrough to ``networkx.triadic_census`` for **directed
    unipartite** size-3 motifs (the classic 16-class census).

    Provided for convenience and interoperability; it is not the package's
    contribution (attributed *bipartite* motifs are). ``graph`` must be a
    :class:`networkx.DiGraph`.
    """
    import networkx as nx  # local import; only needed for this passthrough

    if not isinstance(graph, nx.DiGraph):
        raise TypeError("directed_triadic_census requires a networkx.DiGraph")
    return dict(nx.triadic_census(graph))
