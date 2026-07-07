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
from typing import Dict, List, Tuple

from .graph import BipartiteDiGraph, Edge

Instance = Tuple[Edge, Edge]


def _adjacency(edges):
    out, inn = {}, {}
    for a, o in edges:
        out.setdefault(a, set()).add(o)
        inn.setdefault(o, set()).add(a)
    return out, inn


# -- size 3 --------------------------------------------------------------------
def enumerate_size3(edges) -> Dict[str, List[Instance]]:
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


def size3_counts(edges) -> Dict[str, int]:
    inst = enumerate_size3(edges)
    return {c: len(v) for c, v in inst.items()}


# -- size 4 --------------------------------------------------------------------
def size4_fan_counts(edges) -> Dict[str, int]:
    """Induced size-4 FAN counts: ``fan-out4`` = sum C(agent_deg, 3),
    ``fan-in4`` = sum C(object_deg, 3). Like the size-3 fans these are exact
    degree-sequence statistics."""
    uniq = list(dict.fromkeys((a, o) for a, o in edges))
    out, inn = _adjacency(uniq)
    return {
        "fan-out4": sum(comb(len(objs), 3) for objs in out.values()),
        "fan-in4": sum(comb(len(ags), 3) for ags in inn.values()),
    }


def overlap_count(edges) -> int:
    """Number of 2x2 bicliques (a1,a2 both linked to o1,o2): induced size-4
    'overlap' motifs. Not a pure degree statistic, so a degree-preserving null
    is informative here."""
    uniq = list(dict.fromkeys((a, o) for a, o in edges))
    out, _ = _adjacency(uniq)
    agents = list(out)
    total = 0
    for i in range(len(agents)):
        si = out[agents[i]]
        for j in range(i + 1, len(agents)):
            common = len(si & out[agents[j]])
            if common >= 2:
                total += common * (common - 1) // 2
    return total


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


def directed_triadic_census(graph) -> Dict[str, int]:
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
