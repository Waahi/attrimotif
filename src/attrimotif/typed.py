# -*- coding: utf-8 -*-
"""Node-category stratification and the attribute operator Phi.

Two attribute-aware readings of the size-3 bipartite census:

* :func:`stratified_census` -- counts each motif class broken down by the
  categorical type of the motif's hub node (the shared agent for ``fan-out``,
  the shared object for ``fan-in``). This is the v1.0 "typed" census: a
  stratification over fixed bipartite templates, not an arbitrary colored-graph
  isomorphism engine.
* :func:`phi_distributions` -- operator Phi: attaches the numeric edge
  attribute to every motif instance and returns the per-class *distribution* of
  re-attached values, exposing tails that a count (or a count + mean + a
  percentile) collapses.
"""
from __future__ import annotations

from typing import Callable, Dict, Sequence

import numpy as np

from .census import enumerate_size3
from .graph import BipartiteDiGraph


def stratified_census(g: BipartiteDiGraph, with_support: bool = False) -> Dict[str, Dict]:
    """Size-3 motif counts stratified by hub-node category.

    The stratification is keyed on the motif's **hub**, which is the shared
    agent for ``fan-out`` and the shared object for ``fan-in``; it is not keyed
    on one side of the graph. Categories are read from ``g.agent_type`` /
    ``g.object_type``; nodes without a type contribute under ``None``.

    Returns ``{"fan-out": {agent_category: count}, "fan-in": {object_category:
    count}}``.

    With ``with_support=True`` each entry becomes ``{"instances": count,
    "hubs": n}``, where ``hubs`` is the number of distinct hub nodes the
    instances came from. The counts are exact either way, but a stratum whose
    instances all come from one hub supports far less than the same count spread
    over many, and the count alone cannot tell the two apart.
    """
    inst = enumerate_size3(g.edges)
    fanout: Dict = {}
    fanout_hubs: Dict = {}
    for e1, _e2 in inst["fan-out"]:
        cat = g.agent_type.get(e1[0])
        fanout[cat] = fanout.get(cat, 0) + 1
        fanout_hubs.setdefault(cat, set()).add(e1[0])
    fanin: Dict = {}
    fanin_hubs: Dict = {}
    for e1, _e2 in inst["fan-in"]:
        cat = g.object_type.get(e1[1])
        fanin[cat] = fanin.get(cat, 0) + 1
        fanin_hubs.setdefault(cat, set()).add(e1[1])
    if not with_support:
        return {"fan-out": fanout, "fan-in": fanin}
    return {
        "fan-out": {c: {"instances": n, "hubs": len(fanout_hubs[c])}
                    for c, n in fanout.items()},
        "fan-in": {c: {"instances": n, "hubs": len(fanin_hubs[c])}
                   for c, n in fanin.items()},
    }


def phi_distributions(
    g: BipartiteDiGraph, reduce: Callable = np.mean
) -> Dict[str, np.ndarray]:
    """Operator Phi: per-class multiset of re-attached instance values.

    Each size-3 instance's value is ``reduce`` applied to the attributes of its
    two participating arcs. Instances with a missing attribute are skipped.
    Returns ``{class: np.ndarray}``.
    """
    inst = enumerate_size3(g.edges)
    res: Dict[str, np.ndarray] = {}
    for cls, pairs in inst.items():
        vals = []
        for e1, e2 in pairs:
            a1 = g.edge_attr.get(e1)
            a2 = g.edge_attr.get(e2)
            if a1 is None or a2 is None:
                continue
            vals.append(reduce([a1, a2]))
        res[cls] = np.asarray(vals, dtype=float)
    return res


def tail_summary(x: Sequence[float]) -> Dict[str, float]:
    """Compact tail-aware summary of a value distribution."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"),
                "p95": float("nan"), "p99": float("nan"), "max": float("nan"),
                "skew": float("nan")}
    m, s = x.mean(), x.std()
    return {
        "n": int(x.size),
        "mean": float(m),
        "sd": float(s),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(x.max()),
        "skew": float(((x - m) ** 3).mean() / (s ** 3 + 1e-12)),
    }
