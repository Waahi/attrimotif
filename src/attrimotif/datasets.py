# -*- coding: utf-8 -*-
"""Deterministic synthetic generators for the illustrative examples.

No proprietary or real data ships with the package. All generators are seeded
and fully specified here.

* :func:`planted_tail_example` -- two size-3 classes with matched central
  tendency but a single high-value instance planted in ``fan-out``, so a count
  (or count + mean + percentile) cannot separate the classes but operator Phi's
  per-class distribution can.
* :func:`clustered_core_rim` -- a clustered core of agents co-selecting a small
  object set (genuine ``overlap``) plus a dispersed rim that adds degree without
  co-occurrence, so the fan counts are inflated by degree alone while
  ``overlap`` carries beyond-degree signal. Used for the null-identifiability
  example.
* :func:`synthetic_panel` -- a multi-network panel of attributed directed
  bipartite graphs with non-trivial object categories, an attribute tail in one
  category, and clustered vs. dispersed panels that differ in Portrait
  Divergence -- enough to exercise all four modules end to end.

.. note::
   The arc **order** these generators emit is part of their reproducible
   contract: :func:`attrimotif.nulls.degree_swap` draws edges by positional
   index, so a permuted arc list changes the swap trajectory (and hence the
   null mean, sd and z) at a fixed seed even when the topology is identical.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from .graph import BipartiteDiGraph


def planted_tail_example(seed: int = 0) -> BipartiteDiGraph:
    """Edge-disjoint fan-out and fan-in blocks; one planted fan-out tail."""
    rng = np.random.default_rng(seed)
    edges = []
    attr: Dict = {}

    # fan-out block: 100 agents of degree 3 (private objects) -> 300 instances
    oid = 0
    for i in range(100):
        a = f"FA{i}"
        objs = [f"FO{oid + j}" for j in range(3)]
        oid += 3
        for o in objs:
            edges.append((a, o))
            attr[(a, o)] = float(max(rng.normal(0.30, 0.02), 1e-3))

    # planted single fan-out instance: one degree-2 agent, both arcs high-value
    a = "FA_planted"
    objs = [f"FO{oid}", f"FO{oid + 1}"]
    oid += 2
    for o in objs:
        edges.append((a, o))
        attr[(a, o)] = float(rng.normal(6.48, 0.05))

    # fan-in block: 100 objects of degree 3 (private agents) -> 300 instances
    aid = 0
    for j in range(100):
        o = f"GO{j}"
        ags = [f"GA{aid + k}" for k in range(3)]
        aid += 3
        for a in ags:
            edges.append((a, o))
            attr[(a, o)] = float(max(rng.normal(0.30, 0.02), 1e-3))

    return BipartiteDiGraph(edges, edge_attr=attr)


def clustered_core_rim(
    seed: int = 0,
    n_core: int = 25,
    core_objects: int = 4,
    core_degree: int = 3,
    n_rim: int = 40,
) -> BipartiteDiGraph:
    """Clustered core plus dispersed rim: the null-identifiability example graph.

    ``n_core`` agents each select ``core_degree`` distinct objects, without
    replacement, from a shared core of ``core_objects`` objects. A rim of
    ``n_rim`` degree-1 agents, each attached to its own private object, adds
    degree without any co-occurrence. At the defaults the core agents share
    objects repeatedly and so produce genuine 2x2 ``overlap`` structure; small
    settings (for instance ``n_core=1`` or ``core_degree<2``) cannot, since
    overlap needs two agents sharing two objects.

    With the defaults the graph has 25 core agents over 4 core objects plus 40
    rim agents, i.e. ``25 * 3 + 40 = 115`` arcs. The fan counts are then exact
    functions of the degree sequences (a degree-preserving null cannot identify
    them) while ``overlap`` is not, so the same null does carry power for it.

    Parameters are given in full so the example is reproducible from the
    distributed package alone.
    """
    if core_degree > core_objects:
        raise ValueError("core_degree cannot exceed core_objects")
    if min(n_core, core_objects, core_degree, n_rim) < 0:
        raise ValueError("generator sizes must be non-negative")
    rng = np.random.default_rng(seed)
    edges = []
    core = [f"o{j + 1}" for j in range(core_objects)]
    for i in range(n_core):
        # one rng.choice per agent, arcs appended in the order drawn: this
        # ordering is part of the reproducible contract (see module docstring).
        for o in rng.choice(core, size=core_degree, replace=False):
            edges.append((f"c{i}", str(o)))
    for i in range(n_rim):
        edges.append((f"r{i}", f"p{i}"))
    return BipartiteDiGraph(edges)


def synthetic_panel(
    n_panels: int = 4,
    agents_per_panel: int = 40,
    n_objects: int = 24,
    n_categories: int = 3,
    tail_category: int = 0,
    seed: int = 0,
) -> Dict:
    """A panel of attributed directed bipartite graphs.

    Even-indexed panels concentrate on a small ``core`` of objects (high
    co-occurrence / overlap and a distinct portrait); odd-indexed panels are
    dispersed. Returns a dict with the merged ``graph`` (a
    :class:`BipartiteDiGraph` carrying object categories and edge attributes),
    the ``agent_panel`` membership map, the ``panels`` list, and ``object_type``.
    """
    rng = np.random.default_rng(seed)
    objects = [f"obj{j}" for j in range(n_objects)]
    object_type = {o: f"cat{j % n_categories}" for j, o in enumerate(objects)}

    edges = []
    edge_attr: Dict = {}
    agent_panel: Dict = {}

    for p in range(n_panels):
        clustered = (p % 2 == 0)
        core = list(rng.choice(objects, size=6, replace=False))
        for i in range(agents_per_panel):
            a = f"P{p}_A{i}"
            agent_panel[a] = f"panel{p}"
            k = int(rng.integers(3, 6))
            if clustered:
                pool = core
                k = min(k, len(pool))
            else:
                pool = objects
            objs = list(rng.choice(pool, size=k, replace=False))
            for o in objs:
                edges.append((a, o))
                base = rng.normal(0.30, 0.03)
                if object_type[o] == f"cat{tail_category}" and rng.random() < 0.03:
                    base = rng.normal(6.0, 0.3)
                edge_attr[(a, o)] = float(max(base, 1e-3))

    g = BipartiteDiGraph(edges, object_type=object_type, edge_attr=edge_attr)
    return {
        "graph": g,
        "agent_panel": agent_panel,
        "panels": [f"panel{p}" for p in range(n_panels)],
        "object_type": object_type,
    }
