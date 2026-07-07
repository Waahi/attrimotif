# -*- coding: utf-8 -*-
"""Attributed directed bipartite graph container.

Directionality is fixed **agent -> object** (the empirical convention shared by
the motivating domains: buyer->brand, tourist->spot, user->movie). Nodes on
either side may carry a categorical type; edges may carry one numeric attribute
(e.g. spend, rating, ROI). All motif/null routines in :mod:`attrimotif` consume
either a :class:`BipartiteDiGraph` or its raw ``edges`` list.

Design notes (see the package's methodological-specification docs):
* Structure is defined by the *set* of (agent, object) arcs; a repeated arc is
  collapsed for topology but its edge attribute is still available via
  :attr:`edge_attr`.
* :meth:`project_objects` builds the object-object co-occurrence graph used for
  Portrait-Divergence comparison: an undirected edge is placed between two
  objects sharing at least ``min_shared`` common agents.
"""
from __future__ import annotations

from itertools import combinations
from typing import Dict, Hashable, Iterable, List, Optional, Tuple

try:  # networkx is only needed for projection / Portrait Divergence
    import networkx as nx
except ImportError:  # pragma: no cover
    nx = None

Node = Hashable
Edge = Tuple[Node, Node]


class BipartiteDiGraph:
    """A directed bipartite graph of agents -> objects with optional attributes.

    Parameters
    ----------
    edges : iterable of (agent, object)
        Directed arcs. Duplicates are permitted (used for the edge attribute)
        but collapse to a single arc for all topological counts.
    agent_type, object_type : mapping, optional
        Categorical label per agent / object node.
    edge_attr : mapping (agent, object) -> float, optional
        One numeric attribute per arc, consumed by the operator ``phi``.
    """

    def __init__(
        self,
        edges: Iterable[Edge],
        agent_type: Optional[Dict[Node, Hashable]] = None,
        object_type: Optional[Dict[Node, Hashable]] = None,
        edge_attr: Optional[Dict[Edge, float]] = None,
    ) -> None:
        self.edges: List[Edge] = [(a, o) for (a, o) in edges]
        self.agent_type: Dict[Node, Hashable] = dict(agent_type or {})
        self.object_type: Dict[Node, Hashable] = dict(object_type or {})
        self.edge_attr: Dict[Edge, float] = dict(edge_attr or {})

    # -- basic structure -----------------------------------------------------
    def unique_edges(self) -> List[Edge]:
        seen, out = set(), []
        for e in self.edges:
            if e not in seen:
                seen.add(e)
                out.append(e)
        return out

    def adjacency(self) -> Tuple[Dict[Node, set], Dict[Node, set]]:
        """Return ``(out, inn)`` where ``out[a]`` is the set of objects of agent
        ``a`` and ``inn[o]`` is the set of agents of object ``o``."""
        out: Dict[Node, set] = {}
        inn: Dict[Node, set] = {}
        for a, o in self.edges:
            out.setdefault(a, set()).add(o)
            inn.setdefault(o, set()).add(a)
        return out, inn

    def agents(self) -> List[Node]:
        return sorted({a for a, _ in self.edges}, key=repr)

    def objects(self) -> List[Node]:
        return sorted({o for _, o in self.edges}, key=repr)

    def agent_degrees(self) -> Dict[Node, int]:
        out, _ = self.adjacency()
        return {a: len(v) for a, v in out.items()}

    def object_degrees(self) -> Dict[Node, int]:
        _, inn = self.adjacency()
        return {o: len(v) for o, v in inn.items()}

    def subgraph_agents(self, keep) -> "BipartiteDiGraph":
        """Return the induced sub-bipartite-graph on a subset of agents."""
        keep = set(keep)
        e = [(a, o) for (a, o) in self.edges if a in keep]
        ea = {k: v for k, v in self.edge_attr.items() if k[0] in keep}
        return BipartiteDiGraph(e, self.agent_type, self.object_type, ea)

    # -- projection ----------------------------------------------------------
    def project_objects(self, min_shared: int = 2):
        """Object-object co-occurrence graph (undirected).

        An edge ``(o1, o2)`` is created when ``o1`` and ``o2`` share at least
        ``min_shared`` common agents; its weight is the number of shared agents.
        Isolated objects are dropped. Object ``type`` is copied to node data.
        Returns a :class:`networkx.Graph`.
        """
        if nx is None:  # pragma: no cover
            raise ImportError("networkx is required for project_objects()")
        _, inn = self.adjacency()
        objs = self.objects()
        g = nx.Graph()
        for o in objs:
            g.add_node(o, type=self.object_type.get(o))
        for o1, o2 in combinations(objs, 2):
            s = len(inn.get(o1, set()) & inn.get(o2, set()))
            if s >= min_shared:
                g.add_edge(o1, o2, weight=s)
        g.remove_nodes_from(list(nx.isolates(g)))
        return g

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"BipartiteDiGraph(agents={len(self.agents())}, "
            f"objects={len(self.objects())}, arcs={len(self.unique_edges())})"
        )


def from_edgelist(edges, **kwargs) -> BipartiteDiGraph:
    """Convenience constructor mirroring :class:`BipartiteDiGraph`."""
    return BipartiteDiGraph(edges, **kwargs)
