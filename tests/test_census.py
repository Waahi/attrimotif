# -*- coding: utf-8 -*-
"""Census correctness against an independent hand computation."""
from math import comb

import attrimotif as am

# A small graph with known degree structure:
#   agents: A(deg 3), B(deg 2), C(deg 1)
#   objects: 1(deg 3), 2(deg 2), 3(deg 1)
EDGES = [("A", 1), ("A", 2), ("A", 3), ("B", 1), ("B", 2), ("C", 1)]


def test_size3_counts_match_hand_computation():
    c = am.size3_counts(EDGES)
    # fan-out = sum C(agent_deg, 2) = C(3,2)+C(2,2)+C(1,2) = 3+1+0
    assert c["fan-out"] == comb(3, 2) + comb(2, 2) + comb(1, 2) == 4
    # fan-in  = sum C(object_deg, 2) = C(3,2)+C(2,2)+C(1,2)
    assert c["fan-in"] == 4


def test_size4_fan_counts_match_formula():
    c = am.size4_fan_counts(EDGES)
    assert c["fan-out4"] == comb(3, 3) + comb(2, 3) + comb(1, 3) == 1
    assert c["fan-in4"] == comb(3, 3) == 1


def test_overlap_count_is_2x2_bicliques():
    # A and B share objects {1,2} -> one 2x2 biclique; other pairs share < 2
    assert am.overlap_count(EDGES) == 1


def test_census_bundles_all_classes():
    c = am.census(am.BipartiteDiGraph(EDGES))
    assert set(c) == {"fan-out", "fan-in", "fan-out4", "fan-in4", "overlap"}
    assert c["overlap"] == 1


def test_duplicate_arcs_do_not_change_topology():
    assert am.size3_counts(EDGES + EDGES) == am.size3_counts(EDGES)


def test_directed_triadic_census_passthrough():
    import networkx as nx

    g = nx.DiGraph([(1, 2), (2, 3), (3, 1)])
    tc = am.directed_triadic_census(g)
    assert sum(tc.values()) == comb(3, 3) == 1
    assert tc["030C"] == 1  # the 3-cycle class
