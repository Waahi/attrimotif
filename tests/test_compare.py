# -*- coding: utf-8 -*-
"""Portrait Divergence properties and optional netrd parity."""
import importlib.util

import networkx as nx
import pytest

import attrimotif as am

_HAS_NETRD = importlib.util.find_spec("netrd") is not None


def _graphs():
    path = nx.path_graph(8)
    star = nx.star_graph(7)
    complete = nx.complete_graph(8)
    return path, star, complete


def test_pd_identity_is_zero():
    for g in _graphs():
        assert am.portrait_divergence(g, g) < 1e-9


def test_pd_is_symmetric_and_in_unit_interval():
    path, star, complete = _graphs()
    for a, b in [(path, star), (path, complete), (star, complete)]:
        d1 = am.portrait_divergence(a, b)
        d2 = am.portrait_divergence(b, a)
        assert abs(d1 - d2) < 1e-9
        assert 0.0 <= d1 <= 1.0


def test_pd_separates_distinct_graphs():
    path, _, complete = _graphs()
    assert am.portrait_divergence(path, complete) > 0.0


def test_panel_divergence_matrix_keys():
    path, star, complete = _graphs()
    m = am.panel_divergence_matrix({"p": path, "s": star, "c": complete})
    assert set(m) == {("p", "s"), ("p", "c"), ("s", "c")}


def test_pd_empty_graph_identity():
    e1, e2 = nx.Graph(), nx.Graph()
    assert am.portrait_divergence(e1, e2) == 0.0  # both empty: identical
    assert am.portrait_divergence(e1, nx.path_graph(4)) == 1.0  # empty vs non-empty


@pytest.mark.skipif(not _HAS_NETRD, reason="netrd not installed")
def test_builtin_pd_agrees_with_netrd_backend():
    # The builtin follows the Bagrow-Bollt pair weighting; it need not match
    # netrd's k-binning to the last digit, but the two must agree on identity,
    # stay in [0, 1], and rank graph pairs the same way.
    path, star, complete = _graphs()
    for g in (path, star, complete):
        assert am.portrait_divergence(g, g, backend="netrd") < 1e-9
    near_ref = am.portrait_divergence(path, star, backend="netrd")
    far_ref = am.portrait_divergence(path, complete, backend="netrd")
    near_b = am.portrait_divergence(path, star, backend="builtin")
    far_b = am.portrait_divergence(path, complete, backend="builtin")
    assert (far_ref > near_ref) == (far_b > near_b)  # same ordering
    assert abs(far_b - far_ref) < 0.25  # loose numeric agreement
