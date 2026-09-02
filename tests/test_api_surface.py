# -*- coding: utf-8 -*-
"""Coverage of the public surface that the other suites leave untouched:
alternative-hypothesis branches, argument validation, and the small helpers.
These are all reachable behaviours a user can hit, not dead code."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

import attrimotif as am
from attrimotif.nulls import degree_swap


# -- graph helpers -------------------------------------------------------------
def test_unique_edges_collapses_duplicates_preserving_order():
    g = am.BipartiteDiGraph([("a", "o1"), ("a", "o2"), ("a", "o1"), ("b", "o1")])
    assert g.unique_edges() == [("a", "o1"), ("a", "o2"), ("b", "o1")]


def test_from_edgelist_mirrors_the_constructor():
    g = am.from_edgelist([("a", "o")], object_type={"o": "cat"})
    assert isinstance(g, am.BipartiteDiGraph)
    assert g.object_type["o"] == "cat"


# -- census --------------------------------------------------------------------
def test_directed_triadic_census_requires_a_digraph():
    with pytest.raises(TypeError):
        am.directed_triadic_census(nx.Graph())
    out = am.directed_triadic_census(nx.DiGraph([(1, 2), (2, 3)]))
    assert isinstance(out, dict) and sum(out.values()) > 0


# -- nulls ---------------------------------------------------------------------
def test_degree_swap_on_a_graph_too_small_to_swap():
    edges, done = degree_swap([("a", "o")], 10, np.random.default_rng(0), return_count=True)
    assert done == 0 and edges == [("a", "o")]
    assert degree_swap([], 5, np.random.default_rng(0)) == []


def test_null_test_rejects_unknown_alternative_and_statistic():
    g = am.datasets.clustered_core_rim(seed=0)
    with pytest.raises(ValueError):
        am.null_test(g, "overlap", n_samples=5, alternative="sideways")
    with pytest.raises(ValueError):
        am.null_test(g, "no-such-statistic", n_samples=5)


def test_null_test_less_and_two_sided_branches():
    g = am.datasets.clustered_core_rim(seed=0)
    for alt in ("less", "two-sided"):
        res = am.null_test(g, "overlap", n_samples=20, seed=0, alternative=alt)
        assert 0.0 < res["perm_p"] <= 1.0


def test_swap_convergence_validates_its_arguments():
    g = am.datasets.clustered_core_rim(seed=0)
    with pytest.raises(ValueError):
        am.swap_convergence(g, "overlap", n_samples=0)
    with pytest.raises(ValueError):
        am.swap_convergence(g, "nope", n_samples=3)
    with pytest.raises(ValueError):
        am.swap_convergence(g, "overlap", multipliers=(), n_samples=3)


# -- datasets ------------------------------------------------------------------
def test_clustered_core_rim_rejects_negative_sizes():
    with pytest.raises(ValueError):
        am.datasets.clustered_core_rim(n_rim=-1)


# -- typed ---------------------------------------------------------------------
def test_phi_skips_instances_with_a_missing_attribute():
    # the (b, o2) arc carries no attribute, so its instances are skipped
    g = am.BipartiteDiGraph(
        [("a", "o1"), ("a", "o2"), ("b", "o1"), ("b", "o2")],
        edge_attr={("a", "o1"): 1.0, ("a", "o2"): 2.0, ("b", "o1"): 3.0})
    phi = am.phi_distributions(g)
    assert len(phi["fan-out"]) == 1          # only agent a has both attributes
    assert phi["fan-out"][0] == pytest.approx(1.5)


def test_tail_summary_on_an_empty_distribution():
    out = am.tail_summary(np.array([]))
    assert out["n"] == 0
    assert all(np.isnan(out[k]) for k in ("mean", "sd", "p95", "p99", "max"))


# -- compare -------------------------------------------------------------------
def test_portrait_divergence_rejects_an_unknown_backend():
    g1, g2 = nx.path_graph(3), nx.star_graph(3)
    with pytest.raises(ValueError):
        am.portrait_divergence(g1, g2, backend="nope")


def test_panel_permutation_test_less_and_two_sided_branches():
    pan = am.datasets.synthetic_panel(n_panels=2, agents_per_panel=12, seed=4)
    for alt in ("less", "two-sided"):
        res = am.panel_permutation_test(pan["graph"], pan["agent_panel"],
                                        n_samples=4, min_shared=2, seed=0,
                                        alternative=alt)
        assert res and all(0.0 < v["perm_p"] <= 1.0 for v in res.values())


def test_panel_permutation_test_rejects_bad_arguments():
    pan = am.datasets.synthetic_panel(n_panels=2, agents_per_panel=8, seed=0)
    with pytest.raises(ValueError):
        am.panel_permutation_test(pan["graph"], pan["agent_panel"], alternative="nope")
    with pytest.raises(ValueError):
        am.panel_permutation_test(pan["graph"], pan["agent_panel"], n_samples=0)


def test_every_public_callable_is_fully_annotated():
    """Section 2.1 of the paper claims every public entry point carries type hints.

    A reader can check that claim with one call to inspect.signature, and when
    the second review round did, ten of them were unannotated. This pins the
    claim to the code so it cannot quietly become false again.
    """
    import inspect

    import attrimotif as am

    gaps = {}
    for name in sorted(n for n in dir(am) if not n.startswith("_")):
        obj = getattr(am, name)
        if not callable(obj) or inspect.isclass(obj):
            continue
        try:
            sig = inspect.signature(obj)
        except (ValueError, TypeError):        # pragma: no cover - builtins
            continue
        missing = [p for p, v in sig.parameters.items()
                   if v.annotation is inspect.Parameter.empty]
        if sig.return_annotation is inspect.Signature.empty:
            missing.append("->return")
        if missing:
            gaps[name] = missing
    assert not gaps, f"unannotated public callables: {gaps}"


def test_every_public_callable_has_a_docstring():
    """The same sentence promises a docstring stating inputs, outputs and limits."""
    import inspect

    import attrimotif as am

    bare = [n for n in sorted(x for x in dir(am) if not x.startswith("_"))
            if callable(getattr(am, n))
            and not (inspect.getdoc(getattr(am, n)) or "").strip()]
    assert not bare, f"public callables without a docstring: {bare}"
