# -*- coding: utf-8 -*-
"""Synthetic generators and the end-to-end panel workflow."""
import attrimotif as am


def test_synthetic_panel_shape_and_determinism():
    a = am.datasets.synthetic_panel(seed=7)
    b = am.datasets.synthetic_panel(seed=7)
    assert a["graph"].edges == b["graph"].edges  # deterministic under a seed
    g = a["graph"]
    # every agent has a panel; objects carry categories
    assert set(a["agent_panel"]) == set(g.agents())
    assert all(o in a["object_type"] for o in g.objects())


def test_panel_permutation_test_runs_end_to_end():
    d = am.datasets.synthetic_panel(n_panels=3, agents_per_panel=25, seed=5)
    res = am.panel_permutation_test(
        d["graph"], d["agent_panel"], n_samples=40, seed=0
    )
    # one result per unordered panel pair, each a valid permutation p-value
    assert len(res) == 3
    for pair, r in res.items():
        assert 0.0 < r["perm_p"] <= 1.0
        assert 0.0 <= r["observed"] <= 1.0


def test_panel_permutation_test_rejects_invalid_alternative():
    import pytest

    d = am.datasets.synthetic_panel(n_panels=2, agents_per_panel=10, seed=0)
    with pytest.raises(ValueError):
        am.panel_permutation_test(
            d["graph"], d["agent_panel"], n_samples=5, alternative="bogus"
        )


def test_panel_permutation_test_rejects_nonpositive_n_samples():
    import pytest

    d = am.datasets.synthetic_panel(n_panels=2, agents_per_panel=10, seed=0)
    with pytest.raises(ValueError):
        am.panel_permutation_test(d["graph"], d["agent_panel"], n_samples=0)


def test_clustered_and_dispersed_panels_differ():
    d = am.datasets.synthetic_panel(n_panels=2, agents_per_panel=40, seed=1)
    g0 = d["graph"].subgraph_agents(
        [a for a, p in d["agent_panel"].items() if p == "panel0"]
    ).project_objects()
    g1 = d["graph"].subgraph_agents(
        [a for a, p in d["agent_panel"].items() if p == "panel1"]
    ).project_objects()
    assert am.portrait_divergence(g0, g1) > 0.0
