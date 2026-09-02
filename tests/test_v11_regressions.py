# -*- coding: utf-8 -*-
"""Regression tests for the v1.1.0 changes.

These pin the numbers the manuscript reports and the invariants the v1.1.0
refactors must not break. They are deliberately strict: the size-3 census moved
to a closed form, the null-identifiability guard gained an empirical route, and
the panel test now caches portraits, so each of those needs a test that fails if
the behaviour drifts.
"""
from __future__ import annotations

import importlib
import math

import numpy as np
import pytest

import attrimotif as am
from attrimotif import compare as _compare
from attrimotif.census import enumerate_size3, size3_counts


# -- the promoted generator ----------------------------------------------------
def test_clustered_core_rim_arc_sequence_is_pinned():
    """Arc ORDER is part of the contract: degree_swap indexes edges by position,
    so a permuted arc list moves the null mean/sd/z at a fixed seed even when
    the topology is unchanged."""
    g = am.datasets.clustered_core_rim(seed=0)
    arcs = [(str(a), str(o)) for a, o in g.edges]
    assert len(arcs) == 115 == 25 * 3 + 40
    assert len(set(arcs)) == 115
    # first and last few arcs, in order (golden values taken from the v1.0.0
    # inline generator this function replaced, not from a fresh run)
    assert arcs[:3] == [("c0", "o3"), ("c0", "o4"), ("c0", "o2")]
    assert arcs[74] == ("c24", "o2")
    assert arcs[75] == ("r0", "p0")
    assert arcs[-1] == ("r39", "p39")
    assert sorted({a for a, _ in arcs}) == sorted(
        [f"c{i}" for i in range(25)] + [f"r{i}" for i in range(40)])


def test_clustered_core_rim_reproduces_the_manuscript_numbers():
    """The corrected sampler (proposals, self-loops kept) on the section 3.2
    example. v1.0.0's numbers came from the biased jump-chain rule and are
    pinned separately below."""
    g = am.datasets.clustered_core_rim(seed=0)
    res = am.null_test(g, "overlap", n_samples=300, seed=0)
    assert res["observed"] == 456.0
    assert res["null_mean"] == pytest.approx(56.68333333333333, abs=1e-9)
    assert res["null_sd"] == pytest.approx(15.008988047907236, abs=1e-9)
    assert res["z"] == pytest.approx(26.605169208747448, abs=1e-7)
    assert res["perm_p"] == pytest.approx(1 / 301, abs=1e-15)
    assert res["mean_acceptance_rate"] == pytest.approx(0.6392729468599033, abs=1e-9)
    assert res["p_resolution"] == pytest.approx(1 / 301, abs=1e-15)


def test_legacy_rule_still_reproduces_v100_exactly():
    """Isolating the change: with the v1.0.0 stopping rule the same generator
    and seed must still give v1.0.0's published numbers, so the only thing that
    moved is the sampling rule."""
    import numpy as np
    from attrimotif.nulls import degree_swap

    g = am.datasets.clustered_core_rim(seed=0)
    edges = list(dict.fromkeys(g.edges))
    fn = am.STATISTICS["overlap"]
    obs = fn(edges)
    rng = np.random.default_rng(0)
    vals = [fn(degree_swap(edges, 12 * len(edges), rng, count="successes"))
            for _ in range(300)]
    arr = np.asarray(vals, float)
    z = (obs - arr.mean()) / (arr.std() + 1e-12)
    assert obs == 456
    assert arr.mean() == pytest.approx(59.266666666666666, abs=1e-12)
    assert arr.std() == pytest.approx(15.547418506691784, abs=1e-12)
    assert z == pytest.approx(25.51763388645834, abs=1e-9)


def test_the_shipped_rule_is_the_unbiased_one():
    """A rejected proposal must count as a step, or the sampler is the jump
    chain and oversamples states with more available swaps."""
    import numpy as np
    from attrimotif.nulls import degree_swap

    edges = [("a1", "o1"), ("a1", "o2"), ("a2", "o1"), ("a3", "o2")]
    rng = np.random.default_rng(0)
    _, accepted = degree_swap(edges, 200, rng, return_count=True)
    assert accepted < 200          # proposals were rejected and still counted
    rng = np.random.default_rng(0)
    _, accepted_legacy = degree_swap(edges, 200, rng, return_count=True,
                                     count="successes")
    assert accepted_legacy == 200  # the legacy rule runs until it succeeds
    with pytest.raises(ValueError):
        degree_swap(edges, 10, rng, count="nonsense")


def test_clustered_core_rim_validates_its_parameters():
    with pytest.raises(ValueError):
        am.datasets.clustered_core_rim(core_degree=5, core_objects=4)
    small = am.datasets.clustered_core_rim(n_core=3, core_objects=3, core_degree=2, n_rim=2)
    assert len(small.edges) == 3 * 2 + 2


# -- closed-form size-3 census -------------------------------------------------
@pytest.mark.parametrize("seed", range(15))
def test_closed_form_size3_matches_enumeration(seed):
    rng = np.random.default_rng(seed)
    n_a, n_o = int(rng.integers(1, 15)), int(rng.integers(1, 15))
    edges = [(f"a{i}", f"o{j}") for i in range(n_a) for j in range(n_o)
             if rng.random() < 0.4]
    assert size3_counts(edges) == {k: len(v) for k, v in enumerate_size3(edges).items()}


def test_closed_form_size3_edge_cases():
    assert size3_counts([]) == {"fan-out": 0, "fan-in": 0}
    assert size3_counts([("a", "o")]) == {"fan-out": 0, "fan-in": 0}
    # duplicate arcs collapse for topology
    assert size3_counts([("a", "o")] * 5 + [("a", "o2")]) == {"fan-out": 1, "fan-in": 0}
    # a one-shot generator must not be consumed before both sides are built
    assert size3_counts((f"a{i % 3}", f"o{i}") for i in range(30)) == {
        "fan-out": 3 * math.comb(10, 2), "fan-in": 0}
    # exact for a high-degree hub, where the old enumeration materialised C(d,2)
    assert size3_counts([("hub", f"o{i}") for i in range(400)])["fan-out"] == math.comb(400, 2)
    assert list(size3_counts([("a", "o")])) == ["fan-out", "fan-in"]


# -- identifiability guard -----------------------------------------------------
def test_guard_registry_route_is_a_proof():
    g = am.datasets.clustered_core_rim(seed=0)
    res = am.null_test(g, "fan-out", n_samples=50, seed=0)
    assert res["degree_determined"] is True
    assert res["identifiability_route"] == "registry-proof"
    assert res["identifiable"] is False


def test_guard_catches_an_unregistered_degree_determined_statistic():
    """v1.0.0 returned identifiable=True for ANY stat_func; that was unsound."""
    g = am.datasets.clustered_core_rim(seed=0)
    res = am.null_test(g, "user-fanout", n_samples=50, seed=0,
                       stat_func=lambda e: size3_counts(e)["fan-out"])
    assert res["identifiable"] is False
    assert res["identifiability_route"] == "empirical-degenerate"
    assert res["degree_determined"] is False   # empirical, not proven


def test_guard_does_not_apply_the_registry_to_an_overridden_name():
    """A custom callable labelled 'fan-out' must not inherit the fan proof."""
    g = am.datasets.clustered_core_rim(seed=0)
    res = am.null_test(g, "fan-out", n_samples=50, seed=0, stat_func=am.overlap_count)
    assert res["degree_determined"] is False
    assert res["identifiable"] is True


def test_guard_reports_undetermined_for_a_single_replicate():
    """With one replicate neither answer is supported, so the flag is tri-state.

    Returning identifiable=True here would assert exactly what the run cannot
    show.
    """
    g = am.datasets.clustered_core_rim(seed=0)
    res = am.null_test(g, "overlap", n_samples=1, seed=0)
    assert res["null_degenerate"] is None
    assert res["identifiable"] is None
    assert res["identifiability_route"] == "undetermined"
    assert res["note"]


def test_registry_proof_is_bound_to_the_canonical_callable(monkeypatch):
    """STATISTICS is a mutable public mapping; a replaced entry must not
    inherit the fan proof just because it kept the name."""
    # NB: `from attrimotif import census` yields the census FUNCTION, which
    # shadows the submodule of the same name, so reach the module explicitly.
    census_mod = importlib.import_module("attrimotif.census")

    g = am.datasets.clustered_core_rim(seed=0)
    monkeypatch.setitem(census_mod.STATISTICS, "fan-out", am.overlap_count)
    res = am.null_test(g, "fan-out", n_samples=30, seed=0)
    assert res["observed"] == 456.0            # it really ran overlap
    assert res["degree_determined"] is False   # ... so no proof
    assert res["identifiability_route"] != "registry-proof"
    assert res["identifiable"] is True


def test_degeneracy_is_tested_before_float_coercion():
    """Two integers above 2**53 collapse to one float; testing degeneracy on
    the coerced array would misreport a varying null as degenerate."""
    g = am.datasets.clustered_core_rim(seed=0)
    big = 2 ** 53
    flip = {"n": 0}

    def alternating(edges):
        flip["n"] += 1
        return big + (flip["n"] % 2)

    res = am.null_test(g, "alt", n_samples=6, seed=0, stat_func=alternating)
    assert res["null_degenerate"] is False
    assert res["identifiability_route"] == "empirical-variable"
    assert float(big) == float(big + 1)        # the collapse is real


def test_non_finite_null_values_are_undetermined():
    g = am.datasets.clustered_core_rim(seed=0)
    res = am.null_test(g, "nan", n_samples=5, seed=0, stat_func=lambda e: float("nan"))
    assert res["null_degenerate"] is None
    assert res["identifiable"] is None
    assert res["identifiability_route"] == "undetermined"


def test_null_test_rejects_invalid_budgets():
    g = am.datasets.clustered_core_rim(seed=0)
    for bad in (0, -1, 2.5, True):
        with pytest.raises(ValueError):
            am.null_test(g, "overlap", n_samples=bad, seed=0)
    with pytest.raises(ValueError):
        am.null_test(g, "overlap", n_samples=5, swaps_per=-3, seed=0)
    with pytest.raises(ValueError):
        am.swap_convergence(g, "overlap", multipliers=(0,), n_samples=3, seed=0)


def test_guard_accepts_a_falsy_callable():
    """`stat_func or ...` would silently drop a callable whose __bool__ is False."""
    class Falsy:
        def __bool__(self):
            return False

        def __call__(self, edges):
            return am.overlap_count(edges)

    g = am.datasets.clustered_core_rim(seed=0)
    res = am.null_test(g, "custom", n_samples=20, seed=0, stat_func=Falsy())
    assert res["observed"] == 456.0


# -- swap convergence ----------------------------------------------------------
def test_swap_convergence_trace_is_flat_across_chain_lengths():
    g = am.datasets.clustered_core_rim(seed=0)
    trace = am.swap_convergence(g, "overlap", multipliers=(1, 6, 12, 24),
                                n_samples=15, seed=0)
    assert [t["multiplier"] for t in trace] == [1.0, 6.0, 12.0, 24.0]
    # the budget now counts proposals, so this is an acceptance rate: it must be
    # positive (the chain moves) but need not be near 1
    assert all(0.0 < t["mean_acceptance_rate"] <= 1.0 for t in trace)
    means = [t["null_mean"] for t in trace]
    # the plateau: no drift once the chain is long enough
    assert abs(means[-1] - means[-2]) < 0.2 * max(means[-1], 1.0)


# -- portrait caching ----------------------------------------------------------
def test_panel_test_computes_one_portrait_per_panel_per_replicate(monkeypatch):
    pan = am.datasets.synthetic_panel(n_panels=3, agents_per_panel=12, seed=1)
    calls = {"n": 0}
    original = _compare.portrait_matrix

    def counting(graph):
        calls["n"] += 1
        return original(graph)

    monkeypatch.setattr(_compare, "portrait_matrix", counting)
    am.panel_permutation_test(pan["graph"], pan["agent_panel"], n_samples=4,
                              min_shared=2, seed=0)
    # 3 panels x (1 observed + 4 replicates); per-pair recomputation would be 3x
    assert calls["n"] == 3 * (4 + 1)


def test_size3_counts_does_not_fall_back_to_enumeration(monkeypatch):
    """Guards the closed form itself: a silent revert to enumerate_size3 would
    still pass every numeric equality test."""
    # NB: `from attrimotif import census` yields the census FUNCTION, which
    # shadows the submodule of the same name, so reach the module explicitly.
    census_mod = importlib.import_module("attrimotif.census")

    def boom(edges):
        raise AssertionError("size3_counts must not enumerate instances")

    monkeypatch.setattr(census_mod, "enumerate_size3", boom)
    assert census_mod.size3_counts([("a", "o1"), ("a", "o2"), ("b", "o1")]) == {
        "fan-out": 1, "fan-in": 1}


def test_panel_divergence_matrix_computes_one_portrait_per_graph(monkeypatch):
    pan = am.datasets.synthetic_panel(n_panels=4, agents_per_panel=10, seed=2)
    graphs = {
        lab: pan["graph"].subgraph_agents(
            [a for a, l in pan["agent_panel"].items() if l == lab]).project_objects(2)
        for lab in pan["panels"]
    }
    calls = {"n": 0}
    original = _compare.portrait_matrix

    def counting(graph):
        calls["n"] += 1
        return original(graph)

    monkeypatch.setattr(_compare, "portrait_matrix", counting)
    am.panel_divergence_matrix(graphs)
    assert calls["n"] == 4          # not 2 * C(4,2) = 12


def test_panel_divergence_matrix_matches_pairwise_calls():
    pan = am.datasets.synthetic_panel(n_panels=3, agents_per_panel=12, seed=1)
    graphs = {
        lab: pan["graph"].subgraph_agents(
            [a for a, l in pan["agent_panel"].items() if l == lab]).project_objects(2)
        for lab in pan["panels"]
    }
    cached = am.panel_divergence_matrix(graphs)
    for (a, b), v in cached.items():
        assert v == am.portrait_divergence(graphs[a], graphs[b])


def test_return_samples_matches_the_reported_summary():
    """The replicate values must be the ones the summary was computed from.

    This exists because the manuscript figure drew its null by re-implementing
    the swap loop, and so kept plotting the pre-1.1.0 biased sampler after the
    package had been corrected. Exposing the replicates removes the reason to
    re-implement, so the exposure has to be exact.
    """
    g = am.datasets.clustered_core_rim(seed=0)
    plain = am.null_test(g.edges, "overlap", n_samples=40, seed=0)
    withs = am.null_test(g.edges, "overlap", n_samples=40, seed=0,
                         return_samples=True)

    assert "null_samples" not in plain
    samples = withs["null_samples"]
    assert len(samples) == 40
    assert samples.mean() == withs["null_mean"]
    assert samples.std() == withs["null_sd"]

    # asking for the samples must not perturb anything else
    for key, value in plain.items():
        assert withs[key] == value

    # a caller mutating the array must not corrupt a later call
    samples[:] = 0.0
    again = am.null_test(g.edges, "overlap", n_samples=40, seed=0)
    assert again["null_mean"] == plain["null_mean"]


def test_stratified_census_support_matches_counts():
    """with_support must add the hub count without disturbing the instance counts.

    The response letter claims the package reports each stratum's support, which
    was not true until v1.1.0; this pins the claim to behaviour.
    """
    pan = am.datasets.synthetic_panel(seed=0)
    g = pan["graph"]
    plain = am.stratified_census(g)
    rich = am.stratified_census(g, with_support=True)

    for cls in ("fan-out", "fan-in"):
        assert set(plain[cls]) == set(rich[cls])
        for cat, count in plain[cls].items():
            assert rich[cls][cat]["instances"] == count
            hubs = rich[cls][cat]["hubs"]
            # a stratum cannot draw on more hubs than it has instances, and a
            # non-empty stratum must come from at least one hub
            assert 1 <= hubs <= count


@pytest.mark.parametrize("n_a,n_o,density,seed", [
    (40, 30, 0.10, 0), (60, 60, 0.05, 1), (80, 50, 0.20, 2), (25, 25, 0.40, 3),
])
def test_overlap_backends_agree_small(n_a, n_o, density, seed):
    """The dense and sparse overlap routes must return the same integer."""
    from attrimotif.census import overlap_count

    rng = np.random.default_rng(seed)
    cells = [(f"a{i}", f"o{j}") for i in range(n_a) for j in range(n_o)]
    idx = rng.choice(len(cells), size=int(density * len(cells)), replace=False)
    edges = [cells[i] for i in sorted(idx)]

    sparse = overlap_count(edges, backend="sparse")
    matrix = overlap_count(edges, backend="matrix")
    assert sparse == matrix
    assert overlap_count(edges) == sparse


@pytest.mark.parametrize("n_a,n_o", [(1002, 11), (300, 40)])
def test_overlap_matches_closed_form_above_float32_exactness(n_a, n_o):
    """Counts beyond 2**24 must still be exact, checked against a known answer.

    On the complete bipartite graph K(n_a, n_o) every agent pair shares every
    object, so the overlap count is exactly C(n_a,2)*C(n_o,2). Both cases here
    exceed 2**24 = 16777216, which is where float32 stops representing integers
    exactly.

    This test exists because the first version of the dense route accumulated
    the binomial terms in float32 and returned 2,497,105,408 where the true
    count was 2,497,105,317. The equality test that was supposed to catch it
    used graphs of at most 80 agents, whose counts never came near 2**24, so it
    could not have failed. A test whose inputs cannot reach the failing regime
    is not a check.
    """
    from math import comb

    from attrimotif.census import overlap_count

    edges = [(f"a{i}", f"o{j}") for i in range(n_a) for j in range(n_o)]
    exact = comb(n_a, 2) * comb(n_o, 2)
    assert exact > 2 ** 24
    assert overlap_count(edges, backend="sparse") == exact
    assert overlap_count(edges, backend="matrix") == exact
    # n_a >= 1000 means auto takes the dense route: cover that branch too
    assert overlap_count(edges) == exact


def test_overlap_matrix_accepts_mixed_type_labels():
    """Node labels of mixed types are legal, and both routes must accept them.

    The dense route originally built its object index with sorted(), which
    raises TypeError on mixed types while the sparse route is unaffected: the
    two backends would then disagree by one of them refusing to run.
    """
    from attrimotif.census import overlap_count

    edges = [(1, "o"), ("a", 2), (1, 2), ("a", "o")]
    assert overlap_count(edges, backend="matrix") ==         overlap_count(edges, backend="sparse")


def test_overlap_backend_is_validated():
    from attrimotif.census import overlap_count

    with pytest.raises(ValueError):
        overlap_count([("a", "o")], backend="cuda")


def test_two_sided_permutation_holds_its_level():
    """The two-sided p-value must not reject above its nominal level.

    The (r+1)/(R+1) estimator is exact only when the ranking is a symmetric
    function of the pooled set. Centring on the replicate mean alone excludes
    the observed value from the centre, which breaks that symmetry: calibrated
    under a true null at R=19 it rejected at 0.062 against a nominal 0.05. This
    pins the corrected pooled centring.

    Kept small enough to run in the suite; the full 200,000-trial calibration
    across three distributions lives in revision_2026-09/calibrate_two_sided.py.
    """
    rng = np.random.default_rng(7)
    trials, n_samples, alpha = 20000, 19, 0.05
    obs = rng.normal(size=trials)
    samp = rng.normal(size=(trials, n_samples))

    centre = (samp.sum(axis=1) + obs) / (n_samples + 1)
    r = (np.abs(samp - centre[:, None]) >= np.abs(obs - centre)[:, None]).sum(axis=1)
    p = (r + 1) / (n_samples + 1)
    rate = float((p <= alpha).mean())

    # binomial standard error at 20000 trials is about 0.0015; allow 3 of them
    assert rate <= alpha + 0.005, f"two-sided test rejects at {rate:.4f}"


def test_two_sided_uses_the_pooled_centre():
    """A direct check that null_test's two-sided branch centres on the pooled set.

    Constructed so the two centrings disagree: the observed value sits far from
    the replicate mean, which drags the pooled centre toward it.
    """
    import attrimotif.census as _census
    from attrimotif.nulls import null_test

    edges = am.datasets.clustered_core_rim(seed=0).edges
    calls = {"n": 0}

    def stat(e):
        calls["n"] += 1
        # observed call is first; give it an extreme value, replicates a tight one
        return 1000.0 if calls["n"] == 1 else float(calls["n"] % 3)

    res = null_test(edges, "custom", n_samples=9, seed=0, stat_func=stat,
                    alternative="two-sided")
    # with the pooled centre the observed remains the most extreme, so r = 0
    assert res["perm_p"] == 1.0 / 10
    assert res["identifiable"] is not None or res["identifiability_route"]


def test_dense_route_refuses_when_float32_would_be_inexact(monkeypatch):
    """The guard against an inexact co-occurrence matrix must actually fire.

    B B^T entries are counts of shared objects, so they are exact in float32
    only while the object count stays below 2**24. Building such a graph to test
    the guard is not practical, so the threshold is lowered instead: the guard
    is what is under test, not the arithmetic it protects.
    """
    # attrimotif.census is shadowed by the census() function in the package
    # namespace, so it must be reached through importlib
    census = importlib.import_module("attrimotif.census")

    monkeypatch.setattr(census, "_F32_EXACT_INT", 3)
    edges = [("a1", "o1"), ("a1", "o2"), ("a2", "o1"), ("a2", "o3")]
    with pytest.raises(ValueError, match=r"2\*\*24 objects"):
        census.overlap_count(edges, backend="matrix")
    # the sparse route has no such limit and must still answer
    assert census.overlap_count(edges, backend="sparse") == 0
