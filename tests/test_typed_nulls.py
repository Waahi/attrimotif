# -*- coding: utf-8 -*-
"""Operator Phi, stratification, and the degree-preserving null."""
import numpy as np

import attrimotif as am


def test_phi_exposes_planted_tail_that_counts_miss():
    g = am.datasets.planted_tail_example(seed=1)
    counts = am.size3_counts(g.edges)
    phi = am.phi_distributions(g)
    fo, fi = am.tail_summary(phi["fan-out"]), am.tail_summary(phi["fan-in"])

    # class sizes are essentially matched -> a count cannot separate them
    assert abs(counts["fan-out"] - counts["fan-in"]) <= 2
    # central tendency is matched -> mean/percentile do not separate them
    assert abs(fo["mean"] - fi["mean"]) < 0.2
    assert abs(fo["p99"] - fi["p99"]) < 0.2
    # only Phi's full per-class distribution exposes the planted tail
    assert fo["max"] > 5.0
    assert fi["max"] < 1.0


def test_stratified_census_sums_to_total():
    d = am.datasets.synthetic_panel(seed=2)
    g = d["graph"]
    strat = am.stratified_census(g)
    total = am.size3_counts(g.edges)
    assert sum(strat["fan-out"].values()) == total["fan-out"]
    assert sum(strat["fan-in"].values()) == total["fan-in"]


def test_degree_determined_diagnostic():
    assert am.is_degree_determined("fan-out")[0] is True
    assert am.is_degree_determined("fan-in4")[0] is True
    assert am.is_degree_determined("overlap")[0] is False


def test_fan_counts_are_invariant_under_degree_preserving_null():
    # any degree-preserving swap leaves sum C(deg,2) unchanged -> null sd == 0
    d = am.datasets.synthetic_panel(n_panels=2, agents_per_panel=15, seed=3)
    res = am.null_test(d["graph"], "fan-out", n_samples=40, seed=0)
    assert res["null_sd"] < 1e-9
    assert res["identifiable"] is False
    assert res["note"]  # non-empty explanation


def test_overlap_is_identifiable_and_detected_when_planted():
    # A clustered core over a shared object set creates overlap beyond what the
    # degree sequence forces, so the same null does carry power here.
    g = am.datasets.clustered_core_rim(seed=0)
    res = am.null_test(g, "overlap", n_samples=200, seed=0)
    assert res["identifiable"] is True
    assert res["degree_determined"] is False
    assert res["null_degenerate"] is False
    assert res["identifiability_route"] == "empirical-variable"
    assert res["observed"] > res["null_mean"]
    assert res["null_sd"] > 0
    assert 0.0 < res["perm_p"] <= 1.0


def test_overlap_null_is_degenerate_on_a_forced_degree_sequence():
    """A non-degree-determined statistic can still have a degenerate null.

    ``x`` and ``y`` both have degree 3 over exactly three objects, so both are
    forced onto all of them and the rim agents only permute; every realisation
    of this degree sequence has the same overlap count. Swaps are accepted
    (ratio ~ 1) yet the statistic never moves, so the null has no power here
    even though ``overlap`` is not a function of the degree sequence in general.
    The guard must report that without claiming a degree-determinacy proof.
    """
    edges = [("x", o) for o in (1, 2, 3)] + [("y", o) for o in (1, 2, 3)]
    edges += [("z1", 1), ("z2", 2), ("z3", 3)]
    res = am.null_test(edges, "overlap", n_samples=200, seed=0)
    assert res["null_sd"] == 0.0
    # the chain does move: proposals are being accepted, they just never change
    # the statistic, because every realisation of this fibre has the same value
    assert res["mean_acceptance_rate"] > 0.0
    assert res["identifiable"] is False          # ... but the null has no power
    assert res["null_degenerate"] is True
    assert res["degree_determined"] is False     # NOT claimed as a proof
    assert res["identifiability_route"] == "empirical-degenerate"
    # the note must decline to call this a proof
    assert "does not prove" in res["note"]
    assert "exact degree-sequence" not in res["note"]


def test_multiplicity_corrections_are_monotone_and_conservative():
    raw = [0.01, 0.02, 0.04, 0.5]
    for adj in (am.holm_bonferroni(raw), am.benjamini_hochberg(raw)):
        assert np.all(adj >= np.array(raw) - 1e-12)
        assert np.all(adj <= 1.0)


def test_degree_swap_preserves_both_degree_sequences():
    g = am.datasets.synthetic_panel(n_panels=2, agents_per_panel=20, seed=4)["graph"]
    a_before, o_before = g.agent_degrees(), g.object_degrees()
    rng = np.random.default_rng(0)
    swapped, done = am.degree_swap(g.edges, 500, rng, return_count=True)
    h = am.BipartiteDiGraph(swapped)
    assert h.agent_degrees() == a_before
    assert h.object_degrees() == o_before
    assert done >= 0


def test_null_test_rejects_invalid_alternative():
    import pytest

    with pytest.raises(ValueError):
        am.null_test([("a", 1), ("a", 2), ("b", 1)], "overlap", n_samples=5, alternative="bogus")


def test_null_test_reports_swap_ratio():
    d = am.datasets.synthetic_panel(n_panels=2, agents_per_panel=15, seed=3)
    r = am.null_test(d["graph"], "overlap", n_samples=20, seed=0)
    assert 0.0 <= r["mean_swap_ratio"] <= 1.0


def test_null_test_rejects_unknown_statistic():
    import pytest

    with pytest.raises(ValueError):
        am.null_test([("a", 1), ("a", 2), ("b", 1)], "not_a_stat", n_samples=5)


def test_null_test_rejects_nonpositive_n_samples():
    import pytest

    with pytest.raises(ValueError):
        am.null_test([("a", 1), ("a", 2), ("b", 1)], "overlap", n_samples=0)
