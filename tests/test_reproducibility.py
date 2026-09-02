# -*- coding: utf-8 -*-
"""Reproducibility contract for the shipped generators and the RNG.

Answers, in executable form, the three things a reader needs in order to trust a
reported run: that the synthetic inputs are byte-stable, that the randomness is
seeded deterministically, and that a user parallelising the work knows how to
derive independent streams without silently correlating them.

The checksums are over a canonical serialisation of the generator output, so
they are stable across platforms and Python versions (they do not depend on
dict ordering, float repr, or numpy scalar types).
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

import attrimotif as am


def canonical_digest(graph) -> str:
    """SHA-256 over the arc sequence, node types and edge attributes.

    Arc ORDER is included deliberately: degree_swap draws edges by positional
    index, so two graphs with the same arc set but different order do not give
    the same null at a fixed seed.
    """
    payload = {
        "arcs": [[str(a), str(o)] for a, o in graph.edges],
        "agent_type": {str(k): str(v) for k, v in sorted(graph.agent_type.items(), key=repr)},
        "object_type": {str(k): str(v) for k, v in sorted(graph.object_type.items(), key=repr)},
        # float.hex() is exact and platform-stable, unlike repr rounding
        "edge_attr": [[str(k[0]), str(k[1]), float(v).hex()]
                      for k, v in sorted(graph.edge_attr.items(), key=repr)],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# Regenerate deliberately (and bump the minor version) if a generator changes:
# every number the manuscript reports for that example moves with it.
EXPECTED = {
    "planted_tail_example": "e6500671238c51ebbf8128cc724f85334c65675d979f9cc6abdad2d2ec374214",
    "clustered_core_rim": "8be969d90221c543d967f8d48af781fa8b8d085077b9f216311ee601aed57b7e",
    "synthetic_panel": "ff096483f3b7c60ddb54c4741baf0dc4cbd2e94da8292a11d763bb85d737f23a",
}


def _graphs():
    return {
        "planted_tail_example": am.datasets.planted_tail_example(seed=0),
        "clustered_core_rim": am.datasets.clustered_core_rim(seed=0),
        "synthetic_panel": am.datasets.synthetic_panel(seed=0)["graph"],
    }


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_generator_checksum_is_stable(name):
    assert canonical_digest(_graphs()[name]) == EXPECTED[name]


def test_generators_are_deterministic_across_calls():
    for name, g in _graphs().items():
        again = {
            "planted_tail_example": lambda: am.datasets.planted_tail_example(seed=0),
            "clustered_core_rim": lambda: am.datasets.clustered_core_rim(seed=0),
            "synthetic_panel": lambda: am.datasets.synthetic_panel(seed=0)["graph"],
        }[name]()
        assert canonical_digest(g) == canonical_digest(again)


def test_a_different_seed_gives_a_different_graph():
    assert (canonical_digest(am.datasets.clustered_core_rim(seed=0))
            != canonical_digest(am.datasets.clustered_core_rim(seed=1)))


def test_null_test_is_deterministic_under_a_fixed_seed():
    g = am.datasets.clustered_core_rim(seed=0)
    a = am.null_test(g, "overlap", n_samples=30, seed=7)
    b = am.null_test(g, "overlap", n_samples=30, seed=7)
    assert a == b
    c = am.null_test(g, "overlap", n_samples=30, seed=8)
    assert c["null_mean"] != a["null_mean"]


def test_panel_test_is_deterministic_under_a_fixed_seed():
    pan = am.datasets.synthetic_panel(n_panels=3, agents_per_panel=10, seed=0)
    kw = dict(n_samples=6, min_shared=2, seed=3)
    assert am.panel_permutation_test(pan["graph"], pan["agent_panel"], **kw) == \
           am.panel_permutation_test(pan["graph"], pan["agent_panel"], **kw)


def test_parallel_streams_must_be_derived_with_seed_sequence():
    """The package is single-threaded and every stochastic entry point takes an
    explicit integer seed, so determinism holds by construction. A user running
    replicates in parallel must derive independent streams; reusing one seed
    across workers produces identical, not independent, replicates."""
    naive = [np.random.default_rng(0).integers(0, 2**32) for _ in range(4)]
    assert len(set(naive)) == 1                      # the failure mode
    spawned = [np.random.Generator(np.random.PCG64(s))
               for s in np.random.SeedSequence(0).spawn(4)]
    drawn = [int(r.integers(0, 2**32)) for r in spawned]
    assert len(set(drawn)) == 4                      # the correct pattern
