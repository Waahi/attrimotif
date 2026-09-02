# -*- coding: utf-8 -*-
"""Reproducibility contract for the shipped generators and the RNG.

Answers, in executable form, the three things a reader needs in order to trust a
reported run: that the synthetic inputs are stable, that the randomness is
seeded deterministically, and that a user parallelising the work knows how to
derive independent streams without silently correlating them.

The contract is split in two, because the two halves are not equally strong and
the earlier single checksum hid that.

**Arc sequence and node types: asserted across platforms.** This is the half
that matters for inference. ``degree_swap`` draws edges by positional index, so
two graphs with the same arc set in a different order do not give the same null
at a fixed seed, and every null result in the paper depends on this sequence.
These generators build their structure from ``rng.choice`` and integer
arithmetic, which are bit-stable wherever numpy runs.

**Edge attribute values: asserted, but they are floats from ``rng.normal``.**
The ziggurat method makes accept/reject decisions on floating-point
comparisons, so a one-ulp difference in a build can send the whole stream
elsewhere. This is not hypothetical: v1.1.0's CI went red on macOS while Linux
and Windows passed, on exactly the two generators that draw attributes, and not
on the one that does not. Values are digested rounded to 9 decimal places, so
genuine changes are caught while last-bit noise is not.

If the attribute test fails while the arc test passes, the structure is intact
and only the float stream moved; the paper's null results are unaffected and
only the operator Phi summaries would shift. If the ARC test fails, that is a
real cross-platform reproducibility break and the reported null results do not
transfer to that platform.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

import attrimotif as am


def _graphs():
    return {
        "planted_tail_example": am.datasets.planted_tail_example(seed=0),
        "clustered_core_rim": am.datasets.clustered_core_rim(seed=0),
        "synthetic_panel": am.datasets.synthetic_panel(seed=0)["graph"],
    }


def arc_digest(graph) -> str:
    """SHA-256 over the arc sequence and the node types. No floats."""
    payload = {
        "arcs": [[str(a), str(o)] for a, o in graph.edges],
        "agent_type": {str(k): str(v)
                       for k, v in sorted(graph.agent_type.items(), key=repr)},
        "object_type": {str(k): str(v)
                        for k, v in sorted(graph.object_type.items(), key=repr)},
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def attr_digest(graph) -> str:
    """SHA-256 over the edge attributes, rounded to 9 decimal places."""
    items = sorted((str(k[0]), str(k[1]), round(float(v), 9))
                   for k, v in graph.edge_attr.items())
    return hashlib.sha256(json.dumps(items, separators=(",", ":")).encode()).hexdigest()


# Regenerate deliberately (and bump the minor version) if a generator changes:
# every number the manuscript reports for that example moves with it.
EXPECTED_ARCS = {
    "planted_tail_example": "c2635ec7b90c76cbabe74b759297e18f8a80c720e49b2283ab7aad8a2d153fd1",
    "clustered_core_rim": "01a561489387fcaa0738976c94f9920d9188342c875cdd24a0e09c22a4b95e6a",
    "synthetic_panel": "39942d335229f0915cd1d319823c9a9788475e625672cf57700996194f2e1873",
}

EXPECTED_ATTRS = {
    "planted_tail_example": "2a07f768518d04ffcf1409e005bbdbd43d7daad43006887b326074bc0b211bad",
    "clustered_core_rim": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    "synthetic_panel": "be350d9859b7457efad01d78ab955ab45a2cc9d2b5cfa964900bcd2dff5200c1",
}


@pytest.mark.parametrize("name", sorted(EXPECTED_ARCS))
def test_generator_arc_sequence_is_platform_stable(name):
    """The half the paper's null results actually depend on."""
    g = _graphs()[name]
    assert arc_digest(g) == EXPECTED_ARCS[name], (
        f"{name}: the arc sequence differs from the pinned one. This is a "
        f"cross-platform reproducibility break, not a rounding difference: the "
        f"null draws edges by index, so every reported z and p for this graph "
        f"is conditional on this sequence.")


@pytest.mark.parametrize("name", sorted(EXPECTED_ATTRS))
def test_generator_attributes_reproduce(name):
    """Floats from rng.normal, digested at 9 dp so ulp noise does not trip it."""
    g = _graphs()[name]
    if arc_digest(g) != EXPECTED_ARCS[name]:
        pytest.skip("arc sequence already differs; the attribute check is "
                    "meaningless until that is resolved")
    assert attr_digest(g) == EXPECTED_ATTRS[name], (
        f"{name}: the arc sequence matches but the attribute values do not. "
        f"The structure is intact, so null results are unaffected; what moved "
        f"is the rng.normal stream, which feeds the operator Phi summaries.")


def test_attributes_are_finite_and_in_range():
    """A cheap invariant that holds whatever the float stream does."""
    for name, g in _graphs().items():
        vals = np.asarray(list(g.edge_attr.values()), dtype=float)
        if vals.size == 0:
            continue
        assert np.all(np.isfinite(vals)), name
        assert np.all(vals > 0), name


def test_seeded_runs_are_bit_identical():
    """Same seed, same process, same numbers: the determinism the paper claims."""
    g = am.datasets.clustered_core_rim(seed=0)
    a = am.null_test(g.edges, "overlap", n_samples=25, seed=7)
    b = am.null_test(g.edges, "overlap", n_samples=25, seed=7)
    for key in ("z", "null_mean", "null_sd", "perm_p", "mean_acceptance_rate"):
        assert a[key] == b[key], key


def test_different_seeds_give_different_draws():
    """The negative control: if these matched, the seed would not be doing anything."""
    g = am.datasets.clustered_core_rim(seed=0)
    a = am.null_test(g.edges, "overlap", n_samples=25, seed=7)
    b = am.null_test(g.edges, "overlap", n_samples=25, seed=8)
    assert a["null_mean"] != b["null_mean"]


def test_spawned_streams_are_independent_not_identical():
    """The documented way to parallelise: SeedSequence.spawn, not seed + i.

    Reusing one seed across workers yields identical replicates, which looks
    like a converged null and is not one.
    """
    parent = np.random.SeedSequence(12345)
    children = parent.spawn(4)
    firsts = [np.random.default_rng(c).random() for c in children]
    assert len(set(firsts)) == 4
