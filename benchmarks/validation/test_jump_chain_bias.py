# -*- coding: utf-8 -*-
"""Is the shipped swap sampler uniform on the fixed-margin fibre?

Codex raised this in review of the revision: `degree_swap` runs until a fixed
number of SUCCESSFUL swaps, which removes the rejection self-loops and so
samples the embedded jump chain. The jump chain of a reversible chain with
uniform stationary law has stationary weight proportional to the escape
probability, i.e. to the number of valid moves out of each state. If different
realisations of the same degree sequence admit different numbers of valid
swaps, the sampler is biased towards the better-connected ones.

This settles it empirically rather than by argument: enumerate a small fibre
exhaustively, run both samplers from a fixed start, and compare the realised
frequencies with uniform.

    python benchmarks/validation/test_jump_chain_bias.py
"""
from __future__ import annotations

import collections
import itertools
import os
import sys

import numpy as np

# runs from a checkout (benchmarks/validation/ -> src/) or against the installed
# package, whichever import resolves first
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
from attrimotif.nulls import degree_swap  # noqa: E402


# ---------------------------------------------------------------- fibre -----
def enumerate_fibre(row_sums, col_sums):
    """All 0/1 matrices with the given row and column sums."""
    n_r, n_c = len(row_sums), len(col_sums)
    out = []

    def rec(r, remaining_cols, rows):
        if r == n_r:
            if all(c == 0 for c in remaining_cols):
                out.append([row[:] for row in rows])
            return
        for cols in itertools.combinations(range(n_c), row_sums[r]):
            if any(remaining_cols[c] == 0 for c in cols):
                continue
            nxt = list(remaining_cols)
            row = [0] * n_c
            for c in cols:
                nxt[c] -= 1
                row[c] = 1
            rec(r + 1, nxt, rows + [row])

    rec(0, list(col_sums), [])
    return out


def mat_to_edges(m):
    return [(f"a{i}", f"o{j}") for i, row in enumerate(m) for j, v in enumerate(row) if v]


def edges_to_key(edges, n_r, n_c):
    s = set(edges)
    return tuple(1 if (f"a{i}", f"o{j}") in s else 0
                 for i in range(n_r) for j in range(n_c))


def valid_move_count(edges):
    """Ordered (i, j) edge-index pairs whose swap is legal: the escape rate."""
    E = list(edges)
    S = set(E)
    n = 0
    for i in range(len(E)):
        for j in range(len(E)):
            if i == j:
                continue
            a1, o1 = E[i]
            a2, o2 = E[j]
            if a1 == a2 or o1 == o2:
                continue
            if (a1, o2) in S or (a2, o1) in S:
                continue
            n += 1
    return n


# ------------------------------------------------- the two stopping rules ---
def sample_fixed_successes(edges, n_swaps, rng):
    """What the package ships: stop after n_swaps ACCEPTED swaps."""
    # count="successes" is the v1.0.0 rule. Since v1.1.0 the default counts
    # proposals, so omitting this silently turns the control into the treatment
    # and the comparison reports nothing.
    return degree_swap(edges, n_swaps, rng, count="successes")


def sample_fixed_proposals(edges, n_proposals, rng):
    """The self-loop-preserving variant: stop after n_proposals DRAWS,
    whether or not they were accepted."""
    E = [tuple(e) for e in dict.fromkeys((a, o) for a, o in edges)]
    S = set(E)
    m = len(E)
    if m < 2:
        return E
    for _ in range(n_proposals):
        i, j = int(rng.integers(0, m)), int(rng.integers(0, m))
        if i == j:
            continue
        a1, o1 = E[i]
        a2, o2 = E[j]
        if a1 == a2 or o1 == o2:
            continue
        if (a1, o2) in S or (a2, o1) in S:
            continue
        S.discard((a1, o1)); S.discard((a2, o2))
        S.add((a1, o2)); S.add((a2, o1))
        E[i] = (a1, o2); E[j] = (a2, o1)
    return E


def chi_square(observed, expected):
    return float(sum((o - e) ** 2 / e for o, e in zip(observed, expected)))


def run_case(name, rows, cols, n_draws=40000, swaps=40, proposals=400):
    fibre = enumerate_fibre(rows, cols)
    keys = [tuple(v for row in m for v in row) for m in fibre]
    n_r, n_c = len(rows), len(cols)
    print(f"\n=== {name}: rows={rows} cols={cols} ===")
    print(f"  fibre size: {len(fibre)} realisations")
    moves = [valid_move_count(mat_to_edges(m)) for m in fibre]
    print(f"  valid ordered moves per realisation: min={min(moves)} max={max(moves)} "
          f"distinct={sorted(set(moves))}")
    if len(set(moves)) == 1:
        print("  (escape rate is constant here, so the two rules cannot differ)")

    start = mat_to_edges(fibre[0])
    results = {}
    for label, fn, budget in (("fixed-successes (shipped)", sample_fixed_successes, swaps),
                              ("fixed-proposals (self-loops kept)", sample_fixed_proposals, proposals)):
        rng = np.random.default_rng(12345)
        counts = collections.Counter()
        for _ in range(n_draws):
            e = fn(start, budget, rng)
            counts[edges_to_key(e, n_r, n_c)] += 1
        obs = [counts.get(k, 0) for k in keys]
        exp = [n_draws / len(fibre)] * len(fibre)
        chi = chi_square(obs, exp)
        # correlation between realised frequency and the escape rate
        corr = float(np.corrcoef(obs, moves)[0, 1]) if len(set(moves)) > 1 else float("nan")
        results[label] = (obs, chi, corr)
        print(f"  {label}")
        print(f"     frequencies: {obs}")
        print(f"     expected under uniform: {exp[0]:.0f} each")
        print(f"     chi-square vs uniform: {chi:.1f}  (df={len(fibre) - 1})")
        if len(set(moves)) > 1:
            print(f"     corr(frequency, valid-move count): {corr:+.3f}")
    return results


if __name__ == "__main__":
    # Codex's counterexample: both margins (2,1,1)
    run_case("3x3, margins (2,1,1)/(2,1,1)", [2, 1, 1], [2, 1, 1])
    # a fibre with more spread in the escape rate
    run_case("4x4, margins (2,2,1,1)/(2,2,1,1)", [2, 2, 1, 1], [2, 2, 1, 1])
    # a sparser, more realistic shape
    run_case("5x4, margins (2,2,1,1,1)/(2,2,2,1)", [2, 2, 1, 1, 1], [2, 2, 2, 1])
    print("\nInterpretation: a chi-square far above the degrees of freedom, together")
    print("with a positive correlation between realised frequency and the valid-move")
    print("count, is the signature of the jump-chain bias.")
