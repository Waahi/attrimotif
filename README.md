# attrimotif

![CI](https://github.com/Waahi/attrimotif/actions/workflows/ci.yml/badge.svg)
![coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)
![license](https://img.shields.io/badge/license-MIT-blue)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21239426.svg)](https://doi.org/10.5281/zenodo.21239426)

**An attributed directed bipartite motif inference workflow in Python.**

`attrimotif` chains the four steps an empirical study of an attributed directed
bipartite network (buyer → brand, tourist → spot, user → movie) usually has to
assemble from scattered tools:

1. **Census** — directed bipartite motif counts (size-3 fan-out / fan-in;
   size-4 fans and the 2×2 overlap / 4-cycle).
2. **Typed** — stratify each motif class by a categorical node type, and apply
   the operator **Φ** to read the *distribution* of a numeric edge attribute
   over motif instances (exposing tails a count collapses).
3. **Nulls** — a degree-preserving bipartite null with a **null-identifiability
   diagnostic** that flags statistics which are exact functions of the degree
   sequence (so the null cannot identify them).
4. **Compare** — panel-level **Portrait Divergence** with a random-partition
   permutation test.

It is **not** a general directed-motif or colored-graph-isomorphism engine: the
scope is deliberately attributed *bipartite* motifs on fixed templates, which is
what the motivating empirical problems need and what keeps the package small,
permissively licensed (MIT), and dependency-light (`numpy`, `scipy`,
`networkx`; `netrd` optional).

## Install

```bash
pip install attrimotif          # from PyPI (planned)
pip install -e .[dev]           # from a checkout, with test/plot extras
```

## Quickstart

```python
import attrimotif as am

d = am.datasets.synthetic_panel(seed=0)
g = d["graph"]

am.census(g)                       # {'fan-out':..,'fan-in':..,'overlap':..,..}
am.stratified_census(g)            # counts by node category
am.phi_distributions(g)            # operator Φ: per-class attribute distributions

am.null_test(g, "overlap")         # identifiable → z, permutation p
am.null_test(g, "fan-out")         # flagged: degree-determined, not identifiable

am.panel_permutation_test(g, d["agent_panel"])   # cross-panel Portrait Divergence
```

> **Note.** A repeated arc collapses for all topological counts; ``edge_attr``
> holds one value per arc, so aggregate repeated observations (e.g. several
> purchases of the same brand) into a single weight before constructing the
> graph.

## Portrait Divergence backend

The bundled implementation is dependency-light. If [`netrd`](https://netrd.readthedocs.io)
is installed, `am.portrait_divergence(g1, g2, backend="netrd")` delegates to its
reference implementation for computing Portrait Divergence. The bundled and `netrd`
backends agree on identity (`PD(G, G) = 0`) and on ordering, but may differ slightly
in how the k-axis is binned across graphs of different sizes.

## Relation to existing tools

`attrimotif` does not claim a new algorithm. Directed motif census
(NetworkX, graph-tool, gtrieScanner), degree-preserving randomization (NetworkX,
graph-tool, xswap), and Portrait Divergence (Bagrow & Bollt 2019; netrd) all
exist separately. `attrimotif`'s contribution is the *integration* of an
attributed **bipartite** motif census, the identifiability diagnostic, operator
Φ, and panel-level Portrait Divergence into one reproducible, installable
workflow. See `CITATION.cff` and the paper for the full comparison.

## Data statement

The package ships **only** synthetic generators (`attrimotif.datasets`) and
reads user-supplied or public-download data. No proprietary records are included
or required; `.gitignore` blocks confidential source files from the repository.

## Tests

```bash
pip install -e .[dev]
pytest
```

## License

MIT — see `LICENSE`.
