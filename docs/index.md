# attrimotif

**An attributed directed bipartite motif inference workflow in Python.**

`attrimotif` joins, in one reproducible workflow for attributed directed
bipartite networks (buyer to brand, tourist to spot, user to item), the four
steps such an analysis usually assembles from separate tools:

1. **Census** — size-3 (fan-out, fan-in) and size-4 (fans, 2x2 overlap)
   bipartite motif counts.
2. **Typed** — node-category stratification and the operator **Phi**, which
   reports the per-motif-class distribution of a numeric edge attribute.
3. **Nulls** — a degree-preserving bipartite null with a **null-identifiability
   diagnostic** that flags statistics the null cannot evaluate.
4. **Compare** — panel-level **Portrait Divergence** with a random-partition
   permutation test.

It is deliberately scoped to attributed *bipartite* motifs on fixed templates,
which keeps it small, permissively licensed (MIT), and dependency-light
(`numpy`, `scipy`, `networkx`; `netrd` optional).

## Install

```bash
pip install attrimotif          # from PyPI
pip install -e .[dev]           # from a checkout, with test/plot extras
```

## Quickstart

```python
import attrimotif as am

d = am.datasets.synthetic_panel(seed=0)
g = d["graph"]

am.census(g)                       # motif counts
am.stratified_census(g)            # counts by node category
am.phi_distributions(g)            # operator Phi: per-class attribute distributions
am.null_test(g, "overlap")         # identifiable -> z, permutation p
am.null_test(g, "fan-out")         # flagged: degree-determined, not identifiable
am.panel_permutation_test(g, d["agent_panel"])   # cross-panel Portrait Divergence
```

See **API reference** and **Examples** for the full surface.
