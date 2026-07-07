# API reference

All public entry points are exported from the top-level `attrimotif` namespace.
Every function is deterministic under a fixed `seed`.

## Graph container

### `BipartiteDiGraph(edges, agent_type=None, object_type=None, edge_attr=None)`

A directed bipartite graph of agents to objects.

- `edges`: iterable of `(agent, object)` arcs. Duplicates are allowed but collapse
  for all topological counts; aggregate repeated observations into `edge_attr`
  before construction.
- `agent_type`, `object_type`: optional `{node: category}` maps.
- `edge_attr`: optional `{(agent, object): float}` map (one value per arc).

Key methods: `adjacency()` returns `(out, inn)` sets; `agents()`, `objects()`;
`agent_degrees()`, `object_degrees()`; `subgraph_agents(keep)` returns the induced
sub-bipartite-graph; `project_objects(min_shared=2)` returns a `networkx.Graph`
of the object-object co-occurrence (an edge when two objects share at least
`min_shared` agents; isolates removed). `project_objects` raises `ImportError` if
`networkx` is unavailable.

`from_edgelist(edges, **kwargs)` is a convenience constructor.

## Census (`attrimotif.census`)

- `census(g)` -> `dict` with keys `fan-out`, `fan-in`, `fan-out4`, `fan-in4`,
  `overlap` (integer counts). Counts are on the simple bipartite graph; `overlap`
  counts 2x2 subinstances, not maximal bicliques.
- `size3_counts(edges)`, `size4_fan_counts(edges)`, `overlap_count(edges)` operate
  on a raw edge list and return the same statistics piecewise.
- `enumerate_size3(edges)` -> `{class: [instance, ...]}`.
- `directed_triadic_census(digraph)` -> `dict`; passthrough to
  `networkx.triadic_census`; raises `TypeError` if not a `networkx.DiGraph`.
- `STATISTICS`: `{name: callable(edges) -> scalar}` registry used by `null_test`.

## Typed (`attrimotif.typed`)

- `stratified_census(g)` -> `{"fan-out": {agent_category: count}, "fan-in":
  {object_category: count}}`; nodes without a type appear under `None`.
- `phi_distributions(g, reduce=np.mean)` -> `{class: np.ndarray}`; the operator
  Phi, the per-class distribution of `reduce` applied to each instance's two arc
  attributes. Instances with a missing attribute are skipped.
- `tail_summary(x)` -> `dict` (`n`, `mean`, `sd`, `p95`, `p99`, `max`, `skew`).

## Nulls (`attrimotif.nulls`)

- `degree_swap(edges, n_swaps, rng, return_count=False)` -> swapped edge list
  (or `(edges, realized_count)` when `return_count=True`). Preserves both-side
  degrees; produces a simple graph.
- `null_test(g_or_edges, statistic, n_samples=500, swaps_per=None, seed=0,
  alternative="greater", stat_func=None)` -> `dict` with `statistic`, `observed`,
  `null_mean`, `null_sd`, `z`, `perm_p` (equal to `(r + 1) / (n_samples + 1)`),
  `identifiable` (`False` for degree-determined statistics), `mean_swap_ratio`
  (mixing diagnostic), and `note`. Raises `ValueError` for an unknown
  `alternative` (allowed: `greater`, `less`, `two-sided`), for `n_samples <= 0`,
  or for a `statistic` absent from `STATISTICS` when no `stat_func` is given.
- `is_degree_determined(statistic)` -> `(bool, explanation)`; the
  null-identifiability diagnostic.
- `holm_bonferroni(pvals)`, `benjamini_hochberg(pvals)` -> monotone adjusted
  p-values clipped to 1.

## Compare (`attrimotif.compare`)

- `portrait_matrix(graph)` -> the network portrait B-matrix (`numpy.ndarray`).
- `portrait_divergence(g1, g2, backend="builtin")` -> `float` in `[0, 1]` (0 for
  identical graphs; two empty graphs score 0, empty-vs-non-empty scores 1).
  `backend="netrd"` delegates to `netrd.distance.PortraitDivergence` (optional
  dependency); any other value raises `ValueError`. Edge weights are ignored.
- `panel_divergence_matrix(graphs)` -> `{(label_a, label_b): float}` over a
  `{label: networkx graph}` map.
- `panel_permutation_test(g, agent_panel, n_samples=500, min_shared=2, seed=0,
  alternative="greater")` -> `{(label_a, label_b): {observed, null_mean,
  null_p95, perm_p}}`. Reassigns agents to panels (preserving panel sizes) to
  build the null. Raises `ValueError` for an unknown `alternative` or for
  `n_samples <= 0`. Returns an empty dict when `agent_panel` has fewer than two
  distinct panel labels (no pairs to compare).

## Datasets (`attrimotif.datasets`)

- `planted_tail_example(seed=0)` -> `BipartiteDiGraph` with edge-disjoint fan
  blocks and one planted fan-out tail.
- `synthetic_panel(n_panels=4, agents_per_panel=40, n_objects=24,
  n_categories=3, tail_category=0, seed=0)` -> `dict` with `graph`,
  `agent_panel`, `panels`, `object_type`.
