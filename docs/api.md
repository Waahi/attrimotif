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
  on a raw edge list and return the same statistics piecewise. `size3_counts` is
  evaluated in closed form from the degree sequences rather than by enumeration.
- `overlap_count(edges, backend="auto")` computes the same integer two ways.
  `"sparse"` intersects adjacency sets pairwise, costing
  `O(sum_{a<a'} min(d_a, d_a'))` time and `O(|E| + n_A + n_O)` memory.
  `"matrix"` forms the dense `B B^T`, costing `O(n_A^2 n_O)` operations and
  `O(n_A^2)` memory but spending them inside BLAS; it is the faster of the two
  from about a thousand agents upward, by more than an order of magnitude at ten
  thousand, at roughly ten times the memory. `"auto"` takes the dense route once
  there are at least 1000 agents and the matrix is small enough to allocate. The
  binomial terms are summed in `int64`: accumulating them in `float32` silently
  loses exactness above `2**24` and returned counts off by as much as 91.
- `enumerate_size3(edges)` -> `{class: [instance, ...]}`.
- `directed_triadic_census(digraph)` -> `dict`; passthrough to
  `networkx.triadic_census`; raises `TypeError` if not a `networkx.DiGraph`.
- `STATISTICS`: `{name: callable(edges) -> scalar}` registry used by `null_test`.

## Typed (`attrimotif.typed`)

- `stratified_census(g, with_support=False)` -> `{"fan-out":
  {agent_category: count}, "fan-in": {object_category: count}}`; nodes without a
  type appear under `None`. The key is the motif's **hub**, the shared agent for
  `fan-out` and the shared object for `fan-in`, not one side of the graph. With
  `with_support=True` each entry becomes `{"instances": count, "hubs": n}`, where
  `hubs` counts the distinct hub nodes behind the instances: the counts are exact
  either way, but a stratum drawn from one hub supports far less than the same
  count spread over many.
- `phi_distributions(g, reduce=np.mean)` -> `{class: np.ndarray}`; the operator
  Phi, the per-class distribution of `reduce` applied to each instance's two arc
  attributes. Instances with a missing attribute are skipped.
- `tail_summary(x)` -> `dict` (`n`, `mean`, `sd`, `p95`, `p99`, `max`, `skew`).

## Nulls (`attrimotif.nulls`)

- `degree_swap(edges, n_steps, rng, return_count=False, count="proposals")` ->
  swapped edge list (or `(edges, accepted_count)` when `return_count=True`).
  Preserves both-side degrees and produces a simple graph. `n_steps` is a budget
  of **proposed** swaps: a rejected proposal is a step of the chain and its
  self-loop is kept. `count="successes"` restores the 1.0.0 rule, which ran until
  a fixed number of swaps had been accepted and therefore sampled the embedded
  jump chain rather than the uniform distribution; it is retained only for
  reproducing 1.0.0 numbers.
- `null_test(g_or_edges, statistic, n_samples=500, swaps_per=None, seed=0,
  alternative="greater", stat_func=None)` -> `dict` with `statistic`, `observed`,
  `null_mean`, `null_sd`, `z`, `perm_p` (equal to `(r + 1) / (n_samples + 1)`),
  `p_resolution` (the attainable p-value floor `1 / (n_samples + 1)`),
  `identifiable`, `degree_determined`, `null_degenerate`,
  `identifiability_route`, `n_samples`, `swaps_requested`,
  `proposals_per_replicate`, `mean_acceptance_rate` (also returned under its
  1.0.0 name `mean_swap_ratio`), and `note`. Pass `return_samples=True` to get
  the replicate values themselves under `null_samples`, so that plotting the
  null does not require re-implementing the swap loop. Under
  `alternative="two-sided"` the extremeness is measured from the **pooled**
  centre, observed value included; centring on the replicate mean alone breaks
  the exchangeability the `(r + 1) / (R + 1)` estimator rests on and rejects
  above the nominal level at small `R`. Raises `ValueError` for an unknown `alternative` (allowed:
  `greater`, `less`, `two-sided`), for a non-integer or non-positive
  `n_samples`, for a negative or non-integer `swaps_per`, or for a `statistic`
  absent from `STATISTICS` when no `stat_func` is given.

  `identifiable` is **tri-state**. It is `False` when the null cannot
  discriminate the observed value, `True` when it can, and `None` when neither
  answer is supported by the run. Which of these applies, and on what grounds,
  is given by `identifiability_route`:

  | route | meaning | is it a proof? |
  |---|---|---|
  | `registry-proof` | one of the four built-in fan counts, an exact algebraic function of the degree sequences | yes, and it is withheld if `stat_func` is passed or the `STATISTICS` entry has been replaced, since the proof belongs to the function and not to its name |
  | `empirical-degenerate` | every replicate returned exactly the same value, so this run cannot estimate null variability | no: too few replicates, a degree sequence with (almost) no alternative realisation, and a chain that could not move all produce it |
  | `empirical-variable` | at least two replicates differed, which for a deterministic isomorphism-invariant statistic rules out degree-determinacy | it does not establish that the chain has mixed or that the p-value is valid |
  | `undetermined` | fewer than two replicates, or non-finite values; `identifiable` is `None` | n/a |

  `mean_acceptance_rate` (also returned under its 1.0.0 name
  `mean_swap_ratio`) is the fraction of *proposed* swaps that were legal. It is
  not a threshold on its own: it falls with density simply because fewer swaps
  are legal. What matters is how many accepted moves the budget buys per arc,
  `budget * acceptance / |E|`; below roughly one accepted move per arc the graph
  has barely been rewired and the null should not be interpreted, and the remedy
  is to raise the budget. Neither quantity certifies that the chain has reached
  its stationary distribution. Use `swap_convergence` to check sensitivity to
  the chain length.
- `swap_convergence(g_or_edges, statistic="overlap", multipliers=(0.25, 0.5, 1,
  2, 3, 6, 12, 24), n_samples=20, seed=0, stat_func=None)` -> `list` of `dict`
  (`multiplier`, `swaps_requested`, `observed`, `null_mean`, `null_sd`,
  `mean_acceptance_rate`), one per chain length. The chain length used by
  `null_test` is a budget of `12 * |E|` **proposed** swaps per replicate; a flat trace
  across multipliers spanning that value is evidence the reported null is not
  sensitive to it. This is a diagnostic, not a convergence proof: no mixing-time
  or spectral-gap bound is claimed for the bipartite swap chain.
- `is_degree_determined(statistic)` -> `(bool, explanation)`; an exact registry
  lookup over the four built-in fan statistics. It describes a statistic *name*,
  so `null_test` consults it only when no `stat_func` overrides that name.
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
  null_p95, perm_p, p_resolution, n_samples, min_shared}}`. Reassigns agents to
  panels (preserving panel sizes) to build the null. One projection and one
  portrait are computed per panel per assignment and reused across the pairs of
  that assignment; the cache is rebuilt for every replicate. Raises
  `ValueError` for an unknown `alternative` or for `n_samples <= 0`. Returns an
  empty dict when `agent_panel` has fewer than two distinct panel labels (no
  pairs to compare).

## Datasets (`attrimotif.datasets`)

- `planted_tail_example(seed=0)` -> `BipartiteDiGraph` with edge-disjoint fan
  blocks and one planted fan-out tail.
- `clustered_core_rim(seed=0, n_core=25, core_objects=4, core_degree=3,
  n_rim=40)` -> `BipartiteDiGraph`. `n_core` agents each draw `core_degree`
  distinct objects from a shared core of `core_objects`, producing genuine 2x2
  overlap; `n_rim` degree-1 agents attached to private objects add degree
  without co-occurrence. At the defaults: 115 arcs. Raises `ValueError` if
  `core_degree > core_objects`.
- `synthetic_panel(n_panels=4, agents_per_panel=40, n_objects=24,
  n_categories=3, tail_category=0, seed=0)` -> `dict` with `graph`,
  `agent_panel`, `panels`, `object_type`.

The **arc order** these generators emit is part of their reproducible contract:
`degree_swap` draws edges by positional index, so a permuted arc list changes
the swap trajectory, and hence the null mean, sd and z, at a fixed seed even
when the topology is identical.
