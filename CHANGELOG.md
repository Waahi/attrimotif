# Changelog

All notable changes to attrimotif are documented here. The format follows
Keep a Changelog, and the project adheres to Semantic Versioning.

## [Unreleased]

## [1.1.1] - 2026-09

A patch release. No runtime behaviour changes; nothing reported anywhere moves.

### Fixed
- The source distribution shipped `tests/test_version_consistency.py` without
  the `.zenodo.json` it reads, so anyone who downloaded the sdist and ran the
  suite got a `FileNotFoundError` from a test the package advertises as passing.
  `MANIFEST.in` now includes it, and the reader names `MANIFEST.in` when a
  metadata file is absent instead of letting a bare traceback stand for the
  diagnosis. Found by installing the built wheel into a clean environment and
  running the shipped tests against it, which is the only way it shows up: in
  the source tree the file is always there.
- The generator checksums asserted the arc sequence and the float attributes
  under one digest and called it platform-stable. It is not: CI went red on
  macOS for 1.1.0 while all eight Linux and Windows jobs passed, on exactly the
  two generators that draw attributes from `rng.normal`. The arc sequences are
  identical everywhere; the attribute values differ below the ninth decimal
  place. The contract is now split, with the arc sequence asserted across
  platforms and the attributes pinned at nine decimal places, and a failure
  says which half moved.
- `swap_convergence`'s docstring still carried the interpretive criterion the
  1.1.0 release notes withdraw, telling readers that an acceptance rate below 1
  means the chain is rejection-limited. It falls with density because fewer
  swaps are legal; what matters is accepted moves per arc.

### Added
- `benchmarks/bench_scaling.py` and its measured grid, so the scalability
  results reported in the accompanying paper's supplement are reproducible from
  the archived release rather than from a path in the author's working tree.
  The grid now separates the census as the public API runs it, with the overlap
  on `backend="auto"`, from the same census with the overlap forced sparse: the
  first completes every cell and peaks at 2957 MiB, the second stays at or
  below 221 MiB and does not finish the two densest cells.

## [1.1.0] - 2026-09
Revision release prepared in response to the SoftwareX peer review of the
accompanying manuscript. Two changes move results and are the first two
entries below: the null sampler, which moves every reported null result, and the
two-sided permutation p-value, which moves results for anyone who asked for that
alternative. The remaining changes do not: the promoted generator reproduces the
previous example graph arc for arc, and the closed-form census and the portrait
cache were verified bit-identical to the v1.0.0 outputs.

### Fixed (this changes results, please re-run)
- **The fixed-margin null was not sampling uniformly.** `degree_swap` ran until a
  fixed number of swaps had been *accepted*. That discards the rejection
  self-loops and samples the embedded jump chain, whose stationary weight is
  proportional to the number of valid moves out of each state, so graphs that
  happen to admit more swaps were oversampled. On an exhaustively enumerated
  3x3 fibre with both margins (2,1,1) the realised frequencies tracked the
  valid-move counts with correlation 0.997 and a chi-square of 535 on 4 degrees
  of freedom. The budget now counts **proposals**, so a rejected proposal is a
  step of the chain and the self-loops are kept; the same fibre then gives a
  chi-square of 8.5 on 4 degrees of freedom. `count="successes"` reproduces the
  1.0.0 rule and is retained only for reproducing 1.0.0 numbers.

  Every null test therefore moves. On the package's own example the size-4
  overlap goes from z = 25.5 to z = 26.6 at an unchanged p = 1/301, and on
  MovieLens 100K from z = 32.0 to z = 30.8 at an unchanged p = 1/201; the
  qualitative readings are unchanged. `mean_swap_ratio` is now a genuine
  acceptance rate rather than a completion ratio, and is also exposed under the
  clearer name `mean_acceptance_rate`: on MovieLens it reads 0.47, where the
  1.0.0 field read 1.00 by construction.

- **The two-sided permutation p-value was not exact.** Under
  `alternative="two-sided"`, `null_test` and `panel_permutation_test` measured
  extremeness from the mean of the replicates alone, leaving the observed value
  out of the centre. The `(r+1)/(R+1)` estimator is exact only when the ranking
  is a symmetric function of the pooled set, so this broke the finite-R Type-I
  guarantee: calibrated under a true null at R = 19 it rejected at 0.062 against
  a nominal 0.05 on normal data and 0.057 on Poisson counts. Both now centre on
  the pooled set, observed value included, which holds the level (0.050 and
  0.046 in the same calibration). **If you reported a two-sided p-value from
  1.0.0, re-run it.** The one-sided paths are unchanged, and every result in the
  accompanying manuscript uses `alternative="greater"`, so no published number
  moves. The calibration is in
  `benchmarks/validation/calibrate_two_sided.py` and pinned by
  `tests/test_v11_regressions.py`.

### Changed (behaviour-affecting, please read)
- **`null_test` no longer reports `identifiable=True` by default for a
  statistic it cannot assess.** In v1.0.0 the identifiability verdict was a
  lookup on the statistic *name* over four built-in fan counts, so any
  `stat_func` a caller supplied, and any statistic whose null happened to be
  degenerate, was reported as identifiable. The verdict is now reached by one of
  four routes, reported in the new `identifiability_route` field:
  `registry-proof` (exact, and now bound to the canonical shipped callable
  rather than to the name, since `census.STATISTICS` is a mutable public
  mapping), `empirical-degenerate` (all replicates returned the same value,
  which flags the run and is explicitly *not* claimed as proof that the
  statistic is degree-determined), `empirical-variable`, and `undetermined`.
  Consequently `identifiable` is now **tri-state**: `None` when fewer than two
  replicates were drawn or the null returned non-finite values, because neither
  answer is supported. Code that assumed a bool should test the route, or test
  `is True` / `is False` explicitly.
- `null_test` validates `n_samples` and `swaps_per` as integers and rejects
  non-positive values instead of proceeding.
- Degeneracy is decided on the raw statistic values rather than on the float
  array used for the moments, so two integers above 2**53 are no longer
  collapsed into a spurious zero-variance verdict.

### Added
- `datasets.clustered_core_rim()`: the clustered-core-plus-dispersed-rim
  generator used by the null-identifiability example, which was previously
  built inline inside `examples/example2_null_identifiability.py` and so could
  not be reproduced from the distributed package. Arc order is part of its
  contract, because `degree_swap` draws edges by positional index.
- `nulls.swap_convergence()`: the null statistic as a function of chain length
  in multiples of the arc count, with the realized swap ratio at each, for
  checking that a reported null is not sensitive to the number of swaps.
- `examples/example5_movielens_workflow.py`: all four analysis modules on
  MovieLens 100K, with `--only` to run a subset and incremental writing of
  `--out` so an interrupted long run keeps the modules that finished.
- `null_test` additionally returns `p_resolution` (the attainable p-value floor
  `1 / (R + 1)`), `degree_determined`, `null_degenerate`, `n_samples` and
  `swaps_requested`; `panel_permutation_test` additionally returns
  `p_resolution`, `n_samples` and `min_shared`.

### Performance
- `census.size3_counts` computes the fan counts in closed form as sums of
  binomial coefficients of the degrees instead of materialising every instance.
  On MovieLens 100K this is 16.1 s to 0.04 s and 3.05 GiB to 17 MiB of peak
  memory, with identical counts. `enumerate_size3` is unchanged and is still
  what the stratified census and operator Phi consume, so their cost and memory
  profile are unchanged.
- `panel_permutation_test` and `panel_divergence_matrix` compute one portrait
  per panel per assignment instead of recomputing both portraits for every
  pair; with four panels that is a third of the portrait computations. The
  cache is rebuilt for every replicate, so the null distribution is unaffected,
  and the reported values are bit-identical.

### Documentation
- The `nulls` module documents all four identifiability routes and states
  explicitly that only `registry-proof` is a proof, that the empirical routes
  describe the finite sample drawn in a run, and that neither empirical route
  establishes that the swap chain has mixed.
- The realized swap ratio is described as a acceptance-rate diagnostic rather
  than a mixing certificate.
- `overlap_count` no longer states that a degree-preserving null is informative
  for it unconditionally; whether it is remains graph-specific.

## [1.0.0] - 2026-07-07
### Added
- Initial release candidate.
- `BipartiteDiGraph` container for attributed directed bipartite graphs, with
  an object-object co-occurrence projection.
- `census`: size-3 (fan-out, fan-in) and size-4 (fans, 2x2 overlap) bipartite
  motif counts; a directed triadic census passthrough for the unipartite case.
- `typed`: node-category stratified census and the attribute operator Phi
  (per-motif-class attribute distributions).
- `nulls`: degree-preserving bipartite swap, a permutation significance test
  with (r+1)/(R+1) p-values and a realized-swap-ratio diagnostic, the
  null-identifiability diagnostic, and Holm-Bonferroni / Benjamini-Hochberg
  multiplicity corrections.
- `compare`: a dependency-light Portrait Divergence with an optional `netrd`
  backend; a panel divergence matrix; a random-partition panel permutation test.
- `datasets`: deterministic synthetic generators (planted-tail, attributed panel).
- pytest suite, GitHub Actions CI, and four runnable examples via
  `examples/run_all.py`.

[Unreleased]: https://github.com/Waahi/attrimotif/compare/v1.1.1...HEAD
[1.1.1]: https://github.com/Waahi/attrimotif/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/Waahi/attrimotif/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/Waahi/attrimotif/releases/tag/v1.0.0
