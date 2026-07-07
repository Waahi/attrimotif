# Changelog

All notable changes to attrimotif are documented here. The format follows
Keep a Changelog, and the project adheres to Semantic Versioning.

## [Unreleased]

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

[Unreleased]: https://github.com/Waahi/attrimotif/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Waahi/attrimotif/releases/tag/v1.0.0
