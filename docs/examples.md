# Examples

Run every example end to end:

```bash
python examples/run_all.py
```

All examples use only synthetic or public data. The three synthetic examples are
deterministic under a fixed seed.

## 1. Operator Phi recovers a tail a count conflates

`example1_planted_tail.py` builds two motif classes with matched size and central
tendency, then plants a single high-value instance in the fan-out class. The
counts, means, and percentiles cannot separate the classes; Phi's per-class
distribution exposes the planted maximum.

## 2. The null-identifiability diagnostic

`example2_null_identifiability.py` shows the fan counts having a null standard
deviation of zero (degree-determined, not identifiable) while the size-4 overlap
count carries a real z-score and permutation p-value.

## 3. The full workflow on a synthetic attributed panel

`example3_synthetic_panel.py` runs the census, the category-stratified census,
the operator Phi, and the panel permutation test on a multi-network panel, and
reports the cross-panel Portrait Divergence matrix with per-pair permutation
p-values.

## 4. A public dataset (MovieLens 100K)

`example4_movielens.py` ingests the MovieLens 100K dataset (downloaded
separately) as a user-to-movie bipartite graph with genre categories, reporting
the census and the genre-stratified fan-in counts. It exits cleanly with
download instructions when the data is absent.
