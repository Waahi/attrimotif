# -*- coding: utf-8 -*-
"""attrimotif: an attributed directed bipartite motif inference workflow.

A dependency-light Python workflow that chains a node-attributed directed
bipartite motif census, a degree-preserving null with a null-identifiability
diagnostic, and panel-level Portrait Divergence permutation testing.

Not a general directed-motif or colored-graph-isomorphism engine: the v1.0
scope is attributed *bipartite* motifs on fixed size-3/size-4 templates.
"""
from __future__ import annotations

from . import datasets
from .census import (
    STATISTICS,
    census,
    directed_triadic_census,
    enumerate_size3,
    overlap_count,
    size3_counts,
    size4_fan_counts,
)
from .compare import (
    panel_divergence_matrix,
    panel_permutation_test,
    portrait_divergence,
    portrait_matrix,
)
from .graph import BipartiteDiGraph, from_edgelist
from .nulls import (
    benjamini_hochberg,
    degree_swap,
    holm_bonferroni,
    is_degree_determined,
    null_test,
)
from .typed import phi_distributions, stratified_census, tail_summary

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # graph
    "BipartiteDiGraph",
    "from_edgelist",
    # census
    "census",
    "enumerate_size3",
    "size3_counts",
    "size4_fan_counts",
    "overlap_count",
    "directed_triadic_census",
    "STATISTICS",
    # typed
    "stratified_census",
    "phi_distributions",
    "tail_summary",
    # nulls
    "degree_swap",
    "null_test",
    "is_degree_determined",
    "holm_bonferroni",
    "benjamini_hochberg",
    # compare
    "portrait_matrix",
    "portrait_divergence",
    "panel_divergence_matrix",
    "panel_permutation_test",
    # datasets
    "datasets",
]
