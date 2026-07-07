# -*- coding: utf-8 -*-
"""Example 4 - real public data (MovieLens 100K).

Demonstrates the workflow on a genuine public dataset: the user -> movie
bipartite graph is directed, movies carry a categorical genre attribute, and
ratings are timestamped so panels can be cut by period.

The MovieLens 100K data is NOT bundled (respecting its terms). Download it once:

    https://files.grouplens.org/datasets/movielens/ml-100k.zip

and unzip so that ``examples/data/ml-100k/u.data`` and ``u.item`` exist. If the
data is absent this script prints instructions and exits cleanly (so it is safe
to include in an automated run without network access).

Reference: Harper & Konstan (2015), "The MovieLens Datasets", ACM TiiS 5(4),
DOI 10.1145/2827872.
"""
from __future__ import annotations

import os

import attrimotif as am

DATA = os.path.join(os.path.dirname(__file__), "data", "ml-100k")
GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def _load(period_split_year=1998):
    u_item = os.path.join(DATA, "u.item")
    u_data = os.path.join(DATA, "u.data")
    if not (os.path.exists(u_item) and os.path.exists(u_data)):
        return None

    object_type = {}
    with open(u_item, encoding="latin-1") as f:
        for line in f:
            p = line.rstrip("\n").split("|")
            mid = f"m{p[0]}"
            flags = [int(x) for x in p[-19:]]
            primary = next((GENRES[i] for i, v in enumerate(flags) if v and i > 0), "unknown")
            object_type[mid] = primary

    edges, edge_attr = [], {}
    with open(u_data, encoding="latin-1") as f:
        for line in f:
            uid, mid, rating, ts = line.split("\t")
            a, o = f"u{uid}", f"m{mid}"
            edges.append((a, o))
            edge_attr[(a, o)] = float(rating)
    return am.BipartiteDiGraph(edges, object_type=object_type, edge_attr=edge_attr)


def main():
    g = _load()
    print("Example 4 - MovieLens 100K (public data)")
    if g is None:
        print("  data not found at examples/data/ml-100k/ - skipping.")
        print("  download: https://files.grouplens.org/datasets/movielens/ml-100k.zip")
        return None
    print(f"  loaded: {len(g.agents())} users x {len(g.objects())} movies")
    print("  census:", am.census(g))
    strat = am.stratified_census(g)
    top = sorted(strat["fan-in"].items(), key=lambda kv: -kv[1])[:5]
    print("  fan-in by movie genre (top 5):", top)
    return True


if __name__ == "__main__":
    main()
