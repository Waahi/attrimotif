# -*- coding: utf-8 -*-
"""Example 5 - the complete four-module workflow on real public data.

Example 4 runs only the census on MovieLens 100K. This example exercises all
four analysis modules on the same real graph, which is what an empirical study
of an attributed directed bipartite network actually does:

  1. census                 which local motifs occur
  2. stratification + Phi   whether structure differs across object categories,
                            and how the numeric edge attribute is distributed
                            within each motif class
  3. null                   whether the counts exceed what the degree sequence
                            alone implies, with the identifiability guard
  4. panel comparison       whether structure changes across cohorts

Panels are the four quartiles of users by the timestamp of their FIRST rating,
so the cohorts are disjoint, exhaustive and defined without reference to any
outcome.

The MovieLens 100K data is NOT bundled (respecting its terms). Download it once:

    https://files.grouplens.org/datasets/movielens/ml-100k.zip

and unzip so that ``examples/data/ml-100k/u.data`` and ``u.item`` exist.

Usage
-----
    python example5_movielens_workflow.py --quick   # small R, for a smoke run
    python example5_movielens_workflow.py --full    # the settings the paper reports
    python example5_movielens_workflow.py --full --only 4 --out r.json

Module 3 (the null test) and module 4 (the panel test) each take tens of
minutes at the reported settings, so ``--only`` runs a subset and ``--out``
is rewritten after every module: an interrupted run keeps what finished, and
re-running with ``--only`` on the same ``--out`` resumes rather than restarts.

Reference: Harper & Konstan (2015), "The MovieLens Datasets", ACM TiiS 5(4),
DOI 10.1145/2827872.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import attrimotif as am

DATA = os.path.join(os.path.dirname(__file__), "data", "ml-100k")
GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def load(with_time=True):
    u_item = os.path.join(DATA, "u.item")
    u_data = os.path.join(DATA, "u.data")
    if not (os.path.exists(u_item) and os.path.exists(u_data)):
        return None, None
    object_type = {}
    with open(u_item, encoding="latin-1") as f:
        for line in f:
            p = line.rstrip("\n").split("|")
            flags = [int(x) for x in p[-19:]]
            object_type[f"m{p[0]}"] = next(
                (GENRES[i] for i, v in enumerate(flags) if v and i > 0), "unknown")
    edges, edge_attr, first_ts = [], {}, {}
    with open(u_data, encoding="latin-1") as f:
        for line in f:
            uid, mid, rating, ts = line.split("\t")
            a, o = f"u{uid}", f"m{mid}"
            edges.append((a, o))
            edge_attr[(a, o)] = float(rating)
            t = int(ts)
            if a not in first_ts or t < first_ts[a]:
                first_ts[a] = t
    g = am.BipartiteDiGraph(edges, object_type=object_type, edge_attr=edge_attr)
    return g, (first_ts if with_time else None)


def quartile_cohorts(first_ts):
    """Four disjoint user cohorts by the timestamp of their first rating."""
    users = sorted(first_ts, key=lambda a: (first_ts[a], a))
    n = len(users)
    cuts = [round(n * i / 4) for i in range(5)]
    cohort = {}
    for q in range(4):
        for a in users[cuts[q]:cuts[q + 1]]:
            cohort[a] = f"cohort{q + 1}"
    sizes = {f"cohort{q + 1}": cuts[q + 1] - cuts[q] for q in range(4)}
    return cohort, sizes


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small R, for a smoke run")
    ap.add_argument("--full", action="store_true", help="the settings the paper reports")
    ap.add_argument("--out", default=None, help="write results as JSON")
    ap.add_argument("--min-shared", default=None,
                    help="comma-separated min_shared values for module 4; "
                         "defaults to 2,10 with --full and 10 otherwise. The "
                         "denser (smaller) values cost far more per replicate.")
    ap.add_argument("--only", default="1,2,3,4",
                    help="comma-separated modules to run, e.g. --only 4. Module 3 "
                         "(the null test) is the expensive one, so running the "
                         "modules separately is often what you want.")
    args = ap.parse_args(argv)
    R_null = 200 if args.full else 20
    R_panel = 200 if args.full else 5
    min_shared_values = ((2, 10) if args.full else (10,))
    if args.min_shared:
        min_shared_values = tuple(int(x) for x in args.min_shared.split(",") if x.strip())
    only = {int(x) for x in args.only.split(",") if x.strip()}

    g, first_ts = load()
    print("Example 5 - complete four-module workflow on MovieLens 100K")
    if g is None:
        print("  data not found at examples/data/ml-100k/ - skipping.")
        print("  download: https://files.grouplens.org/datasets/movielens/ml-100k.zip")
        return None

    # Results are written after every module, so a long run that is interrupted
    # still leaves the modules that finished.
    res = {}
    if args.out and os.path.exists(args.out):
        try:
            res = json.load(open(args.out, encoding="utf-8"))
            print(f"  resuming: reloaded {sorted(res)} from {args.out}")
        except (OSError, ValueError):
            res = {}
    res.update({"mode": "full" if args.full else "quick",
                "modules_requested": sorted(only),
                "agents": len(g.agents()), "objects": len(g.objects()),
                "arcs": len(set(g.edges))})

    def save():
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=1, default=str)

    print(f"  loaded: {res['agents']} users x {res['objects']} movies, {res['arcs']} arcs")
    save()

    # -- 1. census ------------------------------------------------------------
    if 1 in only:
        t0 = time.perf_counter()
        res["census"] = am.census(g)
        res["t_census"] = round(time.perf_counter() - t0, 3)
        print(f"  [1] census ({res['t_census']}s): {res['census']}")
        save()

    # -- 2. stratification + operator Phi ------------------------------------
    if 2 in only:
        t0 = time.perf_counter()
        strat = am.stratified_census(g)
        phi = am.phi_distributions(g)
        res["t_strat"] = round(time.perf_counter() - t0, 3)
        res["fan_in_by_genre_top5"] = sorted(
            strat["fan-in"].items(), key=lambda kv: -kv[1])[:5]
        # phi_distributions returns arrays, so test emptiness with len()
        res["phi"] = {
            k: {"n": int(len(v)),
                "mean": float(sum(v) / len(v)) if len(v) else None,
                "max": float(max(v)) if len(v) else None}
            for k, v in phi.items()
        }
        print(f"  [2] stratification + Phi ({res['t_strat']}s)")
        print(f"      fan-in by genre (top 5): {res['fan_in_by_genre_top5']}")
        for k, v in res["phi"].items():
            print(f"      Phi {k:8s}: n={v['n']:>9}  mean rating={v['mean']:.3f}  max={v['max']:.1f}")
        save()

    # -- 3. degree-preserving null + identifiability guard --------------------
    if 3 in only:
        res.setdefault("null", {})
        for stat in ("fan-out", "overlap"):
            t0 = time.perf_counter()
            r = am.null_test(g, stat, n_samples=R_null, seed=0)
            r["seconds"] = round(time.perf_counter() - t0, 1)
            res["null"][stat] = r
            print(f"  [3] null {stat:8s} ({r['seconds']}s, R={R_null}): "
                  f"obs={r['observed']:.0f} null={r['null_mean']:.1f}+-{r['null_sd']:.1f} "
                  f"z={r['z']:.2f} p={r['perm_p']:.4f} ratio={r['mean_swap_ratio']:.3f} "
                  f"[{r['identifiability_route']}]")
            save()

    # -- 4. panel comparison over first-rating cohorts ------------------------
    if 4 in only:
        cohort, sizes = quartile_cohorts(first_ts)
        res["cohort_sizes"] = sizes
        print(f"  [4] cohorts by first-rating quartile: {sizes}")
        res.setdefault("panel", {})
        for ms in min_shared_values:
            t0 = time.perf_counter()
            pt = am.panel_permutation_test(g, cohort, n_samples=R_panel,
                                           min_shared=ms, seed=0)
            secs = round(time.perf_counter() - t0, 1)
            res["panel"][f"min_shared={ms}"] = {
                "seconds": secs,
                "pairs": {f"{a}|{b}": v for (a, b), v in pt.items()},
            }
            print(f"      min_shared={ms} ({secs}s, R={R_panel}):")
            for (a, b), v in sorted(pt.items()):
                print(f"        {a} vs {b}: PD={v['observed']:.4f} "
                      f"null={v['null_mean']:.4f} p={v['perm_p']:.4f}")
            save()

    if args.out:
        print(f"  results -> {args.out}")
    return res


if __name__ == "__main__":
    main()
