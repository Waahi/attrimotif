# -*- coding: utf-8 -*-
r"""R3.7: wall-clock and memory across size and density, and the R3.1 crossover.

Reviewer 3 asks for timings and memory footprints on synthetic bipartite graphs
whose size and density each span at least two orders of magnitude, and for the
practical upper limit of the present implementation.

Grid: n_A in {100, 316, 1000, 3162, 10000} (100x) and density in
{0.001, 0.003, 0.01, 0.03, 0.1} (100x), n_O = n_A.

Each cell runs in its own subprocess with a timeout, so a cell that exceeds the
budget is recorded as a limit rather than taking the run down with it. Results
are appended after every cell, so a partial run is still usable.

Four routes are measured per cell, and every call names its backend explicitly.
Relying on the default would silently change what is being measured, because
``backend="auto"`` switches to the dense route above a thousand agents:

  census-auto     the census as the public API runs it, overlap on backend
                  "auto" (dense above a thousand agents)
  census-sparse   the same census with the overlap forced sparse
  overlap-sparse  the set-intersection overlap count
  overlap-matrix  the dense B B^T overlap count, via the shipped backend
  size3-enumerate the enumeration path, which stratification and Phi still
                  need; this is the memory wall and the R3.1 crossover

Usage:
  python bench_scaling.py                 run the whole grid
  python bench_scaling.py --cell 1000 0.01 --route census-closed   (internal)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
import tracemalloc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "attrimotif", "src"))

OUT = os.path.join(HERE, "bench_scaling.csv")
N_A = [100, 316, 1000, 3162, 10000]
DENSITY = [0.001, 0.003, 0.01, 0.03, 0.1]
ROUTES = ["census-auto", "census-sparse", "overlap-sparse", "overlap-matrix",
          "size3-enumerate"]
TIMEOUT_S = 300
FIELDS = ["n_A", "n_O", "density", "arcs", "route", "seconds",
          "peak_mib", "result", "status"]


def build(n_a: int, n_o: int, density: float, seed: int = 0):
    """A bipartite arc list at the requested density, drawn without replacement."""
    import numpy as np

    rng = np.random.default_rng(seed)
    k = int(round(density * n_a * n_o))
    if k < 1:
        return []
    # sample cell indices without materialising the full n_a*n_o grid
    idx = rng.choice(n_a * n_o, size=k, replace=False)
    idx.sort()
    return [(f"a{int(i) // n_o}", f"o{int(i) % n_o}") for i in idx]


def run_cell(n_a: int, density: float, route: str) -> dict:
    import importlib

    census_mod = importlib.import_module("attrimotif.census")

    n_o = n_a
    edges = build(n_a, n_o, density)
    if not edges:
        return {"arcs": 0, "seconds": 0.0, "peak_mib": 0.0,
                "result": 0, "status": "empty"}

    tracemalloc.start()
    t0 = time.perf_counter()
    if route in ("census-auto", "census-sparse"):
        # census-auto is what a user actually gets from the public API, which
        # leaves overlap_count on backend="auto"; census-sparse forces the
        # sparse route so the two can be compared. Reporting only the forced
        # one, as the first version of this benchmark did, describes a path
        # nobody takes by default.
        backend = "auto" if route == "census-auto" else "sparse"
        res = dict(census_mod.size3_counts(edges))
        res.update(census_mod.size4_fan_counts(edges))
        res["overlap"] = census_mod.overlap_count(edges, backend=backend)
        out = int(sum(res.values()))
    elif route == "overlap-sparse":
        out = int(census_mod.overlap_count(edges, backend="sparse"))
    elif route == "size3-enumerate":
        inst = census_mod.enumerate_size3(edges)
        out = int(sum(len(v) for v in inst.values()))
    elif route == "overlap-matrix":
        # the shipped backend, not a private copy: a benchmark that
        # re-implements the thing it measures can drift from it
        out = int(census_mod.overlap_count(edges, backend="matrix"))
    else:
        raise SystemExit(f"unknown route {route}")
    secs = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {"arcs": len(edges), "seconds": round(secs, 4),
            "peak_mib": round(peak / 2 ** 20, 2), "result": out, "status": "ok"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", nargs=2, type=float, default=None)
    ap.add_argument("--route", default=None)
    args = ap.parse_args()

    if args.cell is not None:                      # child process: one cell
        row = run_cell(int(args.cell[0]), args.cell[1], args.route)
        print("RESULT " + json.dumps(row))
        return 0

    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()

    print(f"grid: {len(N_A)} sizes x {len(DENSITY)} densities x {len(ROUTES)} routes, "
          f"timeout {TIMEOUT_S}s per cell")
    for n_a in N_A:
        for density in DENSITY:
            for route in ROUTES:
                cmd = [sys.executable, os.path.abspath(__file__),
                       "--cell", str(n_a), str(density), "--route", route]
                started = time.perf_counter()
                try:
                    p = subprocess.run(cmd, capture_output=True, text=True,
                                       timeout=TIMEOUT_S,
                                       env={**os.environ, "PYTHONUTF8": "1"})
                    line = next((l for l in p.stdout.splitlines()
                                 if l.startswith("RESULT ")), None)
                    if line:
                        row = json.loads(line[len("RESULT "):])
                    else:
                        tail = (p.stderr or "").strip().splitlines()
                        row = {"arcs": "", "seconds": round(time.perf_counter() - started, 2),
                               "peak_mib": "", "result": "",
                               "status": "error: " + (tail[-1][:70] if tail else "no output")}
                except subprocess.TimeoutExpired:
                    row = {"arcs": "", "seconds": TIMEOUT_S, "peak_mib": "",
                           "result": "", "status": f"timeout>{TIMEOUT_S}s"}
                row.update({"n_A": n_a, "n_O": n_a, "density": density, "route": route})
                with open(OUT, "a", newline="", encoding="utf-8") as f:
                    csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
                print(f"  n_A={n_a:<6} d={density:<6} {route:<16} "
                      f"{str(row['seconds']):>9}s  {str(row['peak_mib']):>9} MiB  {row['status']}",
                      flush=True)
    print(f"\ndone -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
