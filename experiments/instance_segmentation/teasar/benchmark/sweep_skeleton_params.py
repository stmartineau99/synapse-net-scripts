#!/usr/bin/env python
"""Sweep teasar invalidation parameters on a fixed crop.

Objectives: minimise `deg3` on the raw skeleton, maximise `instances_gt200` after postprocessing.
They oppose each other, so results are a Pareto frontier rather than one score.

Postprocessing is held at the production setting, tick_length=200 and join_dist=50, and kept fixed
rather than swept. Pinning tick_length=0 leaves genuine spurs in the graph so each becomes a fragment,
and the instance counts then describe a regime the pipeline never runs.

Ablation detectors: `length_in_real`, the contour length held in instances over 200 A, and
`cov_max`, the largest distance from any mask voxel to the nearest skeleton vertex. A dropped
filament leaves mask far from any skeleton and takes its length out of `length_in_real`.

Self-contained on purpose, so the same metric code can score kimimaro output from another
environment.
"""
import csv
import time
import argparse
import itertools
from pathlib import Path

import numpy as np
import h5py
from scipy.spatial import cKDTree

from bioimage_cpp.graph import connected_components
from bioimage_cpp.skeleton import teasar, skeleton_to_graph, clean_filament_graph

CROP_PATH = ("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/predictions/deepict/"
             "csv/instances/teasar/benchmark/crop200.h5")
FIELDS = ["scale", "constant", "status", "seconds", "vertices", "length_um", "deg1", "deg3",
          "deg3_per_100um", "crossings", "instances", "instances_gt200", "length_in_real_um",
          "cov_p999", "cov_max"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mask_path", type=str, default=CROP_PATH)
    parser.add_argument("--seg_key", type=str, default="labels/actin")
    parser.add_argument("--scales", type=float, nargs="+", default=[0.0])
    parser.add_argument("--constants", type=float, nargs="+", default=[70.0])
    parser.add_argument("--pixel_size", type=float, default=10.0)
    parser.add_argument("--number_of_threads", type=int, default=8)
    parser.add_argument("--direction_span", type=int, default=10)
    parser.add_argument("--min_branch_angle", type=float, default=20.0)
    parser.add_argument("--min_daughter_length", type=float, default=0.0)
    parser.add_argument("--tick_length", type=float, default=200.0)
    parser.add_argument("--join_dist", type=float, default=50.0)
    parser.add_argument("--spur_length", type=float, default=200.0)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def build_adjacency(n_nodes, edges):
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    order = np.argsort(src, kind="stable")
    dst = dst[order]
    degrees = np.bincount(src, minlength=n_nodes)
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(degrees, out=indptr[1:])
    return indptr, dst, degrees


def walk_arm(node, first, indptr, dst, degrees, vertices):
    prev, cur = node, int(first)
    length = float(np.linalg.norm(vertices[cur] - vertices[node]))
    while degrees[cur] == 2:
        s, e = indptr[cur], indptr[cur + 1]
        nbrs = dst[s:e]
        nxt = int(nbrs[0]) if int(nbrs[0]) != prev else int(nbrs[1])
        length += float(np.linalg.norm(vertices[nxt] - vertices[cur]))
        prev, cur = cur, nxt
    return cur, length


def component_lengths(vertices, edges):
    labels = connected_components(skeleton_to_graph(vertices, edges))
    seg = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    return np.bincount(labels[edges[:, 0]], weights=seg, minlength=int(labels.max()) + 1)


def instances(vertices, edges, threshold, **kwargs):
    clean_v, clean_e, _ = clean_filament_graph(vertices, edges, **kwargs)
    if not len(clean_e):
        return 0, 0, 0.0
    lengths = component_lengths(clean_v, clean_e)
    over = lengths > threshold
    return len(lengths), int(over.sum()), float(lengths[over].sum())


def evaluate(vertices, edges, foreground, args):
    n = len(vertices)
    indptr, dst, degrees = build_adjacency(n, edges)
    seg = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    length_um = seg.sum() / 1e4
    deg3 = np.where(degrees == 3)[0]

    crossings = 0
    for node in deg3:
        arms = [walk_arm(int(node), dst[k], indptr, dst, degrees, vertices)
                for k in range(indptr[node], indptr[node + 1])]
        if min(a[1] for a in arms) > args.spur_length:
            crossings += 1

    pinned = dict(direction_span=args.direction_span, min_branch_angle=args.min_branch_angle,
                  min_daughter_length=args.min_daughter_length, tick_length=args.tick_length,
                  join_dist=args.join_dist, min_join_angle=175.0)
    total, over, length_in_real = instances(vertices, edges, args.spur_length, **pinned)

    coverage, _ = cKDTree(vertices).query(foreground, k=1, workers=-1)

    return {
        "vertices": n,
        "length_um": round(length_um, 2),
        "deg1": int((degrees == 1).sum()),
        "deg3": len(deg3),
        "deg3_per_100um": round(100 * len(deg3) / length_um, 1) if length_um else 0,
        "crossings": crossings,
        "instances": total,
        "instances_gt200": over,
        "length_in_real_um": round(length_in_real / 1e4, 2),
        "cov_p999": round(float(np.percentile(coverage, 99.9)), 1),
        "cov_max": round(float(coverage.max()), 1),
    }


def main():
    args = parse_args()
    with h5py.File(args.mask_path, "r") as f:
        mask = np.asarray(f[args.seg_key][:]) > 0
    foreground = np.argwhere(mask).astype(np.float64) * args.pixel_size
    print(f"input {mask.shape} foreground {len(foreground)}", flush=True)

    grid = list(itertools.product(args.scales, args.constants))
    print(f"{len(grid)} configurations", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for scale, constant in grid:
            row = {"scale": scale, "constant": constant, "status": "ok"}
            start = time.perf_counter()
            try:
                vertices, edges, _ = teasar(
                    mask.astype(np.uint8), spacing=(args.pixel_size,) * 3, scale=scale,
                    constant=constant, number_of_threads=args.number_of_threads,
                )
                row.update(evaluate(vertices, np.asarray(edges, dtype=np.int64),
                                    foreground, args))
            except Exception as error:
                row["status"] = f"{type(error).__name__}: {error}"[:120]
            row["seconds"] = round(time.perf_counter() - start, 1)
            writer.writerow(row)
            handle.flush()
            print(f"scale={scale:<5g} const={constant:<6g} deg3={row.get('deg3')} "
                  f"inst>200={row.get('instances_gt200')} real={row.get('length_in_real_um')}um "
                  f"cov_max={row.get('cov_max')} {row['status']} {row['seconds']}s", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
