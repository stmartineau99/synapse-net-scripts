#!/usr/bin/env python
"""Run kimimaro over a parameter grid on the crop and save each skeleton as npz.

Runs in `tardis-em`, which has kimimaro but cannot import `bioimage_cpp`, so this only
generates skeletons. `score_npz.py` scores them afterwards in `synapse-net` with the same
metric code used for teasar, so the two methods are measured identically.

`dust_threshold` is fixed at 0 throughout: dropping small components is not the target.
"""
import csv
import time
import argparse
import itertools
from pathlib import Path

import numpy as np
import h5py

import kimimaro

CROP_PATH = ("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/predictions/deepict/"
             "csv/instances/teasar/benchmark/crop200.h5")
FIELDS = ["index", "scale", "constant", "fix_branching", "fill_holes", "max_paths",
          "status", "seconds", "vertices", "edges", "npz"]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mask_path", type=str, default=CROP_PATH)
    parser.add_argument("--seg_key", type=str, default="labels/actin")
    parser.add_argument("--scales", type=float, nargs="+", default=[0.0])
    parser.add_argument("--constants", type=float, nargs="+", default=[70.0])
    parser.add_argument("--fix_branching", type=int, nargs="+", default=[1])
    parser.add_argument("--fill_holes", type=int, nargs="+", default=[0])
    parser.add_argument("--max_paths", type=int, nargs="+", default=[0],
                        help="0 means unlimited.")
    parser.add_argument("--pixel_size", type=float, default=10.0)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--manifest", type=str, required=True)
    return parser.parse_args()


def run(mask, spacing, scale, constant, fix_branching, fill_holes, max_paths):
    params = {
        "scale": scale, "const": constant,
        "pdrf_scale": 100000, "pdrf_exponent": 4,
        "soma_detection_threshold": float("inf"),
        "soma_acceptance_threshold": float("inf"),
        "soma_invalidation_scale": 1.0, "soma_invalidation_const": 0.0,
    }
    if max_paths:
        params["max_paths"] = max_paths
    result = kimimaro.skeletonize(
        mask.astype(np.uint32), teasar_params=params, anisotropy=spacing,
        object_ids=[1], dust_threshold=0, progress=False,
        fix_branching=bool(fix_branching), fix_borders=False,
        fill_holes=bool(fill_holes), parallel=1,
    )
    skeleton = result[1] if isinstance(result, dict) else result
    return (np.asarray(skeleton.vertices, dtype=np.float64),
            np.asarray(skeleton.edges, dtype=np.int64))


def main():
    args = parse_args()
    with h5py.File(args.mask_path, "r") as f:
        mask = np.asarray(f[args.seg_key][:]) > 0
    print(f"input {mask.shape} foreground {int(mask.sum())}", flush=True)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    grid = list(itertools.product(args.scales, args.constants, args.fix_branching,
                                 args.fill_holes, args.max_paths))
    print(f"{len(grid)} configurations", flush=True)

    manifest = Path(args.manifest)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for index, (scale, constant, fix_branching, fill_holes, max_paths) in enumerate(grid):
            npz = outdir / f"kimimaro_{index:03d}.npz"
            row = {"index": index, "scale": scale, "constant": constant,
                   "fix_branching": fix_branching, "fill_holes": fill_holes,
                   "max_paths": max_paths, "status": "ok", "npz": str(npz)}
            start = time.perf_counter()
            try:
                vertices, edges = run(mask, (args.pixel_size,) * 3, scale, constant,
                                      fix_branching, fill_holes, max_paths)
                np.savez_compressed(npz, vertices=vertices, edges=edges)
                row["vertices"] = len(vertices)
                row["edges"] = len(edges)
            except Exception as error:
                row["status"] = f"{type(error).__name__}: {error}"[:120]
            row["seconds"] = round(time.perf_counter() - start, 1)
            writer.writerow(row)
            handle.flush()
            print(f"[{index:03d}] scale={scale:<5g} const={constant:<6g} "
                  f"fix_branching={fix_branching} fill_holes={fill_holes} "
                  f"max_paths={max_paths} verts={row.get('vertices')} "
                  f"{row['status']} {row['seconds']}s", flush=True)
    print(f"wrote {manifest}", flush=True)


if __name__ == "__main__":
    main()
