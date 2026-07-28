#!/usr/bin/env python
"""Rasterise teasar and kimimaro instances under identical postprocessing, for visual comparison.

Both skeletons are cleaned with the same parameters, so the only difference is the invalidation
region the skeleton came from: teasar's cube against kimimaro's sphere.

Two steps, because the compute node has no display:

    python compare_guarded.py                  # on the interactive node, writes the h5
    python compare_guarded.py --view           # where napari can open a window

Only instances longer than ``--spur_length`` are drawn, so the view shows filaments rather than
debris.
"""
import argparse
from pathlib import Path

import numpy as np
import h5py

BASE = "/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data"
MASK_PATH = f"{BASE}/experimental/deepict/h5/00004.h5"
BENCH = f"{BASE}/predictions/deepict/csv/instances/teasar/benchmark"
TEASAR_NPZ = f"{BASE}/predictions/deepict/csv/instances/teasar/00004_labels_actin_skeleton.npz"
KIMIMARO_NPZ = f"{BENCH}/kimimaro_full/kimimaro_000.npz"
OUT_PATH = f"{BENCH}/guarded_comparison.h5"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--view", action="store_true", help="Open the written h5 in napari.")
    parser.add_argument("--out", type=str, default=OUT_PATH)
    parser.add_argument("--mask_path", type=str, default=MASK_PATH)
    parser.add_argument("--seg_key", type=str, default="labels/actin")
    parser.add_argument("--teasar_npz", type=str, default=TEASAR_NPZ)
    parser.add_argument("--kimimaro_npz", type=str, default=KIMIMARO_NPZ)
    parser.add_argument("--crop", type=int, default=0,
                        help="Centred cube of this many voxels; 0 uses the whole volume.")
    parser.add_argument("--pixel_size", type=float, default=10.0)
    parser.add_argument("--direction_span", type=int, default=10)
    parser.add_argument("--min_branch_angle", type=float, default=20.0)
    parser.add_argument("--min_daughter_length", type=float, default=100.0)
    parser.add_argument("--tick_length", type=float, default=200.0)
    parser.add_argument("--join_dist", type=float, default=50.0)
    parser.add_argument("--circle_size", type=float, default=70.0)
    parser.add_argument("--spur_length", type=float, default=200.0)
    return parser.parse_args()


def instances(npz_path, shape, origin, args):
    from bioimage_cpp.graph import connected_components
    from bioimage_cpp.skeleton import skeleton_to_graph, clean_filament_graph, draw_instances

    data = np.load(npz_path)
    vertices, edges = data["vertices"], np.asarray(data["edges"], dtype=np.int64)
    vertices, edges, _ = clean_filament_graph(
        vertices, edges,
        direction_span=args.direction_span, min_branch_angle=args.min_branch_angle,
        min_daughter_length=args.min_daughter_length, tick_length=args.tick_length,
        join_dist=args.join_dist, min_join_angle=175.0,
    )
    labels = connected_components(skeleton_to_graph(vertices, edges))
    seg = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    lengths = np.bincount(labels[edges[:, 0]], weights=seg, minlength=int(labels.max()) + 1)

    long_enough = lengths > args.spur_length
    coords = vertices / args.pixel_size - np.asarray(origin, dtype=np.float64)
    radius = (args.circle_size / 2) / args.pixel_size
    volume = draw_instances(coords, edges[long_enough[labels[edges[:, 0]]]], labels, shape, radius)
    return volume, int(long_enough.sum()), int((~long_enough).sum())


def write(args):
    with h5py.File(args.mask_path, "r") as f:
        full = f[args.seg_key].shape
        if args.crop > 0:
            origin = tuple(max(0, (n - args.crop) // 2) for n in full)
            slices = tuple(slice(o, o + args.crop) for o in origin)
        else:
            origin, slices = (0, 0, 0), tuple(slice(None) for _ in full)
        mask = np.asarray(f[args.seg_key][slices])
        raw = np.asarray(f["raw"][slices]) if "raw" in f else None
    shape = mask.shape
    print(f"shape {shape} origin {origin}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        if raw is not None:
            f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("mask", data=(mask > 0).astype(np.uint8), compression="gzip")
        for name, npz in (("teasar", args.teasar_npz), ("kimimaro", args.kimimaro_npz)):
            volume, long_count, frag_count = instances(npz, shape, origin, args)
            dataset = f.create_dataset(name, data=volume, compression="gzip")
            dataset.attrs["filaments"] = long_count
            dataset.attrs["fragments_hidden"] = frag_count
            print(f"{name}: {long_count} filaments over {args.spur_length:.0f} A, "
                  f"{frag_count} fragments not drawn", flush=True)
        f.attrs["origin"] = origin
    print(f"wrote {out}", flush=True)


def view(args):
    import napari

    with h5py.File(args.out, "r") as f:
        viewer = napari.Viewer()
        if "raw" in f:
            viewer.add_image(np.asarray(f["raw"][:]), name="raw")
        viewer.add_labels(np.asarray(f["mask"][:]), name="mask", opacity=0.3, visible=False)
        for name in ("teasar", "kimimaro"):
            dataset = f[name]
            viewer.add_labels(np.asarray(dataset[:]),
                              name=f"{name} ({dataset.attrs['filaments']} filaments)")
    napari.run()


def main():
    args = parse_args()
    view(args) if args.view else write(args)


if __name__ == "__main__":
    main()
