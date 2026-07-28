#!/usr/bin/env python
import argparse
from pathlib import Path

import h5py
import numpy as np

from bioimage_cpp.skeleton import clean_filament_graph, draw_instances, skeleton_to_graph, teasar
from bioimage_cpp.graph import connected_components

BASE = "/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data"
TEASAR_DIR = f"{BASE}/predictions/deepict/csv/instances/teasar"
OUTPUT_DIR = TEASAR_DIR
MASK_PATH = f"{BASE}/experimental/deepict/h5/00004.h5"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_path", type=str, default=MASK_PATH,
                        help="Source .h5 volume to skeletonize and overlay.")
    parser.add_argument("--seg_key", type=str, default="labels/actin", help="HDF5 key of the binary mask.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--tag", type=str, default="", help="Suffix appended to output filenames.")
    parser.add_argument("--pixel_size", type=float, default=10.0,
                        help="Voxel spacing passed to teasar; scales the graph to the mask.")
    parser.add_argument("--scale", type=float, default=0, help="teasar invalidation-radius scale.")
    parser.add_argument("--constant", type=float, default=70.0, help="teasar invalidation-radius constant in A.")
    parser.add_argument("--number_of_threads", type=int, default=8)
    parser.add_argument("--direction_span", type=int, default=10)
    parser.add_argument("--min_branch_angle", type=float, default=20.0,
                        help="Min branch angle (deg) for a degree-3 odd arm to be separated.")
    parser.add_argument("--min_daughter_length", type=float, default=100.0,
                        help="Min length (A) of the branch a degree-3 split would separate.")
    parser.add_argument("--tick_length", type=float, default=200.0,
                        help="Max dead-end branch length in A to prune as a spur.")
    parser.add_argument("--join_dist", type=float, default=50.0,
                        help="If > 0, reconnect collinear endpoints across gaps up to this distance (A).")
    parser.add_argument("--min_join_angle", type=float, default=175.0,
                        help="Min straightness (deg) for a join; 180 is collinear.")
    parser.add_argument("--circle_size", type=float, default=70.0, help="Instance tube diameter in A.")
    parser.add_argument("--view", action="store_true", help="Open the result in napari.")
    return parser.parse_args()


def _load_mask(mask_path, seg_key):
    with h5py.File(mask_path, "r") as f:
        if seg_key not in f:
            raise KeyError(f"'{seg_key}' not in {mask_path}")
        return np.asarray(f[seg_key][:])


def _skeletonize(mask, pixel_size, scale, constant, number_of_threads):
    binary = (mask > 0).astype(np.uint8)
    spacing = (pixel_size, pixel_size, pixel_size)
    return teasar(binary, spacing=spacing, scale=scale, constant=constant,
                  number_of_threads=number_of_threads)


def main():
    args = parse_args()

    mask = _load_mask(args.mask_path, args.seg_key)
    name = f"{Path(args.mask_path).stem}_{args.seg_key.replace('/', '_')}"
    if args.tag:
        name = f"{name}_{args.tag}"

    raw_vertices, raw_edges, raw_radii = _skeletonize(
        mask, args.pixel_size, args.scale, args.constant, args.number_of_threads
    )

    vertices, edges, radii = clean_filament_graph(
        raw_vertices, raw_edges, radii=raw_radii,
        direction_span=args.direction_span, min_branch_angle=args.min_branch_angle,
        min_daughter_length=args.min_daughter_length, tick_length=args.tick_length,
        join_dist=args.join_dist, min_join_angle=args.min_join_angle,
    )
    labels = connected_components(skeleton_to_graph(vertices, edges))

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        skel_kwargs = dict(vertices=raw_vertices, edges=raw_edges)
        if raw_radii is not None:
            skel_kwargs["radii"] = raw_radii
        np.savez_compressed(output_dir / f"{name}_skeleton.npz", **skel_kwargs)
        inst_kwargs = dict(vertices=vertices, edges=edges, labels=labels)
        if radii is not None:
            inst_kwargs["radii"] = radii
        np.savez_compressed(output_dir / f"{name}_instances.npz", **inst_kwargs)
        print(f"{name}: {len(np.unique(labels))} instances -> {output_dir}")

    if args.view:
        import napari
        ps = args.pixel_size

        viewer = napari.Viewer()
        with h5py.File(args.mask_path, "r") as f:
            if "raw" in f:
                viewer.add_image(np.asarray(f["raw"][:]), name="raw")
        viewer.add_labels(mask.astype(np.uint8), name="mask", opacity=0.4)
        if len(raw_edges):
            skel = draw_instances(raw_vertices / ps, raw_edges,
                                  np.zeros(len(raw_vertices), np.int64), mask.shape, (args.circle_size / 2) / ps)
            viewer.add_labels(skel, name="instances (raw)", visible=False)
        if len(edges):
            inst = draw_instances(vertices / ps, edges, labels, mask.shape,
                                  (args.circle_size / 2) / ps)
            viewer.add_labels(inst, name="instances (cleaned)")
        napari.run()


if __name__ == "__main__":
    main()
