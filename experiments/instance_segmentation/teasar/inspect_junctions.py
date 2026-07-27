#!/usr/bin/env python
"""Inspect degree-3/4 junctions on the post-remove_ticks graph.

Loads the raw teasar skeleton, runs clean_filament_graph, and shows in napari
where the degree-3/4 graph cut acts. Degree-3 nodes are colored by the
min_branch_angle check: separated (odd arm is cut) vs kept (branch retained).
A "small fragments" layer marks the tiny instances left after splitting, to see
whether they sit at the separated junctions.
"""
import argparse

import h5py
import numpy as np

from bioimage_cpp.skeleton import clean_filament_graph, draw_instances, skeleton_to_graph
from bioimage_cpp.skeleton.postprocessing import _tangent, _pair_angle, _split_degree3
from bioimage_cpp.graph import connected_components

from clean_graph import TEASAR_DIR, MASK_PATH

SKELETON_PATH = f"{TEASAR_DIR}/00004_labels_actin_skeleton.npz"
SEPARATED_COLOR = "#ff4d6d"
KEPT_COLOR = "#2ecc71"
DEG4_COLOR = "#ff9f1c"
FRAGMENT_COLOR = "#00e5ff"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skeleton_path", type=str, default=SKELETON_PATH)
    parser.add_argument("--mask_path", type=str, default=MASK_PATH)
    parser.add_argument("--seg_key", type=str, default="labels/actin")
    parser.add_argument("--pixel_size", type=float, default=10.0)
    parser.add_argument("--direction_span", type=int, default=10)
    parser.add_argument("--min_branch_angle", type=float, default=20.0,
                        help="A degree-3 odd arm is separated when its branch angle is at least this (deg).")
    parser.add_argument("--tick_length", type=float, default=50.0,
                        help="Max dead-end branch length in A to prune as a spur.")
    parser.add_argument("--join_dist", type=float, default=50.0,
                        help="If > 0, reconnect collinear endpoints across gaps up to this (A).")
    parser.add_argument("--min_join_angle", type=float, default=175.0,
                        help="Min straightness (deg) for a join; 180 is collinear.")
    parser.add_argument("--small_fragment_length", type=float, default=200.0,
                        help="Instances shorter than this contour length (A) are highlighted as fragments.")
    parser.add_argument("--junction_size", type=float, default=5.0)
    parser.add_argument("--circle_size", type=float, default=10.0, help="Instance tube diameter in A.")
    return parser.parse_args()


def _degrees(graph):
    return np.fromiter(
        (len(graph.node_adjacency(n)) for n in range(graph.number_of_nodes)),
        dtype=np.int64, count=graph.number_of_nodes,
    )


def _branch_angle(v, graph, vertices, direction_span):
    """Smaller angle between a degree-3 node's odd arm and its through pair.

    This is the value `_split_degree3` compares against `min_branch_angle`.
    """
    neighbors = np.asarray(graph.node_adjacency(int(v)))[:, 0]
    dirs = np.stack([_tangent(int(v), n, graph, vertices, direction_span) for n in neighbors])
    best, pair = -1.0, (0, 1)
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        ang = _pair_angle(dirs[a], dirs[b])
        if ang > best:
            best, pair = ang, (a, b)
    odd = ({0, 1, 2} - set(pair)).pop()
    return float(min(_pair_angle(dirs[odd], dirs[pair[0]]), _pair_angle(dirs[odd], dirs[pair[1]])))


def main():
    args = parse_args()
    data = np.load(args.skeleton_path)
    vertices, edges = data["vertices"], data["edges"]
    radii = data["radii"] if "radii" in data else None

    stages = []
    clean_vertices, clean_edges, _ = clean_filament_graph(
        vertices, edges, radii=radii,
        direction_span=args.direction_span,
        min_branch_angle=args.min_branch_angle, tick_length=args.tick_length,
        join_dist=args.join_dist, min_join_angle=args.min_join_angle,
        save_intermediates=stages,
    )
    steps = {name: (v, e) for name, v, e, _ in stages}
    labels = connected_components(skeleton_to_graph(clean_vertices, clean_edges))

    # degree 3/4 junctions on the post-remove_ticks graph
    tick_vertices, tick_edges = steps["ticks"]
    tick_graph = skeleton_to_graph(tick_vertices, tick_edges)
    tick_coords = tick_vertices / args.pixel_size
    tick_degrees = _degrees(tick_graph)
    deg3 = np.where(tick_degrees == 3)[0]
    deg4 = np.where(tick_degrees == 4)[0]
    passes = np.array(
        [_split_degree3(int(v), tick_graph, tick_vertices,
                        direction_span=args.direction_span,
                        min_branch_angle=args.min_branch_angle) is not None
         for v in deg3],
        dtype=bool,
    )
    branch_angles = np.array(
        [_branch_angle(int(v), tick_graph, tick_vertices, args.direction_span) for v in deg3]
    )
    print(f"degree 3: {len(deg3)} ({int(passes.sum())} separated), degree 4: {len(deg4)}")

    # small instances left after splitting, by contour length in A
    clean_coords = clean_vertices / args.pixel_size
    n_instances = int(labels.max()) + 1
    seg_lengths = np.linalg.norm(
        clean_vertices[clean_edges[:, 0]] - clean_vertices[clean_edges[:, 1]], axis=1)
    comp_length = np.bincount(labels[clean_edges[:, 0]], weights=seg_lengths, minlength=n_instances)
    small = np.where(comp_length <= args.small_fragment_length)[0]
    small_mask = np.isin(labels, small)
    print(f"small fragments (<= {args.small_fragment_length:.0f} A): {len(small)} / {n_instances} instances")

    import napari
    viewer = napari.Viewer()

    with h5py.File(args.mask_path, "r") as f:
        raw = np.asarray(f["raw"][:]) if "raw" in f else None
        mask = np.asarray(f[args.seg_key][:]) if args.seg_key in f else None
    shape = mask.shape if mask is not None else (
        raw.shape if raw is not None else tuple(np.rint(tick_coords.max(0)).astype(int) + 1))
    if raw is not None:
        viewer.add_image(raw, name="raw")
    if mask is not None:
        viewer.add_labels(mask.astype(np.uint8), name="mask", opacity=0.4)

    if len(clean_edges):
        inst = draw_instances(clean_coords, clean_edges, labels, shape,
                              (args.circle_size / 2) / args.pixel_size)
        viewer.add_labels(inst, name="instances")

    def add_junctions(name, coords, color, angle=None):
        kwargs = dict(
            name=name, size=args.junction_size, face_color=color,
            border_color="#1b1b1b", border_width=0.15,
            out_of_slice_display=True, blending="translucent_no_depth",
        )
        if angle is not None:
            kwargs["features"] = {"branch_angle": angle}
        viewer.add_points(coords, **kwargs)

    if len(deg3):
        add_junctions("deg3 separated (angle>=min)", tick_coords[deg3][passes], SEPARATED_COLOR,
                      branch_angles[passes])
        add_junctions("deg3 kept (angle<min)", tick_coords[deg3][~passes], KEPT_COLOR,
                      branch_angles[~passes])
    if len(deg4):
        add_junctions("deg4", tick_coords[deg4], DEG4_COLOR)
    if small_mask.any():
        add_junctions("small fragments", clean_coords[small_mask], FRAGMENT_COLOR)

    napari.run()


if __name__ == "__main__":
    main()
