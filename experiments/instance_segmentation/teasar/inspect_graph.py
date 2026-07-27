#!/usr/bin/env python
import argparse

import h5py
import numpy as np

from bioimage_cpp.skeleton import clean_filament_graph, draw_instances, skeleton_to_graph
from bioimage_cpp.skeleton.postprocessing import _tangent, _pair_angle, _adjacency, _endpoint_tangent
from bioimage_cpp.graph import connected_components

from clean_graph import TEASAR_DIR, MASK_PATH

SKELETON_PATH = f"{TEASAR_DIR}/00004_labels_actin_skeleton.npz"
DEG_COLORS = {1: "#4d88ff", 2: "#999999", 3: "#ff9f1c", 4: "#ff4d6d"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skeleton_path", type=str, default=SKELETON_PATH)
    parser.add_argument("--mask_path", type=str, default=MASK_PATH)
    parser.add_argument("--seg_key", type=str, default="labels/actin")
    parser.add_argument("--pixel_size", type=float, default=10.0)
    parser.add_argument("--direction_span", type=int, default=5)
    parser.add_argument("--min_through_angle", type=float, default=160.0)
    parser.add_argument("--min_branch_angle", type=float, default=20.0)
    parser.add_argument("--tick_length", type=float, default=50.0,
                        help="Max dead-end branch length in A to prune as a spur.")
    parser.add_argument("--join_dist", type=float, default=0.0,
                        help="If > 0, reconnect collinear endpoints across gaps up to this (A).")
    parser.add_argument("--min_join_angle", type=float, default=175.0,
                        help="Min straightness (deg) for a join; 180 is collinear.")
    parser.add_argument("--point_size", type=float, default=2.0)
    parser.add_argument("--junction_size", type=float, default=5.0)
    parser.add_argument("--circle_size", type=float, default=10.0, help="Instance tube diameter in A.")
    return parser.parse_args()


def _degrees(graph):
    return np.fromiter(
        (len(graph.node_adjacency(n)) for n in range(graph.number_of_nodes)),
        dtype=np.int64, count=graph.number_of_nodes,
    )


def _through_angle(v, graph, vertices, direction_span):
    neighbors = np.asarray(graph.node_adjacency(int(v)))[:, 0]
    dirs = np.stack([_tangent(int(v), n, graph, vertices, direction_span) for n in neighbors])
    best = -1.0
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            best = max(best, _pair_angle(dirs[i], dirs[j]))
    return best


def main():
    args = parse_args()
    data = np.load(args.skeleton_path)
    vertices, edges = data["vertices"], data["edges"]
    radii = data["radii"] if "radii" in data else None
    graph = skeleton_to_graph(vertices, edges)
    coords = vertices / args.pixel_size

    # degree 3/4 junction analysis
    # degrees = _degrees(graph)
    # junctions = np.where(np.isin(degrees, (3, 4)))[0]
    # angles = np.array([_through_angle(v, graph, vertices, args.direction_span) for v in junctions])
    # for d in (1, 2, 3, 4):
    #     print(f"degree {d}: {int((degrees == d).sum())} nodes")

    stages = []
    clean_vertices, clean_edges, _ = clean_filament_graph(
        vertices, edges, radii=radii,
        direction_span=args.direction_span, min_through_angle=args.min_through_angle,
        min_branch_angle=args.min_branch_angle,
        tick_length=args.tick_length,
        join_dist=args.join_dist, min_join_angle=args.min_join_angle,
        save_intermediates=stages,
    )
    labels = connected_components(skeleton_to_graph(clean_vertices, clean_edges))
    print(f"instances: {len(np.unique(labels))}")

    # endpoints (degree <= 1) of the split-stage graph -- what join_close_components sees
    split_dict = {name: (v, e) for name, v, e, _ in stages}
    split_vertices, split_edges = split_dict["split"]
    indptr, dst, _, split_degrees = _adjacency(len(split_vertices), split_edges)
    endpoints = np.where(split_degrees <= 1)[0]
    tangents, endpoint_ids = [], []
    for ep in endpoints:
        tangent = _endpoint_tangent(int(ep), indptr, dst, split_degrees, split_vertices, args.direction_span)
        if tangent is not None:
            tangents.append(tangent)
            endpoint_ids.append(int(ep))
    tangents = np.asarray(tangents)
    endpoint_ids = np.asarray(endpoint_ids)
    endpoint_coords = split_vertices[endpoint_ids] / args.pixel_size
    print(f"endpoints: {len(endpoint_ids)}")

    import napari
    viewer = napari.Viewer()

    with h5py.File(args.mask_path, "r") as f:
        raw = np.asarray(f["raw"][:]) if "raw" in f else None
        mask = np.asarray(f[args.seg_key][:]) if args.seg_key in f else None
    shape = mask.shape if mask is not None else (
        raw.shape if raw is not None else tuple(np.rint(coords.max(0)).astype(int) + 1))
    if raw is not None:
        viewer.add_image(raw, name="raw")
    if mask is not None:
        viewer.add_labels(mask.astype(np.uint8), name="mask", opacity=0.4)

    if len(edges):
        skel = draw_instances(coords, edges, np.zeros(len(vertices), np.int64), shape, (args.circle_size / 2) / args.pixel_size)
        viewer.add_labels(skel, name="skeleton", visible=False)

    # degree 3/4 junction layers (commented out; investigating endpoints instead)
    # viewer.add_points(
    #     coords[degrees == 3], name="degree 3", size=args.junction_size,
    #     face_color=DEG_COLORS[3], border_color="#1b1b1b", border_width=0.15,
    #     out_of_slice_display=True, blending="translucent_no_depth", visible=False,
    # )
    # viewer.add_points(
    #     coords[degrees == 4], name="degree 4", size=args.junction_size,
    #     face_color=DEG_COLORS[4], border_color="#1b1b1b", border_width=0.15,
    #     out_of_slice_display=True, blending="translucent_no_depth", visible=False,
    # )
    #
    # if len(junctions):
    #     from napari.utils.colormaps import ensure_colormap
    #     angle_colors = ensure_colormap("turbo").map(angles / 180.0)
    #     print("branch_angle: turbo 0 (blue) -> 180 (red) deg; hover a point for its value")
    #     viewer.add_points(coords[junctions], name="branch_angle", size=args.junction_size,
    #                       features={"angle": angles}, face_color=angle_colors,
    #                       out_of_slice_display=True)
    #     thr = np.array([(0, 1, 0, 1) if a >= args.min_through_angle else (1, 0, 0, 1) for a in angles])
    #     viewer.add_points(coords[junctions], name="split_pass", size=args.junction_size,
    #                       face_color=thr, out_of_slice_display=True, visible=False)

    if len(endpoint_ids):
        viewer.add_points(
            endpoint_coords, name="endpoints", size=args.junction_size,
            face_color=DEG_COLORS[1], border_color="#1b1b1b", border_width=0.15,
            features={"tz": tangents[:, 0], "ty": tangents[:, 1], "tx": tangents[:, 2]},
            out_of_slice_display=True, blending="translucent_no_depth",
        )
        vectors = np.zeros((len(endpoint_ids), 2, 3))
        vectors[:, 0] = endpoint_coords
        vectors[:, 1] = tangents
        viewer.add_vectors(vectors, name="endpoint tangents", vector_style="arrow",
                           edge_color="#ffd000", out_of_slice_display=True)

    if len(clean_edges):
        inst = draw_instances(clean_vertices / args.pixel_size, clean_edges, labels, shape,
                              (args.circle_size / 2) / args.pixel_size)
        viewer.add_labels(inst, name="instances")

    napari.run()


if __name__ == "__main__":
    main()
