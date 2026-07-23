"""Visualize suspect (merged) filaments and their unsplit degree-3/4 junctions in napari.

An instance is a suspect when its skeleton graph keeps a degree-3 (branch) or degree-4
(crossing) node that clean_filament_graph did not split. For each such node this recomputes
the split-decision angle clean_filament_graph measured, so you can see why the node missed
the cutoff (degree-3: branch angle vs min_branch_angle; degree-4: through-pair min angle vs
min_through_angle). The analysis runs without napari and saves a suspects npz; the viewer
opens only in main.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np

from bioimage_cpp.skeleton import draw_instances, skeleton_to_graph
from bioimage_cpp.skeleton.postprocessing import _tangent, _pair_angle

DEG_COLORS = {3: "#ff9f1c", 4: "#ff4d6d"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str, required=True,
                        help="Instance npz with vertices, edges, labels.")
    parser.add_argument("--h5", type=str, required=True, help="Deepict h5 with raw and the mask.")
    parser.add_argument("--seg_key", type=str, default="labels/actin")
    parser.add_argument("--pixel_size", type=float, default=10.0)
    parser.add_argument("--direction_span", type=int, default=10)
    parser.add_argument("--min_branch_angle", type=float, default=30.0)
    parser.add_argument("--min_through_angle", type=float, default=170.0)
    parser.add_argument("--out", type=str, default=None, help="Suspects npz path; default beside --npz.")
    parser.add_argument("--junction_size", type=float, default=6.0)
    parser.add_argument("--circle_size", type=float, default=10.0, help="Tube diameter in A.")
    parser.add_argument("--arrow_len", type=float, default=8.0, help="Arm arrow length in voxels.")
    return parser.parse_args()


def _branch_angle(dirs):
    best, pair = -1.0, (0, 1)
    for a, b in [(0, 1), (0, 2), (1, 2)]:
        ang = _pair_angle(dirs[a], dirs[b])
        if ang > best:
            best, pair = ang, (a, b)
    i, j = pair
    odd = ({0, 1, 2} - {i, j}).pop()
    return min(_pair_angle(dirs[odd], dirs[i]), _pair_angle(dirs[odd], dirs[j]))


def _through_min_angle(dirs):
    best_score, best_min = -np.inf, 0.0
    for (a, b), (c, d) in [((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2))]:
        ang1, ang2 = _pair_angle(dirs[a], dirs[b]), _pair_angle(dirs[c], dirs[d])
        if ang1 + ang2 > best_score:
            best_score, best_min = ang1 + ang2, min(ang1, ang2)
    return best_min


def analyze(vertices, edges, direction_span, min_branch_angle, min_through_angle):
    """Find degree-3/4 junctions and the split-decision angle at each. Returns a dict of arrays."""
    graph = skeleton_to_graph(vertices, edges)
    degrees = np.bincount(edges.reshape(-1), minlength=len(vertices))
    nodes, deg, angle, threshold, arm_dirs = [], [], [], [], []
    for v in np.where((degrees == 3) | (degrees == 4))[0]:
        neighbors = np.asarray(graph.node_adjacency(int(v)))[:, 0]
        dirs = np.stack([_tangent(int(v), n, graph, vertices, direction_span) for n in neighbors])
        if degrees[v] == 3:
            a, thr = _branch_angle(dirs), min_branch_angle
        else:
            a, thr = _through_min_angle(dirs), min_through_angle
        nodes.append(int(v))
        deg.append(int(degrees[v]))
        angle.append(float(a))
        threshold.append(float(thr))
        arm_dirs.append(dirs)
    return {
        "degrees": degrees,
        "node": np.asarray(nodes, dtype=np.int64),
        "degree": np.asarray(deg, dtype=np.int64),
        "angle": np.asarray(angle, dtype=np.float64),
        "threshold": np.asarray(threshold, dtype=np.float64),
        "arm_dirs": arm_dirs,
    }


def suspect_label_set(labels, degrees):
    return {int(labels[v]) for v in np.where(degrees >= 3)[0]}


def save_suspects(out_path, vertices, edges, labels, suspects, junctions):
    passed = junctions["angle"] >= junctions["threshold"]
    np.savez_compressed(
        out_path,
        vertices=vertices, edges=edges, labels=labels,
        suspect_labels=np.asarray(sorted(suspects), dtype=np.int64),
        junction_node=junctions["node"], junction_degree=junctions["degree"],
        junction_angle=junctions["angle"], junction_threshold=junctions["threshold"],
        junction_passed=passed,
    )


def main():
    args = parse_args()
    npz_path = Path(args.npz)
    data = np.load(npz_path)
    vertices, edges, labels = data["vertices"], data["edges"], data["labels"]
    ps = args.pixel_size

    junctions = analyze(vertices, edges, args.direction_span, args.min_branch_angle, args.min_through_angle)
    degrees = junctions["degrees"]
    suspects = suspect_label_set(labels, degrees)
    passed = junctions["angle"] >= junctions["threshold"]
    print(f"{len(suspects)} suspect filaments, {len(junctions['node'])} degree-3/4 junctions "
          f"({int((~passed).sum())} below cutoff)")

    out_path = Path(args.out) if args.out else npz_path.with_name(f"{npz_path.stem}_suspects.npz")
    save_suspects(out_path, vertices, edges, labels, suspects, junctions)
    print(f"suspects npz -> {out_path}")

    with h5py.File(args.h5, "r") as f:
        raw = np.asarray(f["raw"][:]) if "raw" in f else None
        mask = np.asarray(f[args.seg_key][:]) if args.seg_key in f else None
    shape = mask.shape if mask is not None else (
        raw.shape if raw is not None else tuple(np.rint((vertices / ps).max(0)).astype(int) + 1))

    import napari

    viewer = napari.Viewer()
    if raw is not None:
        viewer.add_image(raw, name="raw")
    if mask is not None:
        viewer.add_labels(mask.astype(np.uint8), name="mask", opacity=0.4)

    radius = (args.circle_size / 2) / ps
    viewer.add_labels(
        draw_instances(vertices / ps, edges, labels, shape, radius),
        name="instances", visible=False,
    )
    suspect_labels = np.where(np.isin(labels, list(suspects)), labels, 0)
    viewer.add_labels(
        draw_instances(vertices / ps, edges, suspect_labels, shape, radius),
        name="suspect instances",
    )

    node = junctions["node"]
    if len(node):
        coords = vertices[node] / ps
        colors = np.array([DEG_COLORS[d] for d in junctions["degree"]])
        viewer.add_points(
            coords, name="junctions", size=args.junction_size, face_color=colors,
            border_color="#1b1b1b", border_width=0.15, out_of_slice_display=True,
            blending="translucent_no_depth",
            features={
                "degree": junctions["degree"], "angle": junctions["angle"],
                "threshold": junctions["threshold"], "passed": passed,
            },
        )

        arrows = []
        for v, dirs in zip(node, junctions["arm_dirs"]):
            for d in dirs:
                arrows.append([vertices[v] / ps, d * args.arrow_len])
        vectors = np.asarray(arrows).reshape(-1, 2, 3)
        viewer.add_vectors(vectors, name="arm directions", vector_style="arrow",
                           edge_color="#ffd000", out_of_slice_display=True)

        points_layer = viewer.layers["junctions"]

        @points_layer.mouse_drag_callbacks.append
        def _on_click(layer, event):
            idx = layer.get_value(
                event.position, view_direction=event.view_direction,
                dims_displayed=event.dims_displayed, world=True,
            )
            if idx is None:
                return
            f = layer.features.iloc[int(idx)]
            verdict = "split" if f["passed"] else "NOT split"
            print(f"degree {int(f['degree'])}: angle {f['angle']:.1f} deg vs "
                  f"threshold {f['threshold']:.0f} -> would {verdict}")

    napari.run()


if __name__ == "__main__":
    main()
