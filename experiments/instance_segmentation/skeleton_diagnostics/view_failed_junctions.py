#!/usr/bin/env python
"""Show the degree-3 junctions that remove_ticks cannot fix.

remove_ticks prunes dead-end branches only. The junctions it leaves behind still have a short arm, but
that arm runs to another degree-3 node instead of ending, so pruning never reaches it. Those are the
junctions that fragment filaments downstream, and this renders exactly that population: the two junction
nodes, the short arm bridging them, and the arm directions at each node.

Run in the synapse-net environment, on the default mask:
    micromamba run -n synapse-net python view_failed_junctions.py --crop

--crop restricts to the centred box, which is much faster for interactive work. --no_view prints the
counts and exits, for checking the analysis without a display.
"""
import argparse
from pathlib import Path

import h5py
import mrcfile
import numpy as np

from bioimage_cpp.skeleton import draw_instances, remove_ticks, skeleton_to_graph, teasar
from bioimage_cpp.skeleton.postprocessing import _tangent

from skeleton_stats import build_adjacency, center_crop

BASE = "/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data"
MASK_PATH = f"{BASE}/experimental/deepict/h5/00004.h5"

FAILED_COLOR = "#ff4d6d"
REAL_COLOR = "#2ecc71"
DEADEND_COLOR = "#4d88ff"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mask_path", type=str, default=MASK_PATH, help="Binary mask as .mrc, or an .h5 volume.")
    parser.add_argument("--seg_key", type=str, default="labels/actin", help="HDF5 key of the mask, h5 input only.")
    parser.add_argument("--pixel_size", type=float, default=10.0, help="Voxel spacing in A passed to teasar.")
    parser.add_argument("--scale", type=float, default=0.0, help="teasar invalidation-radius scale.")
    parser.add_argument("--constant", type=float, default=70.0, help="teasar invalidation radius in A.")
    parser.add_argument("--number_of_threads", type=int, default=8)
    parser.add_argument("--tick_length", type=float, default=200.0, help="Dead-end length in A that is pruned.")
    parser.add_argument("--spur_length", type=float, default=200.0, help="An arm at or below this is short.")
    parser.add_argument("--crop", action="store_true", help="Use the centred crop instead of the full volume.")
    parser.add_argument("--crop_size", type=int, default=200, help="Edge length of the centred crop in voxels.")
    parser.add_argument("--direction_span", type=int, default=10, help="Tangent smoothing distance in nodes.")
    parser.add_argument("--circle_size", type=float, default=70.0, help="Drawn tube diameter in A.")
    parser.add_argument("--junction_size", type=float, default=5.0, help="Junction point size in voxels.")
    parser.add_argument("--vector_length", type=float, default=10.0, help="Drawn tangent length in voxels.")
    parser.add_argument("--no_view", action="store_true", help="Print the counts and exit without napari.")
    return parser.parse_args()


def load_volume(mask_path, seg_key):
    """Return (mask, raw). raw is None unless the input is an h5 that carries one."""
    if Path(mask_path).suffix.lower() in (".mrc", ".rec", ".map"):
        with mrcfile.open(mask_path, permissive=True) as mrc:
            return np.asarray(mrc.data) > 0, None
    with h5py.File(mask_path, "r") as f:
        if seg_key not in f:
            raise KeyError(f"'{seg_key}' not in {mask_path}")
        mask = np.asarray(f[seg_key][:]) > 0
        raw = np.asarray(f["raw"][:]) if "raw" in f else None
    return mask, raw


def walk_arm_path(node, first, indptr, dst, degrees, vertices):
    """Walk away from `node` through degree-2 nodes. Return (end node, arc length, vertices traversed)."""
    prev, cur = node, int(first)
    length = float(np.linalg.norm(vertices[cur] - vertices[node]))
    path = [node, cur]
    while degrees[cur] == 2:
        s, e = indptr[cur], indptr[cur + 1]
        nbrs = dst[s:e]
        nxt = int(nbrs[0]) if int(nbrs[0]) != prev else int(nbrs[1])
        length += float(np.linalg.norm(vertices[nxt] - vertices[cur]))
        prev, cur = cur, nxt
        path.append(cur)
    return cur, length, path


def classify_junctions(vertices, edges, spur_length):
    """Split degree-3 nodes by their shortest arm into dead-end, inter-junction, and all-arms-long.

    The three cases are disjoint and exhaustive. Returns the three node arrays plus, for the
    inter-junction case, the bridging arms keyed by the node pair they join.
    """
    indptr, dst, degrees = build_adjacency(len(vertices), edges)
    deadend, failed, real, links = [], [], [], {}
    for node in np.where(degrees == 3)[0]:
        arms = [walk_arm_path(int(node), dst[k], indptr, dst, degrees, vertices)
                for k in range(indptr[node], indptr[node + 1])]
        short = [(end, length, path) for end, length, path in arms if length <= spur_length]
        if any(degrees[end] == 1 for end, _, _ in short):
            deadend.append(int(node))
        elif short:
            failed.append(int(node))
            for end, length, path in short:
                if degrees[end] != 1:
                    links[(min(int(node), end), max(int(node), end))] = (length, path)
        else:
            real.append(int(node))
    return np.array(deadend, int), np.array(failed, int), np.array(real, int), links


def link_layer(vertices, links):
    """Edge list and per-vertex labels covering only the bridging arms, one label per link."""
    edges, labels = [], np.zeros(len(vertices), np.int64)
    for index, (_, path) in enumerate(links.values(), start=1):
        for a, b in zip(path[:-1], path[1:]):
            edges.append((a, b))
        labels[np.asarray(path, int)] = index
    return np.asarray(edges, np.int64).reshape(-1, 2), labels


def arm_vectors(nodes, graph, vertices, indptr, dst, pixel_size, direction_span, length):
    """One arrow per arm at each node, pointing along the arm, in voxel coordinates."""
    starts, directions = [], []
    for node in nodes:
        for k in range(indptr[node], indptr[node + 1]):
            starts.append(vertices[node] / pixel_size)
            directions.append(_tangent(int(node), int(dst[k]), graph, vertices, direction_span) * length)
    vectors = np.zeros((len(starts), 2, 3))
    vectors[:, 0] = np.asarray(starts)
    vectors[:, 1] = np.asarray(directions)
    return vectors


def main():
    args = parse_args()
    mask, raw = load_volume(args.mask_path, args.seg_key)
    if args.crop:
        origin = center_crop(mask.shape, args.crop_size)
        box = tuple(slice(o, o + args.crop_size) for o in origin)
        mask = mask[box]
        raw = raw[box] if raw is not None else None
    print(f"{args.mask_path}: shape {mask.shape}, foreground {int(mask.sum())} voxels")

    spacing = (args.pixel_size,) * 3
    vertices, edges, radii = teasar(
        mask.astype(np.uint8), spacing=spacing, scale=args.scale,
        constant=args.constant, number_of_threads=args.number_of_threads
    )
    vertices, edges, _ = remove_ticks(vertices, edges, args.tick_length, radii=radii)
    edges = np.asarray(edges, np.int64)

    deadend, failed, real, links = classify_junctions(vertices, edges, args.spur_length)
    total = len(deadend) + len(failed) + len(real)
    print(f"after remove_ticks {args.tick_length:.0f} A, degree-3 nodes: {total}")
    print(f"  short dead-end arm (remove_ticks should leave none): {len(deadend)}")
    print(f"  short arm to another junction (what it cannot fix):  {len(failed)}  in {len(links)} links")
    print(f"  all arms > {args.spur_length:.0f} A:                            {len(real)}")
    if links:
        lengths = np.array([length for length, _ in links.values()])
        print(f"  link length: median {np.median(lengths):.0f} A, "
              f"range {lengths.min():.0f} to {lengths.max():.0f} A")
    if args.no_view:
        return

    import napari
    viewer = napari.Viewer()
    coords = vertices / args.pixel_size
    radius = (args.circle_size / 2) / args.pixel_size
    if raw is not None:
        viewer.add_image(raw, name="raw")
    viewer.add_labels(mask.astype(np.uint8), name="mask", opacity=0.4)
    if len(edges):
        skeleton = draw_instances(coords, edges, np.zeros(len(vertices), np.int64), mask.shape, radius)
        viewer.add_labels(skeleton, name="skeleton", visible=False)

    if links:
        link_edges, link_labels = link_layer(vertices, links)
        drawn = draw_instances(coords, link_edges, link_labels, mask.shape, radius)
        viewer.add_labels(drawn, name="short links")

    def add_points(name, nodes, color, features=None):
        if not len(nodes):
            return
        viewer.add_points(coords[nodes], name=name, size=args.junction_size, face_color=color,
                          border_color="#1b1b1b", border_width=0.15, features=features,
                          out_of_slice_display=True, blending="translucent_no_depth")

    shortest = {}
    for (a, b), (length, _) in links.items():
        shortest[a] = min(shortest.get(a, np.inf), length)
        shortest[b] = min(shortest.get(b, np.inf), length)
    add_points("failed junctions", failed, FAILED_COLOR,
               features={"link_length": np.array([shortest.get(n, np.nan) for n in failed])})
    add_points("real junctions", real, REAL_COLOR)
    add_points("dead-end junctions", deadend, DEADEND_COLOR)

    if len(failed):
        indptr, dst, _ = build_adjacency(len(vertices), edges)
        graph = skeleton_to_graph(vertices, edges)
        viewer.add_vectors(
            arm_vectors(failed, graph, vertices, indptr, dst, args.pixel_size,
                        args.direction_span, args.vector_length),
            name="arm tangents", vector_style="arrow", edge_color="#ffd000",
            out_of_slice_display=True,
        )

    napari.run()


if __name__ == "__main__":
    main()
