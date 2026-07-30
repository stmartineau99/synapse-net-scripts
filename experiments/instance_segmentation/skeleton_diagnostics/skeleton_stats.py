"""Skeleton graph statistics, shared by the TEASAR and kimimaro diagnostics.

Both diagnostics must report identical measurements to be comparable, so the statistics live here rather
than in each script. Nothing in this module imports a skeletonisation library, which is what lets it be
imported from either the synapse-net or the tardis-em environment.

Arm lengths and the spur length are in the same units as the vertex coordinates, which are Angstrom when
the voxel spacing is passed to the skeletonisation call.
"""
import csv

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

FIELDS = ["method", "volume", "stage", "foreground", "vertices", "edges", "length_um", "n_components",
          "degree_1", "degree_3", "degree_4", "arm_p25", "arm_p50", "arm_p75", "arm_p90",
          "n_spurs", "n_spurs_pct", "real_junctions", "real_junction_pct"]


def center_crop(shape, size):
    """Origin of the centred box of the given size."""
    return tuple(max(0, (extent - size) // 2) for extent in shape)


def build_adjacency(n_nodes, edges):
    """CSR-style neighbour lists and node degrees from an undirected edge list."""
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    order = np.argsort(src, kind="stable")
    dst = dst[order]
    degrees = np.bincount(src, minlength=n_nodes)
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(degrees, out=indptr[1:])
    return indptr, dst, degrees


def walk_arm(node, first, indptr, dst, degrees, vertices):
    """Walk away from `node` through degree-2 chain nodes. Return (end node, arc length)."""
    prev, cur = node, int(first)
    length = float(np.linalg.norm(vertices[cur] - vertices[node]))
    while degrees[cur] == 2:
        s, e = indptr[cur], indptr[cur + 1]
        nbrs = dst[s:e]
        nxt = int(nbrs[0]) if int(nbrs[0]) != prev else int(nbrs[1])
        length += float(np.linalg.norm(vertices[nxt] - vertices[cur]))
        prev, cur = cur, nxt
    return cur, length


def component_count(n_nodes, edges):
    """Number of connected components, computed the same way for every method."""
    if not len(edges):
        return n_nodes
    data = np.ones(len(edges), dtype=np.int8)
    graph = coo_matrix((data, (edges[:, 0], edges[:, 1])), shape=(n_nodes, n_nodes))
    return int(connected_components(graph, directed=False, return_labels=False))


def skeleton_stats(vertices, edges, spur_length):
    """Degree counts and degree-3 junction quality for one skeleton graph.

    A real junction has all three arms longer than `spur_length`, so it is a plausible filament crossing
    rather than a skeletonisation artifact. `n_spurs` counts degree-3 nodes with at least one dead-end arm
    at or below that length.
    """
    n = len(vertices)
    edges = np.asarray(edges, dtype=np.int64)
    indptr, dst, degrees = build_adjacency(n, edges)
    seg = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1) if len(edges) else np.zeros(0)
    deg3 = np.where(degrees == 3)[0]

    row = {
        "vertices": n,
        "edges": len(edges),
        "length_um": float(seg.sum()) / 1e4,
        "n_components": component_count(n, edges),
        "degree_1": int((degrees == 1).sum()),
        "degree_3": len(deg3),
        "degree_4": int((degrees == 4).sum()),
    }

    shortest, real_junctions, n_spurs = [], 0, 0
    for node in deg3:
        arms = [walk_arm(int(node), dst[k], indptr, dst, degrees, vertices)
                for k in range(indptr[node], indptr[node + 1])]
        lengths = [length for _, length in arms]
        shortest.append(min(lengths))
        if min(lengths) > spur_length:
            real_junctions += 1
        if any(degrees[end] == 1 and length <= spur_length for end, length in arms):
            n_spurs += 1

    shortest = np.asarray(shortest)
    for q in (25, 50, 75, 90):
        row[f"arm_p{q}"] = float(np.percentile(shortest, q)) if shortest.size else float("nan")
    row["real_junctions"] = real_junctions
    row["n_spurs"] = n_spurs
    row["real_junction_pct"] = 100.0 * real_junctions / len(deg3) if len(deg3) else float("nan")
    row["n_spurs_pct"] = 100.0 * n_spurs / len(deg3) if len(deg3) else float("nan")
    return row


def print_rows(rows, spur_length):
    """One block per stage, laid out as the README table is: measurements down, stages across."""
    labels = [
        ("total length", "length_um", "{:.1f} um"),
        ("n_components", "n_components", "{:d}"),
        ("degree 1", "degree_1", "{:d}"),
        ("degree 3", "degree_3", "{:d}"),
        ("degree 4", "degree_4", "{:d}"),
        ("shortest arm p25", "arm_p25", "{:.0f} A"),
        ("shortest arm p50", "arm_p50", "{:.0f} A"),
        ("shortest arm p75", "arm_p75", "{:.0f} A"),
        ("shortest arm p90", "arm_p90", "{:.0f} A"),
    ]
    width = max(len(label) for label, _, _ in labels)
    header = "".join(f"{row['stage']:>26}" for row in rows)
    print(f"\n{'measurement':<{width}}{header}")
    for label, key, fmt in labels:
        cells = "".join(f"{fmt.format(row[key]):>26}" for row in rows)
        print(f"{label:<{width}}{cells}")
    for label, count_key, pct_key in (("n_spurs", "n_spurs", "n_spurs_pct"),
                                      ("real junctions", "real_junctions", "real_junction_pct")):
        cells = "".join(f"{f'{row[count_key]} ({row[pct_key]:.1f}%)':>26}" for row in rows)
        print(f"{label:<{width}}{cells}")
    print(f"\nn_spurs: degree-3 node with a dead-end arm <= {spur_length:.0f} A, share of degree-3 nodes")
    print(f"real junction: degree-3 node with all three arms > {spur_length:.0f} A, share of degree-3 nodes")


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS})
