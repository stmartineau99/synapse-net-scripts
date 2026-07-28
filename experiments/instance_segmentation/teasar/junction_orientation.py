#!/usr/bin/env python
"""Test whether TEASAR degree-3 nodes are voxel-grid staircase artifacts.

A line on a voxel grid avoids a staircase only when its direction is one voxel step
repeated: an axis (1,0,0), a face diagonal (1,1,0), or a body diagonal (1,1,1). Ignoring
sign there are 13 such directions. Any other direction must alternate step types, and each
turn is a corner where skeletonization can spawn a stub. `theta` is the angle from a
filament direction to the nearest of those 13, so theta=0 means no staircase is possible
and the largest theta is about 19.5 deg, near (2,1,1).

The hypothesis predicts the degree-3 rate rises with theta, and that the spacing between
successive degree-3 nodes along one filament scales as 1/tan(theta).

Vertex coordinates are physical (z, y, x) in Angstrom.
"""
import csv
import argparse
import itertools
from pathlib import Path

import numpy as np

from bioimage_cpp.skeleton import skeleton_to_graph
from bioimage_cpp.skeleton.postprocessing import _tangent, _pair_angle

from clean_graph import TEASAR_DIR

SKELETON_PATH = f"{TEASAR_DIR}/00004_labels_actin_skeleton.npz"
OUT_FILE = "junction_orientation"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skeleton_path", type=str, default=SKELETON_PATH)
    parser.add_argument("--direction_span", type=int, default=10)
    parser.add_argument("--pixel_size", type=float, default=10.0)
    parser.add_argument("--invalidation_radius", type=float, default=70.0,
                        help="teasar --constant, the floor on branch-point separation.")
    parser.add_argument("--sample", type=int, default=20000, help="Degree-2 nodes sampled as the baseline.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=TEASAR_DIR)
    return parser.parse_args()


def lattice_directions():
    """The 13 sign-independent directions a repeated voxel step can follow."""
    dirs = []
    for combo in itertools.product((-1, 0, 1), repeat=3):
        if not any(combo):
            continue
        d = np.array(combo, dtype=np.float64)
        d /= np.linalg.norm(d)
        if not any(abs(abs(d @ k) - 1.0) < 1e-9 for k in dirs):
            dirs.append(d)
    return np.stack(dirs)


def off_axis_angle(direction, lattice):
    """Angle in degrees to the nearest lattice direction; 0 means no staircase."""
    norm = np.linalg.norm(direction)
    if norm == 0:
        return np.nan
    cos = np.abs(lattice @ (direction / norm))
    return float(np.degrees(np.arccos(np.clip(cos.max(), -1.0, 1.0))))


def adjacency(n_nodes, edges):
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    order = np.argsort(src, kind="stable")
    dst = dst[order]
    degrees = np.bincount(src, minlength=n_nodes)
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(degrees, out=indptr[1:])
    return indptr, dst, degrees


def walk_arm(node, first, indptr, dst, degrees, vertices):
    """Walk a degree-2 chain to the next critical node. Returns (end, steps, length)."""
    prev, cur = node, int(first)
    steps = 1
    length = float(np.linalg.norm(vertices[cur] - vertices[node]))
    while degrees[cur] == 2:
        s, e = indptr[cur], indptr[cur + 1]
        nbrs = dst[s:e]
        nxt = int(nbrs[0]) if int(nbrs[0]) != prev else int(nbrs[1])
        length += float(np.linalg.norm(vertices[nxt] - vertices[cur]))
        prev, cur = cur, nxt
        steps += 1
    return cur, steps, length


def nearest_deg3(node, indptr, dst, degrees, vertices):
    """Shortest arm from a degree-3 node that lands on another degree-3 node."""
    best = (np.nan, np.nan)
    for k in range(indptr[node], indptr[node + 1]):
        end, steps, length = walk_arm(node, dst[k], indptr, dst, degrees, vertices)
        if degrees[end] == 3 and (np.isnan(best[1]) or length < best[1]):
            best = (steps, length)
    return best


def junction_geometry(v, graph, vertices, span, lattice):
    """Off-axis angle of the straightest arm pair, and the odd arm's first step length."""
    adj = np.asarray(graph.node_adjacency(int(v)))
    neighbors = adj[:, 0]
    dirs = np.stack([_tangent(int(v), n, graph, vertices, span) for n in neighbors])
    i, j = max([(0, 1), (0, 2), (1, 2)], key=lambda p: _pair_angle(dirs[p[0]], dirs[p[1]]))
    odd = ({0, 1, 2} - {i, j}).pop()
    through = dirs[i] - dirs[j]
    odd_step = float(np.linalg.norm(vertices[int(neighbors[odd])] - vertices[int(v)]))
    return off_axis_angle(through, lattice), odd_step


def report_spacing(steps, lengths, invalidation_radius):
    finite = lengths[~np.isnan(lengths)]
    print("1. spacing to nearest degree-3 node along the filament")
    print(f"degree-3 nodes with another on an arm: {len(finite)} of {len(lengths)}")
    if not len(finite):
        return
    for q in (10, 25, 50, 75, 90):
        print(f"p{q}: {np.percentile(finite, q):7.1f} A")
    below = int((finite < invalidation_radius).sum())
    print(f"below invalidation radius {invalidation_radius:.0f} A: {below} "
          f"({100 * below / len(finite):.1f}%)")
    st = steps[~np.isnan(steps)]
    print(f"steps: p50={np.percentile(st, 50):.0f} p90={np.percentile(st, 90):.0f}")


def report_rate(theta_deg3, theta_deg2, n_deg2_total, sample_size):
    print("2. degree-3 rate against off-axis angle theta")
    edges = np.array([0, 2, 4, 6, 8, 10, 12, 14, 16, 20])
    scale = n_deg2_total / sample_size
    print(f"{'theta (deg)':>12} {'deg3':>7} {'deg2 est':>9} {'rate %':>7}")
    for lo, hi in zip(edges[:-1], edges[1:]):
        n3 = int(((theta_deg3 >= lo) & (theta_deg3 < hi)).sum())
        n2 = int(((theta_deg2 >= lo) & (theta_deg2 < hi)).sum()) * scale
        rate = 100 * n3 / n2 if n2 else np.nan
        print(f"{lo:>5}-{hi:<6} {n3:>7} {n2:>9.0f} {rate:>7.2f}")


def report_spacing_vs_theta(theta, lengths, pixel_size):
    print("3. spacing against theta, staircase predicts spacing ~ pixel / tan(theta)")
    ok = ~np.isnan(lengths) & ~np.isnan(theta) & (theta > 0)
    t, L = theta[ok], lengths[ok]
    print(f"{'theta (deg)':>12} {'n':>6} {'median spacing':>15} {'predicted':>10}")
    for lo, hi in [(0, 4), (4, 8), (8, 12), (12, 16), (16, 20)]:
        m = (t >= lo) & (t < hi)
        if not m.any():
            continue
        mid = np.deg2rad((lo + hi) / 2)
        print(f"{lo:>5}-{hi:<6} {int(m.sum()):>6} {np.median(L[m]):>15.1f} "
              f"{pixel_size / np.tan(mid):>10.1f}")
    if len(t) > 2:
        r = np.corrcoef(1.0 / np.tan(np.deg2rad(t)), L)[0, 1]
        print(f"correlation of spacing with 1/tan(theta): {r:+.3f}")


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    lattice = lattice_directions()
    data = np.load(args.skeleton_path)
    vertices, edges = data["vertices"], data["edges"]
    graph = skeleton_to_graph(vertices, edges)
    indptr, dst, degrees = adjacency(len(vertices), edges)
    deg3 = np.where(degrees == 3)[0]
    print(f"skeleton: {len(vertices)} nodes, {len(edges)} edges, {len(deg3)} degree-3")
    print(f"lattice directions: {len(lattice)}")

    spacing_steps = np.full(len(deg3), np.nan)
    spacing_len = np.full(len(deg3), np.nan)
    theta = np.full(len(deg3), np.nan)
    odd_step = np.full(len(deg3), np.nan)
    for k, v in enumerate(deg3):
        spacing_steps[k], spacing_len[k] = nearest_deg3(int(v), indptr, dst, degrees, vertices)
        theta[k], odd_step[k] = junction_geometry(int(v), graph, vertices, args.direction_span, lattice)

    deg2 = np.where(degrees == 2)[0]
    sample = rng.choice(deg2, size=min(args.sample, len(deg2)), replace=False)
    theta_deg2 = []
    for v in sample:
        nbrs = np.asarray(graph.node_adjacency(int(v)))[:, 0]
        d = _tangent(int(v), int(nbrs[0]), graph, vertices, args.direction_span)
        theta_deg2.append(off_axis_angle(d, lattice))
    theta_deg2 = np.array(theta_deg2)

    print()
    report_spacing(spacing_steps, spacing_len, args.invalidation_radius)
    print()
    report_rate(theta, theta_deg2, len(deg2), len(sample))
    print()
    report_spacing_vs_theta(theta, spacing_len, args.pixel_size)
    print()
    print("4. odd arm first step length, Angstrom (control, expect 10, 14.1, 17.3)")
    print(f"p50={np.percentile(odd_step, 50):.1f} p90={np.percentile(odd_step, 90):.1f} "
          f"max={odd_step.max():.1f}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.skeleton_path).stem
    csv_path = output_dir / f"{stem}_{OUT_FILE}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node", "spacing_steps", "spacing_A", "theta_deg", "odd_step_A"])
        for v, s, L, t, o in zip(deg3, spacing_steps, spacing_len, theta, odd_step):
            writer.writerow([int(v), s, round(float(L), 2) if not np.isnan(L) else "",
                             round(float(t), 3), round(float(o), 2)])
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
