#!/usr/bin/env python
"""Sweep remove_ticks tick_length and report how many small instances survive.

Small fragments colocalize with degree-3 nodes, so pruning a spur before the
degree-3 split should stop that junction from producing one. Raising tick_length
also deletes real skeleton, so retained contour length is reported alongside the
fragment count; a drop in fragments only means something if length is kept.
"""
import csv
import argparse
from pathlib import Path

import numpy as np

from bioimage_cpp.graph import connected_components
from bioimage_cpp.skeleton import clean_filament_graph, skeleton_to_graph

from clean_graph import TEASAR_DIR

SKELETON_PATH = f"{TEASAR_DIR}/00004_labels_actin_skeleton.npz"
OUT_FILE = "tick_sweep"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--skeleton_path", type=str, default=SKELETON_PATH)
    parser.add_argument("--tick_lengths", type=float, nargs="+",
                        default=[0.0, 25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0])
    parser.add_argument("--small_fragment_length", type=float, default=200.0,
                        help="Instances at or below this contour length (A) count as fragments.")
    parser.add_argument("--direction_span", type=int, default=10)
    parser.add_argument("--min_branch_angle", type=float, default=20.0)
    parser.add_argument("--join_dist", type=float, default=50.0)
    parser.add_argument("--min_join_angle", type=float, default=175.0)
    parser.add_argument("--output_dir", type=str, default=TEASAR_DIR)
    return parser.parse_args()


def component_lengths(vertices, edges, labels):
    seg = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    n = int(labels.max()) + 1
    return np.bincount(labels[edges[:, 0]], weights=seg, minlength=n), float(seg.sum())


def sweep_tick_length(tick_length, vertices, edges, radii, args):
    stages = []
    clean_vertices, clean_edges, _ = clean_filament_graph(
        vertices, edges, radii=radii,
        direction_span=args.direction_span, min_branch_angle=args.min_branch_angle,
        tick_length=tick_length, join_dist=args.join_dist, min_join_angle=args.min_join_angle,
        save_intermediates=stages,
    )
    steps = {name: (v, e) for name, v, e, _ in stages}
    tick_vertices, tick_edges = steps["ticks"]
    tick_degrees = np.bincount(tick_edges.reshape(-1), minlength=len(tick_vertices))

    labels = connected_components(skeleton_to_graph(clean_vertices, clean_edges))
    comp_length, total_length = component_lengths(clean_vertices, clean_edges, labels)
    n_instances = len(comp_length)
    is_small = comp_length <= args.small_fragment_length
    n_small = int(is_small.sum())
    return {
        "tick_length": tick_length,
        "n_deg3": int((tick_degrees == 3).sum()),
        "n_deg4": int((tick_degrees == 4).sum()),
        "n_instances": n_instances,
        "n_small": n_small,
        "n_real": int((~is_small).sum()),
        "frac_small": round(n_small / n_instances, 4) if n_instances else 0.0,
        "median_length": round(float(np.median(comp_length)), 1),
        "length_in_real": round(float(comp_length[~is_small].sum()), 1),
        "total_length": round(total_length, 1),
    }


def plot_summary(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ticks = [r["tick_length"] for r in rows]
    baseline = rows[0]["total_length"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(ticks, [r["n_small"] for r in rows], marker="o", label="small fragments")
    ax1.plot(ticks, [r["n_instances"] for r in rows], marker="s", label="all instances")
    ax1.plot(ticks, [r["n_deg3"] for r in rows], marker="^", label="degree-3 nodes")
    ax1.set_xlabel("tick_length (A)")
    ax1.set_ylabel("count")
    ax1.set_title("fragments vs tick_length")
    ax1.legend()

    ax2.plot(ticks, [100.0 * r["total_length"] / baseline for r in rows], marker="o", color="#c1121f")
    ax2.set_xlabel("tick_length (A)")
    ax2.set_ylabel("retained contour length (% of tick_length=0)")
    ax2.set_title("cost of pruning")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    data = np.load(args.skeleton_path)
    vertices, edges = data["vertices"], data["edges"]
    radii = data["radii"] if "radii" in data else None
    print(f"skeleton: {len(vertices)} nodes, {len(edges)} edges")

    rows = []
    for tick_length in args.tick_lengths:
        row = sweep_tick_length(tick_length, vertices, edges, radii, args)
        rows.append(row)
        print(f"tick {row['tick_length']:>6.0f} A: deg3={row['n_deg3']:>5} "
              f"instances={row['n_instances']:>5} small={row['n_small']:>5} "
              f"({100 * row['frac_small']:>4.1f}%) median={row['median_length']:>7.1f} A "
              f"length={row['total_length'] / 1e4:>7.1f} um")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.skeleton_path).stem
    csv_path = output_dir / f"{stem}_{OUT_FILE}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")

    plot_path = output_dir / f"{stem}_{OUT_FILE}.png"
    plot_summary(rows, plot_path)
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
