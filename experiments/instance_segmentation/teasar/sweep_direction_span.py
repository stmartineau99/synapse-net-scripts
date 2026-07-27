#!/usr/bin/env python
import argparse
import csv
from pathlib import Path

import numpy as np

from bioimage_cpp.skeleton import skeleton_to_graph
from bioimage_cpp.skeleton.postprocessing import _tangent, _pair_angle

from clean_graph import TEASAR_DIR

SKELETON_PATH = f"{TEASAR_DIR}/00004_labels_actin_skeleton.npz"
OUT_FILE = "span_sweep"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep direction_span and report junction through-angle percentiles."
    )
    parser.add_argument("--skeleton_path", type=str, default=SKELETON_PATH)
    parser.add_argument("--spans", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12, 16, 20])
    parser.add_argument("--output_dir", type=str, default=TEASAR_DIR)
    return parser.parse_args()


def _through_angle(v, graph, vertices, direction_span):
    neighbors = np.asarray(graph.node_adjacency(int(v)))[:, 0]
    dirs = np.stack([_tangent(int(v), n, graph, vertices, direction_span) for n in neighbors])
    best = -1.0
    for i in range(len(dirs)):
        for j in range(i + 1, len(dirs)):
            best = max(best, _pair_angle(dirs[i], dirs[j]))
    return best


def sweep_span(span, graph, vertices, junctions):
    angles = np.array([_through_angle(int(v), graph, vertices, span) for v in junctions])
    return {
        "direction_span": span,
        "n_junctions": int(len(angles)),
        "p50": round(float(np.percentile(angles, 50)), 2),
        "p75": round(float(np.percentile(angles, 75)), 2),
        "p90": round(float(np.percentile(angles, 90)), 2),
        "p95": round(float(np.percentile(angles, 95)), 2),
    }


def plot_summary(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    spans = [r["direction_span"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    for key in ("p50", "p75", "p90", "p95"):
        ax.plot(spans, [r[key] for r in rows], marker="o", label=key)
    ax.set_xlabel("direction_span")
    ax.set_ylabel("junction through-angle (deg)")
    ax.set_title("through-angle vs direction_span")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    data = np.load(args.skeleton_path)
    vertices, edges = data["vertices"], data["edges"]
    graph = skeleton_to_graph(vertices, edges)
    degrees = np.bincount(edges.reshape(-1), minlength=len(vertices))
    junctions = np.where(np.isin(degrees, (3, 4)))[0]
    if len(junctions) == 0:
        raise SystemExit("no degree-3/4 junctions in skeleton")

    rows = []
    for span in args.spans:
        row = sweep_span(span, graph, vertices, junctions)
        rows.append(row)
        print(f"span {span:>3}: p50={row['p50']:6.1f}  p75={row['p75']:6.1f}  "
              f"p90={row['p90']:6.1f}  p95={row['p95']:6.1f}")

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
