#!/usr/bin/env python
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import numpy as np
from scipy.spatial import cKDTree

from bioimage_cpp.skeleton import clean_graph, skeleton_to_graph, teasar
from bioimage_cpp.graph import connected_components

BASE = "/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data"
MASK_PATH = f"{BASE}/experimental/deepict/h5/00004.h5"
OUTPUT_DIR = Path(__file__).resolve().parent
OUT_FILE = "fragment_profile"
CROP_SIZE = 400


def parse_args():
    parser = argparse.ArgumentParser(
        description="Profile short skeleton fragments and their parallel overlap with longer filaments.")
    parser.add_argument("--mask_path", type=str, default=MASK_PATH)
    parser.add_argument("--seg_key", type=str, default="labels/actin")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--crop_size", type=int, default=CROP_SIZE, help="Side length of the centered cubic crop.")
    parser.add_argument("--z_center", type=int, default=None, help="Central z of the slab (default middle).")
    parser.add_argument("--z_thickness", type=int, default=10, help="Slab thickness for the crop-centering projection.")
    parser.add_argument("--pixel_size", type=float, default=10.0)
    parser.add_argument("--scale", type=float, default=0)
    parser.add_argument("--constant", type=float, default=70.0)
    parser.add_argument("--number_of_threads", type=int, default=8)
    parser.add_argument("--direction_span", type=int, default=10)
    parser.add_argument("--min_through_angle", type=float, default=170.0)
    parser.add_argument("--min_branch_angle", type=float, default=30.0)
    parser.add_argument("--tick_length", type=float, default=50.0)
    parser.add_argument("--overlap_radius", type=float, default=50.0,
                        help="Max centerline distance (A) for a fragment node to count as overlapping a longer one.")
    parser.add_argument("--small_length", type=float, default=200.0,
                        help="Length (A) below which a component is flagged as a short fragment (plots only).")
    return parser.parse_args()


def load_crop(mask_path, seg_key, crop_size, z_center, z_thickness):
    """Crop a crop_size x crop_size xy column (full z), centered on the mask."""
    with h5py.File(mask_path, "r") as f:
        if seg_key not in f:
            raise KeyError(f"'{seg_key}' not in {mask_path}")
        dset = f[seg_key]
        nz, ny, nx = dset.shape
        zc = z_center if z_center is not None else nz // 2
        z0 = max(0, zc - z_thickness // 2)
        z1 = min(nz, z0 + max(1, z_thickness))

        slab_bg = (np.asarray(dset[z0:z1]) > 0).max(axis=0)
        ys, xs = np.where(slab_bg)
        cy = (ys.min() + ys.max()) // 2
        cx = (xs.min() + xs.max()) // 2
        y0 = min(max(0, cy - crop_size // 2), ny - crop_size)
        x0 = min(max(0, cx - crop_size // 2), nx - crop_size)

        return np.asarray(dset[:, y0:y0 + crop_size, x0:x0 + crop_size])


def split_graph(mask, args):
    """Skeletonize and return the split-stage graph (per-filament components)."""
    binary = (mask > 0).astype(np.uint8)
    spacing = (args.pixel_size,) * 3
    vertices, edges, radii = teasar(
        binary, spacing=spacing, scale=args.scale, constant=args.constant,
        number_of_threads=args.number_of_threads)
    stages = []
    clean_graph(
        vertices, edges, radii=radii,
        direction_span=args.direction_span, min_through_angle=args.min_through_angle,
        min_branch_angle=args.min_branch_angle, tick_length=args.tick_length,
        join_dist=0.0, save_intermediates=stages)
    return {name: (v, e) for name, v, e, _ in stages}["split"]


def profile_components(vertices, edges, overlap_radius, neighbors=24):
    """One dict per connected component: length and colocation-overlap with longer ones."""
    n = len(vertices)
    labels = connected_components(skeleton_to_graph(vertices, edges))
    n_comp = int(labels.max()) + 1

    edge_vec = vertices[edges[:, 0]] - vertices[edges[:, 1]]
    edge_len = np.linalg.norm(edge_vec, axis=1)
    comp_len = np.zeros(n_comp)
    np.add.at(comp_len, labels[edges[:, 0]], edge_len)
    comp_nodes = np.bincount(labels, minlength=n_comp).astype(float)

    tree = cKDTree(vertices)
    k = min(neighbors, n)
    dist, idx = tree.query(vertices, k=k)

    # a node overlaps if a node of a longer component lies within overlap_radius
    neighbor_comp = labels[idx]
    valid = (dist <= overlap_radius) & (neighbor_comp != labels[:, None]) & \
            (comp_len[neighbor_comp] > comp_len[labels][:, None])
    overlap = valid.any(axis=1)
    nearest = np.where(valid, dist, np.inf).argmin(axis=1)
    rows_n = np.arange(n)
    matched_gap = dist[rows_n, nearest]
    matched_len = comp_len[labels[idx[rows_n, nearest]]]

    overlap_count = np.bincount(labels, weights=overlap, minlength=n_comp)
    overlap_fraction = overlap_count / comp_nodes
    gap_sum = np.bincount(labels[overlap], weights=matched_gap[overlap], minlength=n_comp)
    len_sum = np.bincount(labels[overlap], weights=matched_len[overlap], minlength=n_comp)
    ov_nodes = np.bincount(labels[overlap], minlength=n_comp).astype(float)
    mean_gap = np.divide(gap_sum, ov_nodes, out=np.full(n_comp, np.nan), where=ov_nodes > 0)
    longer_len = np.divide(len_sum, ov_nodes, out=np.full(n_comp, np.nan), where=ov_nodes > 0)

    rows = []
    for c in range(n_comp):
        rows.append({
            "component_id": c,
            "n_nodes": int(comp_nodes[c]),
            "length_A": round(float(comp_len[c]), 1),
            "overlap_fraction": round(float(overlap_fraction[c]), 3),
            "mean_gap_A": round(float(mean_gap[c]), 1) if ov_nodes[c] > 0 else "",
            "longer_len_A": round(float(longer_len[c]), 1) if ov_nodes[c] > 0 else "",
            "length_ratio": round(float(comp_len[c] / longer_len[c]), 3) if ov_nodes[c] > 0 else "",
        })
    return rows


def summary_plot(rows, small_length, path):
    lengths = np.array([r["length_A"] for r in rows])
    fractions = np.array([r["overlap_fraction"] for r in rows])
    short = lengths < small_length

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    positive = lengths[lengths > 0]
    bins = np.logspace(np.log10(max(positive.min(), 1)), np.log10(positive.max()), 40)
    axes[0].hist(positive, bins=bins, color="#4d88ff")
    axes[0].axvline(small_length, color="k", ls="--", lw=1)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("component length (A)")
    axes[0].set_ylabel("count")
    axes[0].set_title("component length distribution")

    axes[1].scatter(lengths[~short], fractions[~short], s=8, color="#999999", label="long")
    axes[1].scatter(lengths[short], fractions[short], s=8, color="#ff4d6d", label="short")
    axes[1].axvline(small_length, color="k", ls="--", lw=1)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("component length (A)")
    axes[1].set_ylabel("overlap fraction")
    axes[1].set_title("length vs overlap")
    axes[1].legend()

    axes[2].hist(fractions[short], bins=np.linspace(0, 1, 21), color="#ff4d6d")
    axes[2].set_xlabel("overlap fraction")
    axes[2].set_ylabel("count")
    axes[2].set_title(f"short fragments (< {small_length:.0f} A)")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    mask = load_crop(args.mask_path, args.seg_key, args.crop_size, args.z_center, args.z_thickness)
    vertices, edges = split_graph(mask, args)
    rows = profile_components(vertices, edges, args.overlap_radius)

    lengths = np.array([r["length_A"] for r in rows])
    fractions = np.array([r["overlap_fraction"] for r in rows])
    short = lengths < args.small_length
    print(f"components: {len(rows)}  short (< {args.small_length:.0f} A): {int(short.sum())}")
    if short.any():
        print(f"short-fragment overlap fraction: median={np.median(fractions[short]):.2f}  "
              f">=0.9: {int((fractions[short] >= 0.9).sum())}  ==0: {int((fractions[short] == 0).sum())}")

    tomogram = Path(args.mask_path).stem
    tag = f"{tomogram}_r{args.overlap_radius:.0f}"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"{OUT_FILE}_{tag}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")

    plot_path = output_dir / f"{OUT_FILE}_{tag}.png"
    summary_plot(rows, args.small_length, plot_path)
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
