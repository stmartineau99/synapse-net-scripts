#!/usr/bin/env python
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import numpy as np

from bioimage_cpp.skeleton import clean_filament_graph, draw_instances, skeleton_to_graph, teasar
from bioimage_cpp.graph import connected_components

BASE = "/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data"
MASK_PATH = f"{BASE}/experimental/deepict/h5/00004.h5"
OUTPUT_DIR = Path(__file__).resolve().parent
OUT_FILE = "postprocessing_comparison"
CROP_SIZE = 400


def parse_args():
    parser = argparse.ArgumentParser(description="Matplotlib comparison of instance masks across postprocessing stages.")
    parser.add_argument("--mask_path", type=str, default=MASK_PATH)
    parser.add_argument("--seg_key", type=str, default="labels/actin")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--crop_size", type=int, default=CROP_SIZE, help="Side length of the centered cubic crop.")
    parser.add_argument("--z_center", type=int, default=None, help="Central z of the slab (default middle).")
    parser.add_argument("--z_thickness", type=int, default=10, help="Number of z-slices to project.")
    parser.add_argument("--pixel_size", type=float, default=10.0)
    parser.add_argument("--scale", type=float, default=0)
    parser.add_argument("--constant", type=float, default=70.0)
    parser.add_argument("--number_of_threads", type=int, default=8)
    parser.add_argument("--direction_span", type=int, default=10)
    parser.add_argument("--min_through_angle", type=float, default=170.0)
    parser.add_argument("--min_branch_angle", type=float, default=30.0)
    parser.add_argument("--tick_length", type=float, default=50.0)
    parser.add_argument("--join_dist", type=float, default=50.0)
    parser.add_argument("--min_join_angle", type=float, default=175.0)
    parser.add_argument("--circle_size", type=float, default=40.0, help="Instance tube diameter in A.")
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()


def load_crop(mask_path, seg_key, crop_size, z_center, z_thickness):
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

        mask = np.asarray(dset[:, y0:y0 + crop_size, x0:x0 + crop_size])
    return mask, zc, z0, z1


def n_components(vertices, edges):
    if len(edges) == 0:
        return np.zeros(len(vertices), dtype=np.int64), len(vertices)
    labels = connected_components(skeleton_to_graph(vertices, edges))
    return labels, len(np.unique(labels))


def build_stages(vertices, edges, radii, args):
    """Return [(label, vertices, edges, labels, n_instances)] for each stage.

    Runs clean_filament_graph once, collecting its intermediate graphs via
    save_intermediates, so the panels match the real pipeline order.
    """
    intermediates = []
    clean_filament_graph(
        vertices, edges, radii=radii,
        direction_span=args.direction_span, min_through_angle=args.min_through_angle,
        min_branch_angle=args.min_branch_angle, tick_length=args.tick_length,
        join_dist=args.join_dist, min_join_angle=args.min_join_angle,
        save_intermediates=intermediates,
    )
    steps = {name: (v, e) for name, v, e, _ in intermediates}

    raw_v, raw_e = steps["raw"]
    raw_labels, n_raw = n_components(raw_v, raw_e)
    stages = [
        ("1. Raw teasar skeleton", raw_v, raw_e, np.zeros(len(raw_v), np.int64), n_raw),
        ("2. Skeleton instances", raw_v, raw_e, raw_labels, n_raw),
    ]
    for title, key in [
        ("3. After remove ticks", "ticks"),
        ("4. After degree 3/4 graph cut", "split"),
    ]:
        v, e = steps[key]
        labels, n = n_components(v, e)
        stages.append((title, v, e, labels, n))
    return stages


def project_labels(volume):
    """Project a label volume to 2D by taking the first non-zero label along z."""
    nz = volume != 0
    idx = nz.argmax(axis=0)
    proj = np.take_along_axis(volume, idx[None], axis=0)[0]
    proj[~nz.any(axis=0)] = 0
    return proj


def random_rgb(labels, seed=0):
    """Map integer labels to random RGB (0 -> black), returned as float image + alpha."""
    rng = np.random.default_rng(seed)
    colors = rng.uniform(0.25, 1.0, size=(int(labels.max()) + 1, 3))
    colors[0] = 0.0
    return colors[labels], (labels > 0).astype(float)


def main():
    args = parse_args()
    mask, zc, z0, z1 = load_crop(
        args.mask_path, args.seg_key, args.crop_size, args.z_center, args.z_thickness)
    mask_bg = (mask[z0:z1] > 0).max(axis=0)

    binary = (mask > 0).astype(np.uint8)
    spacing = (args.pixel_size,) * 3
    vertices, edges, radii = teasar(
        binary, spacing=spacing, scale=args.scale, constant=args.constant,
        number_of_threads=args.number_of_threads,
    )

    stages = build_stages(vertices, edges, radii, args)
    for label, _, _, _, n in stages:
        print(f"{label}: {n} instances")

    ps = args.pixel_size
    radius = (args.circle_size / 2) / ps
    fig, axes = plt.subplots(1, len(stages), figsize=(4.5 * len(stages), 6), squeeze=False)
    for ax, (label, verts, eds, labs, n) in zip(axes[0], stages):
        ax.imshow(mask_bg, cmap="gray", interpolation="nearest")
        if len(eds):
            vol = draw_instances(verts / ps, eds, labs, mask.shape, radius)
            rgb, alpha = random_rgb(project_labels(vol[z0:z1]))
            ax.imshow(np.dstack([rgb, alpha]), interpolation="nearest")
        ax.set_title(f"{label}\n({n} instances)", fontsize=12)
        ax.axis("off")

    mask_path = Path(args.mask_path)
    fig.suptitle(
        f"{mask_path.stem}  postprocessing stages  "
        f"(z{z0}-{z1})", fontsize=14, y=0.99)
    fig.subplots_adjust(left=0.005, right=0.995, top=0.84, bottom=0.02, wspace=0.04)

    out_path = args.out or Path(args.output_dir) / f"{OUT_FILE}_{mask_path.stem}_z{zc}_t{args.z_thickness}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
