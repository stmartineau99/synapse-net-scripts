import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mrcfile
import numpy as np

DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
CROP_SIZE = 400

# (display label, tag) for each config's <tomogram>_<tag>_instance_mask.mrc
CONFIGS = [
    ("Baseline", "baseline"),
    ("Eroded", "eroded"),
    ("Eroded + Greedy", "greedy"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Matplotlib comparison of instance masks across configs.")
    parser.add_argument("--tomogram", type=str, default="00004")
    parser.add_argument("--dir", type=str, default=DEFAULT_DIR)
    parser.add_argument("--mask", type=str, default=None, help="Base mask mrc (default <dir>/<tomogram>_gt_mask.mrc).")
    parser.add_argument("--z_center", type=int, default=None, help="Central z of the slab (default middle).")
    parser.add_argument("--z_thickness", type=int, default=10, help="Number of z-slices to project.")
    parser.add_argument("--out", type=str, default=None)
    return parser.parse_args()


def load_slab(mrc_path, z0, z1):
    with mrcfile.mmap(mrc_path, permissive=True, mode="r") as mrc:
        return np.asarray(mrc.data[z0:z1])


def project_labels(slab):
    """Project a label slab to 2D by taking the first non-zero label along z."""
    nz = slab != 0
    idx = nz.argmax(axis=0)
    proj = np.take_along_axis(slab, idx[None], axis=0)[0]
    proj[~nz.any(axis=0)] = 0
    return proj


def random_rgb(labels, seed=0):
    """Map integer labels to random RGB (0 -> black), returned as float image + alpha."""
    rng = np.random.default_rng(seed)
    colors = rng.uniform(0.25, 1.0, size=(int(labels.max()) + 1, 3))
    colors[0] = 0.0
    return colors[labels], (labels > 0).astype(float)


def count_instances(mrc_path):
    """Total number of instances (unique non-zero labels) in the full volume."""
    with mrcfile.mmap(mrc_path, permissive=True, mode="r") as mrc:
        vals = np.unique(np.asarray(mrc.data))
    return int((vals > 0).sum())


def main():
    args = parse_args()
    mask_path = args.mask or os.path.join(args.dir, f"{args.tomogram}_gt_mask.mrc")

    with mrcfile.mmap(mask_path, permissive=True, mode="r") as mrc:
        nz = mrc.data.shape[0]
    zc = args.z_center if args.z_center is not None else nz // 2
    z0 = max(0, zc - args.z_thickness // 2)
    z1 = min(nz, z0 + max(1, args.z_thickness))

    mask_bg = (load_slab(mask_path, z0, z1) > 0).max(axis=0)

    # centered square crop for a closeup
    ys, xs = np.where(mask_bg)
    cy = (ys.min() + ys.max()) // 2
    cx = (xs.min() + xs.max()) // 2
    y0 = min(max(0, cy - CROP_SIZE // 2), mask_bg.shape[0] - CROP_SIZE)
    x0 = min(max(0, cx - CROP_SIZE // 2), mask_bg.shape[1] - CROP_SIZE)
    y1, x1 = y0 + CROP_SIZE, x0 + CROP_SIZE
    mask_bg = mask_bg[y0:y1, x0:x1]

    present = []
    for label, tag in CONFIGS:
        p = os.path.join(args.dir, f"{args.tomogram}_{tag}_instance_mask.mrc")
        if os.path.exists(p):
            present.append((label, p))
        else:
            print(f"missing (skipping): {p}")

    fig, axes = plt.subplots(1, len(present), figsize=(6 * len(present), 6), squeeze=False)
    for ax, (label, p) in zip(axes[0], present):
        proj = project_labels(load_slab(p, z0, z1))[y0:y1, x0:x1]
        rgb, alpha = random_rgb(proj)
        ax.imshow(mask_bg, cmap="gray", interpolation="nearest")
        ax.imshow(np.dstack([rgb, alpha]), interpolation="nearest")
        ax.set_title(f"{label} ({count_instances(p)} instances)", fontsize=12)
        ax.axis("off")

    fig.suptitle(f"{args.tomogram}  instance segmentation  (z{z0}-{z1})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    out_path = args.out or os.path.join(
        args.dir, f"instance_comparison_{args.tomogram}_z{zc}_t{args.z_thickness}.png"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
