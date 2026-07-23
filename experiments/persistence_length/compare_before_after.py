"""Before/after instance comparison for one tomogram: GT mask, instances before, after.

Rasterize each instance npz to a label volume with draw_instances, then reuse the crop and
slab projection of experiments/instance_segmentation/compare_instances_plot.py.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import h5py
import numpy as np

from bioimage_cpp.skeleton import draw_instances

CROP_SIZE = 400


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", type=str, required=True)
    parser.add_argument("--seg_key", type=str, default="labels/actin")
    parser.add_argument("--before_npz", type=str, required=True)
    parser.add_argument("--after_npz", type=str, required=True)
    parser.add_argument("--pixel_size", type=float, default=10.0)
    parser.add_argument("--circle_size", type=float, default=10.0)
    parser.add_argument("--z_center", type=int, default=None)
    parser.add_argument("--z_thickness", type=int, default=10)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def project_labels(slab):
    nz = slab != 0
    idx = nz.argmax(axis=0)
    proj = np.take_along_axis(slab, idx[None], axis=0)[0]
    proj[~nz.any(axis=0)] = 0
    return proj


def random_rgb(labels, seed=0):
    rng = np.random.default_rng(seed)
    colors = rng.uniform(0.25, 1.0, size=(int(labels.max()) + 1, 3))
    colors[0] = 0.0
    return colors[labels], (labels > 0).astype(float)


def instance_volume(npz_path, shape, pixel_size, radius):
    data = np.load(npz_path)
    return draw_instances(data["vertices"] / pixel_size, data["edges"], data["labels"], shape, radius)


def main():
    args = parse_args()
    with h5py.File(args.h5, "r") as f:
        mask = np.asarray(f[args.seg_key][:])
    shape = mask.shape
    radius = (args.circle_size / 2) / args.pixel_size

    before = instance_volume(args.before_npz, shape, args.pixel_size, radius)
    after = instance_volume(args.after_npz, shape, args.pixel_size, radius)

    nz = shape[0]
    zc = args.z_center if args.z_center is not None else nz // 2
    z0 = max(0, zc - args.z_thickness // 2)
    z1 = min(nz, z0 + max(1, args.z_thickness))

    mask_bg = (mask[z0:z1] > 0).max(axis=0)
    ys, xs = np.where(mask_bg)
    cy = (ys.min() + ys.max()) // 2
    cx = (xs.min() + xs.max()) // 2
    y0 = min(max(0, cy - CROP_SIZE // 2), mask_bg.shape[0] - CROP_SIZE)
    x0 = min(max(0, cx - CROP_SIZE // 2), mask_bg.shape[1] - CROP_SIZE)
    y1, x1 = y0 + CROP_SIZE, x0 + CROP_SIZE
    mask_bg = mask_bg[y0:y1, x0:x1]

    panels = [("GT mask", None), ("instances before", before), ("instances after", after)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), squeeze=False)
    for ax, (label, volume) in zip(axes[0], panels):
        ax.imshow(mask_bg, cmap="gray", interpolation="nearest")
        if volume is not None:
            proj = project_labels(volume[z0:z1])[y0:y1, x0:x1]
            rgb, alpha = random_rgb(proj)
            ax.imshow(np.dstack([rgb, alpha]), interpolation="nearest")
            count = int((np.unique(volume) > 0).sum())
            ax.set_title(f"{label} ({count} instances)", fontsize=12)
        else:
            ax.set_title(label, fontsize=12)
        ax.axis("off")

    fig.suptitle(f"{Path(args.h5).stem} instance segmentation (z{z0}-{z1})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
