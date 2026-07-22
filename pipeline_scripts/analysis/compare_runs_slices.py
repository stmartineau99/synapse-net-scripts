import numpy as np
import matplotlib.pyplot as plt
import h5py
from matplotlib.patches import Patch
from pathlib import Path

DATASET = "deepict"
SUBVOLUME = "00004_0"
Z = 202
TAG = "easymode"

RUNS = {
    "Easymode t=0.5": "easymode-actin",
    "Easymode t=0.07": "easymode-actin_t0.07",
    "SL Full Label": "actin-deepict-run28",
}

TP_COLOR = (0.0, 1.0, 1.0)
FP_COLOR = (1.0, 1.0, 0.0)
FN_COLOR = (1.0, 0.0, 1.0)


def diff_rgb(gt, seg):
    rgb = np.zeros((*gt.shape, 3), dtype=np.float32)
    rgb[seg & gt] = TP_COLOR
    rgb[seg & ~gt] = FP_COLOR
    rgb[~seg & gt] = FN_COLOR
    return rgb


def main():
    PARENT_DIR = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/predictions")
    h5_path = PARENT_DIR / DATASET / "subvolumes" / f"{SUBVOLUME}.h5"
    png_dir = PARENT_DIR / DATASET / "png"
    png_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as f:
        raw = f["raw"][Z]
        gt = f["labels/actin"][Z].astype(bool)
        segs = [(title, f[f"segmentations/{key}"][Z].astype(bool)) for title, key in RUNS.items()]

    lo, hi = np.percentile(raw, [1, 99])
    raw_gray = np.clip((raw - lo) / (hi - lo), 0, 1)

    panels = [("Raw", raw_gray), ("GT", gt.astype(np.float32))]
    panels += [(title, diff_rgb(gt, seg)) for title, seg in segs]

    n = len(panels)
    ncols = n if n <= 4 else 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.7 * nrows))
    for ax, (title, img) in zip(axes.flat, panels):
        if img.ndim == 3:
            ax.imshow(img)
        else:
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title, fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.flat[n:]:
        ax.axis("off")

    handles = [
        Patch(facecolor=TP_COLOR, label="TP (pred ∩ GT)"),
        Patch(facecolor=FP_COLOR, label="FP (pred ∖ GT)"),
        Patch(facecolor=FN_COLOR, label="FN (GT ∖ pred)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10)
    fig.suptitle(f"{SUBVOLUME}  z={Z}", fontsize=12, y=1.02)
    fig.patch.set_facecolor("white")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97), h_pad=3.0)

    out_path = png_dir / f"{SUBVOLUME}_z{Z}_{TAG}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
