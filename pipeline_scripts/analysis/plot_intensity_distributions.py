import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA_ROOT = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/predictions")

DEEPICT_DIR = DATA_ROOT / "deepict"
OPTO_DIR = DATA_ROOT / "optogenetics"

N_BINS = 200


def load_masked_raw(path):
    with h5py.File(path, "r") as f:
        raw = f["raw"][:]
        mask = f["sample_mask"][:].astype(bool)
    return raw[mask].astype(np.float32)

def percentile_stats(vals):
    return {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "p01": float(np.percentile(vals, 1)),
        "p99": float(np.percentile(vals, 99)),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }

def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(direction="out", length=4, labelsize=9)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)
    ax.set_axisbelow(True)


def main():
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10})

    deepict_paths = sorted(DEEPICT_DIR.glob("*.h5"))
    opto_paths = sorted(OPTO_DIR.glob("*.h5"))

    deepict_colors = plt.cm.Blues(np.linspace(0.45, 0.85, max(len(deepict_paths), 1)))
    opto_colors = plt.cm.Oranges(np.linspace(0.35, 0.85, max(len(opto_paths), 1)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    # --- deepict ---
    ax = axes[0]
    ax.set_title("Deepict", fontsize=12, fontweight="bold")
    for p, color in zip(deepict_paths, deepict_colors):
        vals = load_masked_raw(p)
        s = percentile_stats(vals)
        print(f"[deepict] {p.stem}: mean={s['mean']:.3f}  std={s['std']:.3f}  "
              f"p01={s['p01']:.3f}  p99={s['p99']:.3f}  min={s['min']:.3f}  max={s['max']:.3f}")
        ax.hist(vals, bins=N_BINS, density=True, alpha=0.7, label=p.stem, color=color, linewidth=0)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, framealpha=0.7, edgecolor="none")
    style_ax(ax)

    # --- optogenetics ---
    ax = axes[1]
    ax.set_title("Optogenetics", fontsize=12, fontweight="bold")
    for p, color in zip(opto_paths, opto_colors):
        vals = load_masked_raw(p)
        s = percentile_stats(vals)
        print(f"[opto]    {p.stem}: mean={s['mean']:.3f}  std={s['std']:.3f}  "
              f"p01={s['p01']:.3f}  p99={s['p99']:.3f}  min={s['min']:.3f}  max={s['max']:.3f}")
        ax.hist(vals, bins=N_BINS, density=True, alpha=0.5, label=p.stem, color=color, linewidth=0)
    ax.set_xlabel("Intensity")
    ax.legend(fontsize=7, framealpha=0.7, edgecolor="none", ncol=2)
    style_ax(ax)

    fig.suptitle("Intensity distributions", fontsize=13, fontweight="bold")
    fig.tight_layout()

    out_path = DEEPICT_DIR / "intensity_distributions.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
