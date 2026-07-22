import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

DATASET = "deepict"
OUT_FILE  = "deepict_runs1.png"

RUNS = {
    "Synthetic Baseline": "actin-deepict-run6.csv",
    "USDA": "actin-deepict-run24.csv",
    "SSDA Low Label": "actin-deepict-run29.csv",
    "SL Low Label": "actin-deepict-run32.csv",
    "SL Full Label": "actin-deepict-run28.csv"
}

METRICS = ["precision", "recall", "dice"]
COLORS = [plt.get_cmap("Set2")(i) for i in range(len(RUNS))]


def main():
    PARENT_DIR = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/predictions")
    csv_dir = PARENT_DIR / DATASET / "csv"
    png_dir = PARENT_DIR / DATASET / "png"
    png_dir.mkdir(parents=True, exist_ok=True)

    means = {}
    for label, fname in RUNS.items():
        df = pd.read_csv(csv_dir / fname, dtype={"tomogram": str})
        means[label] = df[METRICS].mean().to_dict()

    x = np.arange(len(METRICS))
    width = 0.15

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (label, vals) in enumerate(means.items()):
        bars = ax.bar(
            x + i * width,
            [vals[m] for m in METRICS],
            width,
            label=label,
            color=COLORS[i],
        )
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=8,
            )

    ax.set_xticks(x + width * (len(RUNS) - 1) / 2)
    ax.set_xticklabels([m.capitalize() for m in METRICS])
    ax.set_ylim(0, 1.0)
    ax.set_yticks(np.arange(0, 1.0, 0.2))
    ax.set_ylabel("Score")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")

    out_path = png_dir / OUT_FILE
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
