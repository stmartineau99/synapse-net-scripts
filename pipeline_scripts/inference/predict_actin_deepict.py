import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
from pathlib import Path
from synapse_net.inference.actin import segment_actin


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    return parser.parse_args()


def compute_metrics(seg, gt, eps=1e-7):
    seg = seg.astype(bool)
    gt = gt.astype(bool)
    tp = np.logical_and(seg, gt).sum()
    fp = np.logical_and(seg, ~gt).sum()
    fn = np.logical_and(~seg, gt).sum()
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    dice = (2 * precision * recall) / (precision + recall)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "dice": float(dice),
    }

def save_metrics_png(df, model_name, out_path):
    metrics = ["precision", "recall", "dice"]

    mean_row = df[metrics].mean().to_frame().T
    mean_row.insert(0, "tomogram", "Mean")
    table = pd.concat(
        [df[["tomogram"] + metrics], mean_row], ignore_index=True
    )
    table[metrics] = table[metrics].round(3)

    col_labels = ["Tomogram", "Precision", "Recall", "Dice"]
    cell_text = table.values.tolist()
    n_rows, n_cols = len(cell_text), len(col_labels)

    fig, ax = plt.subplots(figsize=(n_cols * 1.6, (n_rows + 1) * 0.5))
    ax.axis("off")
    t = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    t.auto_set_font_size(False)
    t.set_fontsize(11)
    t.scale(1, 1.4)

    for col in range(n_cols):
        cell = t[0, col]
        cell.set_facecolor("white")
        cell.set_text_props(fontweight="bold")
        cell.set_edgecolor("white")
        cell.visible_edges = "TB"

    for row in range(1, n_rows + 1):
        is_mean = row == n_rows
        for col in range(n_cols):
            cell = t[row, col]
            cell.set_facecolor("white")
            cell.set_edgecolor("white")
            cell.visible_edges = "B" if is_mean else "open"
            if is_mean:
                cell.set_text_props(fontweight="bold")
                cell.set_edgecolor("black")

    fig.suptitle(
        model_name, fontsize=12, fontweight="bold", x=0.1, ha="left", y=0.95
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved PNG: {out_path}")


def main():
    args = parse_args()

    data_root = Path(
        "/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data"
    )
    checkpoint = args.checkpoint
    model_name = Path(checkpoint).stem
    data_dir = data_root / "experimental" / "deepict" / "h5"
    out_dir = data_root / "predictions" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    data_paths = sorted(data_dir.glob("*.h5"))
    rows = []

    for p in data_paths:
        with h5py.File(p, "r") as f:
            raw = f["raw"][:]
            gt = f["labels"]["actin"][:]
            mask = f["sample_mask"][:]

        seg, pred = segment_actin(
            raw, checkpoint, verbose=True, return_predictions=True
        )

        assert seg.shape == mask.shape, (
            f"Shape mismatch: {seg.shape} != {mask.shape}"
        )
        seg = seg * mask.astype(seg.dtype)

        m = compute_metrics(seg, gt)
        print(f"{p.stem}: {m}")
        rows.append({"tomogram": p.stem, **m})

        with h5py.File(p, "a") as f:
            if f"segmentations/{model_name}" not in f:
                f.create_dataset(
                    f"segmentations/{model_name}", data=seg, compression="gzip"
                )
            if f"predictions/{model_name}" not in f:
                f.create_dataset(
                    f"predictions/{model_name}", data=pred, compression="gzip"
                )

    print("\n--- Summary ---")
    print(f"mean precision: {np.mean([r['precision'] for r in rows]):.4f}")
    print(f"mean recall:    {np.mean([r['recall'] for r in rows]):.4f}")
    print(f"mean dice:      {np.mean([r['dice'] for r in rows]):.4f}")

    csv_path = out_dir / f"{model_name}_metrics.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["tomogram", "precision", "recall", "dice"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV: {csv_path}")

    df = pd.read_csv(csv_path, dtype={"tomogram": str})
    png_path = out_dir / f"{model_name}_metrics.png"
    save_metrics_png(df, model_name, png_path)


if __name__ == "__main__":
    main()
