import configargparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
from pathlib import Path
from synapse_net.inference.actin import segment_actin


def parse_args():
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.TomlConfigParser(["run_info"])
    )
    parser.add_argument("--config", is_config_file_arg=True, help="Path to TOML config file.")

    # run_info
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--synthetic_dataset", type=str, required=True)
    parser.add_argument("--real_dataset", type=str, required=True)
    parser.add_argument("--run", type=int, required=True)

    # optional overrides
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override checkpoint path (default: derived from run).")

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
    return {"precision": float(precision), "recall": float(recall), "dice": float(dice)}


def save_metrics_png(df, model_name, out_path):
    metrics = ["precision", "recall", "dice"]

    mean_row = df[metrics].mean().to_frame().T
    mean_row.insert(0, "tomogram", "mean")
    table = pd.concat([df[["tomogram"] + metrics], mean_row], ignore_index=True)
    table[metrics] = table[metrics].round(3)

    col_labels = ["Tomogram", "Precision", "Recall", "Dice"]
    cell_text = table.values.tolist()
    n_rows, n_cols = len(cell_text), len(col_labels)

    fig, ax = plt.subplots(figsize=(n_cols * 1.6, (n_rows + 1) * 0.5))
    ax.axis("off")
    t = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
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

    fig.suptitle(model_name, fontsize=12, fontweight="bold", x=0.1, ha="left", y=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved PNG: {out_path}")


def main():
    args = parse_args()

    data_root = Path(args.data_root)

    checkpoint = args.checkpoint or str(
        data_root / "training" / "out" / "deepict" / f"run{args.run}"
        / "checkpoints" / f"actin-deepict-run{args.run}"
    )
    model_name = Path(checkpoint).stem

    test_dir = data_root / "training" / args.synthetic_dataset / "test"
    data_paths = sorted(test_dir.glob("*.h5"))

    if not data_paths:
        raise FileNotFoundError(f"No H5 files found in {test_dir}")

    print(f"Found {len(data_paths)} test files in {test_dir}")
    print(f"Checkpoint: {checkpoint}")

    out_dir = data_root / "predictions" / args.real_dataset
    rows = []

    for p in data_paths:
        with h5py.File(p, "r") as f:
            raw = f["raw"][:]
            gt = f["labels"]["actin"][:]

        seg, pred = segment_actin(raw, checkpoint, verbose=True, return_predictions=True)

        m = compute_metrics(seg, gt)
        print(f"{p.stem}: {m}")
        rows.append({"tomogram": p.stem, **m})

        with h5py.File(p, "a") as f:
            if f"segmentations/{model_name}" not in f:
                f.create_dataset(f"segmentations/{model_name}", data=seg, compression="gzip")
            if f"predictions/{model_name}" not in f:
                f.create_dataset(f"predictions/{model_name}", data=pred, compression="gzip")

    print("\n--- Summary ---")
    print(f"mean precision: {np.mean([r['precision'] for r in rows]):.4f}")
    print(f"mean recall:    {np.mean([r['recall'] for r in rows]):.4f}")
    print(f"mean dice:      {np.mean([r['dice'] for r in rows]):.4f}")

    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"{model_name}_synthetic.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["tomogram", "precision", "recall", "dice"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV: {csv_path}")

    df = pd.read_csv(csv_path, dtype={"tomogram": str})
    png_dir = out_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    png_path = png_dir / f"{model_name}_synthetic.png"
    save_metrics_png(df, model_name, png_path)


if __name__ == "__main__":
    main()
