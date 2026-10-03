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
        config_file_parser_class=configargparse.TomlConfigParser(["run_info", "predict_actin"]),
        ignore_unknown_config_file_keys=True,
    )
    parser.add_argument("--config", is_config_file_arg=True, help="Path to TOML config file.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--run", type=int, required=True)

    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--use_sample_mask", action="store_true", help="Mask the prediction with sample_mask.")
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--data_dir", type=str, default=None)
    group.add_argument("--data_paths", type=str, nargs="+")

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
    mean_row.insert(0, "tomogram", "mean")
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
    DATA_ROOT = Path(args.data_root)

    checkpoint = args.checkpoint or str(
        DATA_ROOT / "training" / "out" / args.dataset / f"run{args.run}"
        / "checkpoints" / f"actin-{args.dataset}-run{args.run}"
    )
    model_name = Path(checkpoint).stem
    threshold = args.threshold

    if args.data_dir:
        data_paths = sorted(Path(args.data_dir).glob("*.h5"))
    elif args.data_paths:
        data_paths = [Path(p) for p in args.data_paths]
    else:
        data_paths = sorted((DATA_ROOT / "predictions" / args.dataset).glob("*.h5"))

    if not data_paths:
        raise FileNotFoundError("No h5 files found.")

    rows = []
    print(f"\nRunning predictions using model {model_name}.\n")

    for input_path in data_paths:
        print(f"Predicting: {input_path.stem}\n")

        with h5py.File(input_path, "r") as f:
            raw = f["raw"][:]
            has_labels = "labels/actin" in f
            gt = f["labels/actin"][:] if has_labels else None
            mask = f["sample_mask"][:] if args.use_sample_mask and "sample_mask" in f else None
            pred_key = f"predictions/{model_name}"
            pred = f[pred_key][:] if pred_key in f else None

        if pred is not None:
            print(f"Using stored predictions (threshold={threshold})\n")
        else:
            _, pred = segment_actin(
                raw, checkpoint, foreground_threshold=threshold, verbose=True, return_predictions=True
            )

        if mask is not None:
            pred = pred * mask.astype(pred.dtype)
        seg = (pred > threshold).astype(bool)

        if has_labels:
            m = compute_metrics(seg, gt)
            print(f"{input_path.stem}: {m}")
            rows.append({"tomogram": input_path.stem, **m})
        else:
            print(f"{input_path.stem}: no GT labels, skipping metrics")

        with h5py.File(input_path, "a") as f:
            if threshold != 0.5:
                seg_key = f"segmentations/{model_name}_t{threshold}"
            else:
                seg_key = f"segmentations/{model_name}"

            if seg_key not in f:
                f.create_dataset(seg_key, data=seg, compression="gzip")
            if pred_key not in f:
                f.create_dataset(pred_key, data=pred, compression="gzip")

    if rows:
        print("\n--- Summary ---")
        print(f"mean precision: {np.mean([r['precision'] for r in rows]):.4f}")
        print(f"mean recall:    {np.mean([r['recall'] for r in rows]):.4f}")
        print(f"mean dice:      {np.mean([r['dice'] for r in rows]):.4f}")

        csv_dir = DATA_ROOT / f"predictions/{args.dataset}/csv"
        csv_dir.mkdir(parents=True, exist_ok=True)
        csv_path = csv_dir / f"{model_name}.csv"
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["tomogram", "precision", "recall", "dice"]
            )
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved CSV: {csv_path}")

        df = pd.read_csv(csv_path, dtype={"tomogram": str})
        png_dir = DATA_ROOT / f"predictions/{args.dataset}/png"
        png_dir.mkdir(parents=True, exist_ok=True)
        png_path = png_dir / f"{model_name}.png"
        save_metrics_png(df, model_name, png_path)


if __name__ == "__main__":
    main()
