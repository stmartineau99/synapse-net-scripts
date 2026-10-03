import configargparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import h5py
import torch
import torch_em
from pathlib import Path
from synapse_net.inference.actin import segment_actin


def parse_args():
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.TomlConfigParser(
            ["run_info"]
        )
    )
    parser.add_argument(
        "--config", is_config_file_arg=True, help="Path to TOML config file."
    )
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--synthetic_dataset", type=str, required=False)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--step", type=int, default=1)
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


def plot_metrics(df, model_name, out_path):
    best = df.loc[df["recall"].idxmax()]
    fig, ax = plt.subplots(figsize=(10, 5))
    for metric in ["precision", "recall", "dice"]:
        ax.plot(df["iteration"], df[metric], marker="o", label=metric)
    ax.plot(best["iteration"], best["recall"], marker="o", color="red")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Score")
    ax.set_title(model_name, fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved plot: {out_path}")


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    model_name = f"actin-{args.dataset}-run{args.run}"

    checkpoint_dir = (
        data_root / "training" / "out" / args.dataset / f"run{args.run}"
        / "checkpoints" / model_name
    )
    data_dir = data_root / "predictions" / args.dataset
    out_dir = data_dir

    epoch_pts = sorted(
        checkpoint_dir.glob("epoch-*.pt"),
        key=lambda p: int(p.stem.split("-")[1]),
    )[::args.step]
    if not epoch_pts:
        print(f"No epoch checkpoints found in {checkpoint_dir}")
        return

    tomo_ids = {"00004", "00012"}
    data_paths = sorted(p for p in data_dir.glob("*.h5") if p.stem in tomo_ids)
    if not data_paths:
        print(f"No h5 files found in {data_dir}")
        return

    rows = []
    for pt in epoch_pts:
        ckpt_dict = torch.load(str(pt), weights_only=False)
        epoch = ckpt_dict["epoch"] + 1
        iteration = ckpt_dict["iteration"]
        print(f"\n--- Checkpoint: {pt.name} (iter {iteration}) ---")

        model = torch_em.util.load_model(
            checkpoint=str(checkpoint_dir), name=pt.stem
        )

        for p in data_paths:
            with h5py.File(p, "r") as f:
                raw = f["raw"][:]
                gt = f["labels"]["actin"][:]
                mask = f["sample_mask"][:]

            seg = segment_actin(raw, model=model, verbose=True)
            seg = seg * mask.astype(seg.dtype)
            m = compute_metrics(seg, gt)
            print(f"{p.stem}: {m}")
            rows.append({"epoch": epoch, "iteration": iteration, "tomogram": p.stem, **m})

    csv_dir = out_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"{model_name}_checkpoints.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "iteration", "tomogram", "precision", "recall", "dice"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    df_mean = df.groupby(["epoch", "iteration"])[["precision", "recall", "dice"]].mean().reset_index()

    print("\n--- Summary ---")
    for _, r in df_mean.iterrows():
        print(f"iter {int(r['iteration']):>6d}  epoch {int(r['epoch'])}: precision={r['precision']:.4f}  recall={r['recall']:.4f}  dice={r['dice']:.4f}")
    png_dir = out_dir / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    png_path = png_dir / f"{model_name}_checkpoints.png"
    plot_metrics(df_mean, model_name, png_path)

if __name__ == "__main__":
    main()
