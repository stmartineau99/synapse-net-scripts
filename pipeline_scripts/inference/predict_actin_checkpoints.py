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

RESULTS_DIR = Path(
    "/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/synapse-net-scripts/experiments/checkpoint_analysis"
)

def parse_args():
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.TomlConfigParser(["run_info"])
    )
    parser.add_argument("--config", is_config_file_arg=True, help="Path to TOML config file.")
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--synthetic_dataset", type=str, required=True)
    parser.add_argument("--real_dataset", type=str, required=True)
    parser.add_argument("--run", type=int, required=True)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--real_data_dir", type=str, default=None)

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

def evaluate_checkpoint(model, input_paths, domain, epoch, iteration):
    rows = []
    for input_path in input_paths: 
        with h5py.File(input_path, "r") as f:
            raw = f["raw"][:]
            has_labels = "labels/actin" in f
            gt = f["labels/actin"][:] if has_labels else None
            if not has_labels:
                print(f"{input_path.stem}: no GT labels, skipping metrics")
                continue
        seg = segment_actin(raw, model=model, verbose=True)

        m = compute_metrics(seg, gt)
        rows.append({
            "epoch": epoch,
            "iteration": iteration,
            "domain": domain,
            "tomogram": input_path.stem,
            **m,
            })
        print(f"[{domain}] {input_path.stem}: {m}")
    return rows

def plot_metrics(df, model_name, output_path):
    metrics = ["precision", "recall", "dice"]
    colors = {"real": "steelblue", "synthetic": "coral"}
    domains = ["synthetic", "real"]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 12), sharex=True)
    for ax, metric in zip(axes, metrics):
        for domain in domains:
            sub = df[df["domain"] == domain].sort_values("iteration")
            ax.plot(sub["iteration"], sub[metric], marker="o",
                    color=colors.get(domain), label=domain)
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()
    axes[0].set_title(model_name, fontsize=12, fontweight="bold")
    axes[-1].set_xlabel("Iteration")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    model_name = f"actin-{args.real_dataset}-run{args.run}"

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else (
        data_root / "training" / "out" / args.real_dataset / f"run{args.run}"
        / "checkpoints" / model_name
    )
    synthetic_data_dir = data_root / "training" / args.synthetic_dataset / "test"
    synthetic_paths = sorted(synthetic_data_dir.glob("*.h5"))

    if not synthetic_paths:
        raise FileNotFoundError(f"No synthetic h5 files found in {synthetic_data_dir}.")

    if args.real_data_dir:
        real_data_dir = Path(args.real_data_dir)
        real_paths = sorted(real_data_dir.glob("*.h5"))
    else:
        real_data_dir = data_root / "predictions" / args.real_dataset
        real_paths = sorted(real_data_dir.glob("*.h5"))

        if not real_paths:
            raise FileNotFoundError(f"No experimental h5 files found in {real_data_dir}.")

    epoch_pts = sorted(
        checkpoint_dir.glob("epoch-*.pt"), key=lambda p: int(p.stem.split("-")[1]),
    )[::args.step]
    if not epoch_pts:
        print(f"No epoch checkpoints found in {checkpoint_dir}")
        return

    rows = []
    epoch_to_ckpt = {}
    print(f"\nEvaluating {len(epoch_pts)} checkpoints for model {model_name}.\n")

    for pt in epoch_pts:
        ckpt_dict = torch.load(str(pt), weights_only=False)
        epoch = ckpt_dict["epoch"] + 1
        iteration = ckpt_dict["iteration"]
        epoch_to_ckpt[epoch] = pt.stem
        print(f"\n--- Checkpoint: {pt.name} (iter {iteration}) ---")

        model = torch_em.util.load_model(checkpoint=str(checkpoint_dir), name=pt.stem)

        for domain, input_paths in zip(["synthetic", "real"], (synthetic_paths, real_paths)):
            rows.extend(evaluate_checkpoint(model, input_paths, domain, epoch, iteration))

    out_dir = RESULTS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{model_name}_checkpoints.csv"
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["epoch", "iteration", "domain", "tomogram", "precision", "recall", "dice"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV: {csv_path}")

    df = pd.read_csv(csv_path, dtype={"tomogram": str})
    df_mean = df.groupby(["epoch", "iteration", "domain"])[["precision", "recall", "dice"]].mean().reset_index()

    summary_path = out_dir / f"{model_name}_summary.csv"
    df_mean.to_csv(summary_path, index=False)
    print(f"Saved summary CSV: {summary_path}")

    png_path = out_dir / f"{model_name}_checkpoints.png"
    plot_metrics(df_mean, model_name, png_path)

    sub = df_mean[df_mean["domain"] == "real"]
    for metric in ["precision", "recall", "dice"]:
        best = sub.loc[sub[metric].idxmax()]
        best_epoch = int(best["epoch"])
        print(f"\nBest {metric}: epoch {best_epoch} (iter {int(best['iteration'])}), "
              f"precision={best["precision"]:.4f}  recall={best["recall"]:.4f}  dice={best["dice"]:.4f}")

if __name__ == "__main__":
    main()
