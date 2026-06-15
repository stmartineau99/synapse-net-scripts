import argparse
import numpy as np
import matplotlib.pyplot as plt
import h5py
from pathlib import Path
from sklearn.metrics import precision_recall_curve, auc

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--gt_key", type=str, default="labels/actin")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    png_dir = Path(args.results_dir)
    png_dir.mkdir(parents=True, exist_ok=True)
    gt_key = args.gt_key
    pred_key = f"predictions/{args.model_name}"

    all_pred, all_gt = [], []
    for data_path in sorted(Path(args.data_dir).glob("*.h5")):
        with h5py.File(data_path, "r") as f:
            if gt_key not in f:
                print(f"GT labels not found in {data_path.stem}.")
                continue
            if pred_key not in f: 
                print(f"{pred_key} predictions not found in {data_path.stem}.")
                continue
            gt = f[gt_key][:].ravel().astype(bool)
            pred = f[pred_key][:].ravel().astype(np.float32)
            all_gt.append(gt)
            all_pred.append(pred)
        
    precision, recall, _ = precision_recall_curve(
        np.concatenate(all_gt), np.concatenate(all_pred)
    )
    pr_auc = auc(recall, precision)
    print(f"AUC-PR: {pr_auc:.4f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#4878d0", linewidth=2,
            label=f"{args.model_name}  AUC={pr_auc:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    fig.tight_layout()

    out_path = png_dir / f"pr_curve_{args.model_name}.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
