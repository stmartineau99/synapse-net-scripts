from glob import glob
from pathlib import Path
import h5py
import numpy as np
import torch_em
from synapse_net.inference.actin import segment_actin
from synapse_net.file_utils import read_mrc
import csv

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

def predict_actin(data_paths, model_path, metrics=False, csv_path=None):
    model_name = Path(model_path).stem

    rows = []

    for p in data_paths:
        with h5py.File(p, "r") as f: 
            raw = f["raw"][:]
            gt = f["labels"]["actin"][:]
            mask = f["sample_mask"][:]

        seg, pred = segment_actin(raw, model_path, verbose=True, return_predictions=True)
        
        assert seg.shape == mask.shape, f"Shape mismatch: {seg.shape} != {mask.shape}"
        seg = seg * mask.astype(seg.dtype)

        if metrics:
            m = compute_metrics(seg, gt)
            print(m)
            rows.append({"tomogram": p.stem, **m})

        with h5py.File(p, "a") as f:
            if f"segmentations/{model_name}" not in f:
                f.create_dataset(f"segmentations/{model_name}", data=seg, compression="gzip")
            if f"predictions/{model_name}" not in f:
                f.create_dataset(f"predictions/{model_name}", data=pred, compression="gzip")

            print(f"Predictions written to {p}.")
    
    if metrics and rows:
        print("---Summary---")
        print(f"mean precision: {np.mean([r['precision'] for r in rows]):.4f}")
        print(f"mean recall:    {np.mean([r['recall'] for r in rows]):.4f}")
        print(f"mean dice:      {np.mean([r['dice'] for r in rows]):.4f}")

        if csv_path is not None:
            with open(csv_path, mode="w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["tomogram", "precision", "recall", "dice"])
                writer.writeheader()
                writer.writerows(rows)
            print(f"Metrics written to {csv_path}.")


def main():
    RESULTS_DIR = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/predictions")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = "/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/training/out/deepict/run8/checkpoints/actin-deepict-adapted-run8"
    model_name = Path(model_path).stem

    csv_dir = RESULTS_DIR / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"results_{model_name}.csv"
    data_paths = sorted(RESULTS_DIR.glob("*.h5"))
    predict_actin(data_paths, model_path, metrics=True, csv_path=csv_path)

if __name__ == "__main__":
    main()