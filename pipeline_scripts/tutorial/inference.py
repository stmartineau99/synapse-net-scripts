import argparse
import csv
from pathlib import Path

import numpy as np
import h5py

from elf.io import open_file

from synapse_net.inference.actin import segment_actin as segment


def parse_args():
    parser = argparse.ArgumentParser(description="Segment filaments in cryo-ET tomograms.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with mrc tomograms to segment.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where h5 outputs and metrics will be saved.")  # noqa
    parser.add_argument("--label_dir", type=str, default=None, help="Directory with ground truth mrc labels. Used to compute metrics, if given.")  # noqa
    parser.add_argument("--mask_dir", type=str, default=None, help="Directory with mrc masks. Sample masks are applied to restrict the prediction.")  # noqa
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binarizing the foreground prediction.")  # noqa

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


def load_volume(path):
    with open_file(str(path), "r") as f:
        return f["data"][:]


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = Path(args.checkpoint).name

    input_paths = sorted(Path(args.input_dir).glob("*.mrc"))
    label_paths = sorted(Path(args.label_dir).glob("*.mrc")) if args.label_dir else None
    mask_paths = sorted(Path(args.mask_dir).glob("*.mrc")) if args.mask_dir else None

    rows = []
    for i, input_path in enumerate(input_paths):
        print(f"Predicting: {input_path.stem}")
        raw = load_volume(input_path)

        _, pred = segment(
            raw, model_path=args.checkpoint, foreground_threshold=args.threshold,
            verbose=False, return_predictions=True
        )

        if mask_paths is not None:
            mask = load_volume(mask_paths[i])
            pred = pred * mask.astype(pred.dtype)
        seg = (pred > args.threshold).astype(bool)

        gt = load_volume(label_paths[i]) if label_paths is not None else None
        if gt is not None:
            m = compute_metrics(seg, gt)
            print(f"{input_path.stem}: {m}")
            rows.append({"tomogram": input_path.stem, **m})

        out_path = output_dir / f"{input_path.stem}.h5"
        with h5py.File(out_path, "a") as f:
            if "raw" not in f:
                f.create_dataset("raw", data=raw.astype("float32"), compression="gzip")
            if gt is not None and "labels/gt" not in f:
                f.create_dataset("labels/gt", data=gt.astype("uint8"), compression="gzip")
            pred_key, seg_key = f"predictions/{model_name}", f"segmentations/{model_name}"
            if pred_key not in f:
                f.create_dataset(pred_key, data=pred, compression="gzip")
            if seg_key not in f:
                f.create_dataset(seg_key, data=seg, compression="gzip")
        print(f"Saved: {out_path}")

    if rows:
        print("\n--- Summary ---")
        print(f"mean precision: {np.mean([r['precision'] for r in rows]):.4f}")
        print(f"mean recall:    {np.mean([r['recall'] for r in rows]):.4f}")
        print(f"mean dice:      {np.mean([r['dice'] for r in rows]):.4f}")

        csv_path = output_dir / "metrics.csv"
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["tomogram", "precision", "recall", "dice"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()