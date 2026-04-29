import os
from glob import glob
from pathlib import Path
import h5py
import numpy as np
import torch_em
from synapse_net.inference.actin import segment_actin
from synapse_net.file_utils import read_voxel_size, read_mrc
import mrcfile
import csv

def resize_tomos(data_path, out_dir, target_vsize, source_vsize, mask_path):

    with h5py.File(data_path, "r") as f:
        raw = f["raw"][:]
        gt = f["labels/actin"][:]

    with mrcfile.open(mask_path, permissive=True) as mrc:
        mask = np.asarray(mrc.data[:])

    scale = (
        source_vsize["z"] / (target_vsize / 10),
        source_vsize["y"] / (target_vsize / 10),
        source_vsize["x"] / (target_vsize / 10),
    )

    rescale_raw = torch_em.transform.generic.Rescale(scale)
    rescale_label = torch_em.transform.generic.Rescale(scale, is_label=True)
    raw = rescale_raw(raw)
    gt = rescale_label(gt)
    mask = rescale_label(mask)

    out_path = out_dir / f"{data_path.stem}.h5"
    with h5py.File(out_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("labels/actin", data=gt, compression="gzip")
        f.create_dataset("sample_mask", data=mask, compression="gzip")

    print(f"Rescaled {data_path.stem} to {target_vsize}Å.")

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
    PARENT_DIR = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/public/deepict")
    DATA_DIR = PARENT_DIR / "h5/orig"
    MASK_DIR = PARENT_DIR / "sample_masks"

    RESULTS_DIR = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/predictions")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # build dict for source_vsice
    source_vsizes = {}
    RAW_DIR = PARENT_DIR / "raw"
    raw_paths = sorted(RAW_DIR.glob("*.mrc"))

    for p in raw_paths:
        source_vsizes[p.stem] = read_voxel_size(p)

    out_dir = PARENT_DIR / "h5"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    data_paths = sorted(DATA_DIR.glob("*.h5"))
    for data_path in data_paths:
        mask_path = next(MASK_DIR.glob(f"{data_path.stem}.mrc"), None)
        if mask_path is None:
            print(f"Skipping {data_path.stem}: no matching mask found.")
            continue
        source_vsize = source_vsizes[data_path.stem]
        #resize_tomos(data_path, out_dir, target_vsize=10, 
        #             source_vsize=source_vsize, mask_path=mask_path)
        
    source_paths = sorted(out_dir.glob("*.h5"))
    for source_path in source_paths:
        dest_path = RESULTS_DIR / f"{source_path.stem}.h5"
        os.symlink(source_path, dest_path)
        if os.path.islink(dest_path):
            print(f"{dest_path} is a symlink pointing to {os.readlink(dest_path)}.")
        else:
            print("Symlink creation failed.")

    model_path = "/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/training/out/deepict/run6/checkpoints/actin-deepict-run6"
    model_name = Path(model_path).stem

    csv_dir = RESULTS_DIR / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / f"results_{model_name}.csv"
    data_paths = sorted(RESULTS_DIR.glob("*.h5"))
    predict_actin(data_paths, model_path, metrics=True, csv_path=csv_path)

if __name__ == "__main__":
    main()


