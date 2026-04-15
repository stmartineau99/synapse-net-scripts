import os
from glob import glob
from pathlib import Path
import h5py
import numpy as np
import torch_em
from synapse_net.inference.actin import segment_actin
from synapse_net.file_utils import read_mrc
import mrcfile
from sklearn.metrics import precision_score, recall_score, f1_score
import csv

def predict_actin(data_paths, model_path):
    model_name = Path(model_path).stem

    for p in data_paths:
        raw = read_mrc(p)[0]

        seg, pred = segment_actin(raw, model_path, verbose=True, return_predictions=True)
        
        with h5py.File(p, "a") as f:
            if f"segmentations/{model_name}" not in f:
                f.create_dataset(f"segmentations/{model_name}", data=seg, compression="gzip")
            if f"predictions/{model_name}" not in f:
                f.create_dataset(f"predictions/{model_name}", data=pred, compression="gzip")

            print(f"Predictions written to {p}.")



def main():
    PARENT_DIR = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/public/synapse")
    DATA_DIR = PARENT_DIR / "h5"
    MASK_DIR = PARENT_DIR / "sample_masks"

    RESULTS_DIR = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/predictions")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # build dict for source_vsice
    source_vsizes = {}
    RAW_DIR = PARENT_DIR / "raw"
    raw_paths = sorted(RAW_DIR.glob("*.mrc"))

    for p in raw_paths:
        source_vsizes[p.stem] = read_voxel_size(p)

    out_dir = DATA_DIR / "rescale_10A"
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
        #os.symlink(source_path, dest_path)
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


