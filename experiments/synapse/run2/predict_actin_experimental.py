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

def predict_actin(data_paths, out_dir, model_path):
    model_name = Path(model_path).stem

    for p in data_paths:
        raw = read_mrc(p)[0]

        seg, pred = segment_actin(raw, model_path, verbose=True, return_predictions=True)
        
        out_path = out_dir / f"{Path(p).stem}.h5"

        with h5py.File(out_path, "a") as f:
            f.create_dataset(f"raw", data=raw, compression="gzip")
            if f"segmentations/{model_name}" not in f:
                f.create_dataset(f"segmentations/{model_name}", data=seg, compression="gzip")
            if f"predictions/{model_name}" not in f:
                f.create_dataset(f"predictions/{model_name}", data=pred, compression="gzip")

            print(f"Predictions written to {p}.")

def main():
    DATA_DIR = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/public/synapse/rescaled_10A")

    RESULTS_DIR = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/predictions")

    model_path = "/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/training/out/synapse/run2/checkpoints/actin-deepict-run2"
    model_name = Path(model_path).stem

    data_paths = [DATA_DIR / "TS_001.mrc",
                  DATA_DIR / "TS_003.mrc",
                  DATA_DIR / "TS_004.mrc"]
    predict_actin(data_paths, model_path)

if __name__ == "__main__"
    main()


