from glob import glob
from pathlib import Path

import h5py
import numpy as np
import torch_em
from synapse_net.inference.actin import segment_actin
from synapse_net.file_utils import read_mrc
import mrcfile

def resize_tomos(data_paths, out_dir, target_vsize):
    out_paths = []
    for p in data_paths:
        raw, source_vsize = read_mrc(p)

        scale = (
                source_vsize["z"] / (target_vsize / 10),
                source_vsize["y"] / (target_vsize / 10),
                source_vsize["x"] / (target_vsize / 10),
                )

        rescale_raw = torch_em.transform.generic.Rescale(scale)                
        raw = rescale_raw(raw)

        out_path = out_dir / f"{Path(p).stem}.mrc"

        with mrcfile.new(out_path, overwrite=True) as mrc:
            mrc.set_data(raw.astype("float32")) 
            mrc.voxel_size = (target_vsize, target_vsize, target_vsize)

        out_paths.append(out_path)
        print(f"Rescaled input tomogram {Path(p).stem} to {target_vsize}.")
        
    return out_paths

def predict_actin(data_paths, out_dir, model_path):
    model_name = Path(model_path).stem

    for p in data_paths:
        raw = read_mrc(p)[0][:]

        seg, pred = segment_actin(raw, model_path, verbose=True, return_predictions=True)

        out_path = out_dir / f"{Path(p).stem}.h5"

        with h5py.File(out_path, "a") as f:
            if "raw" not in f:
                f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset(f"segmentations/{model_name}", data=seg, compression="gzip")
            f.create_dataset(f"predictions/{model_name}", data=pred, compression="gzip")


def main():
    ### predictions on synthetic data ### 
    #TODO run predictions on h5
    DATA_DIR = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/public/training/test")
    results_dir =  Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/public/predictions/deepict_dataset_3")
    results_dir.mkdir(parents=True, exist_ok=True)
    model_path = "/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/training/out/deepict/run6/checkpoints/actin-deepict-run6"

    data_paths = [str(p) for p in DATA_DIR.glob("*.mrc")]
    
    ### predictions on deepict data ###
    PARENT_DIR = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/public/deepict")
    DATA_DIR = PARENT_DIR / "raw"

    rescale_dir = DATA_DIR / "rescale_10A"
    rescale_dir.mkdir(parents=True, exist_ok=True)
    
    rescaled_paths = [str(p) for p in rescale_dir.glob("00011.mrc")]
    predict_actin(rescaled_paths, results_dir, model_path)

if __name__ == "__main__":
    main()
