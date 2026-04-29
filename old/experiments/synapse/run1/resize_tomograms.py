from glob import glob
from pathlib import Path

import h5py
import numpy as np
from synapse_net.file_utils import read_mrc
import torch_em
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
        raw = np.flip(raw, axis = 1)
        
        out_path = out_dir / f"{Path(p).stem}.mrc"

        with mrcfile.new(out_path, overwrite=True) as mrc:
            mrc.set_data(raw.astype("float32")) 
            mrc.voxel_size = (target_vsize, target_vsize, target_vsize)

        out_paths.append(out_path)
        print(f"Rescaled input tomogram {Path(p).stem} to {target_vsize}A.")

    return out_paths

def main():
    PARENT_DIR = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/synapse")
    IN_DIR = PARENT_DIR / "raw"
    OUT_DIR = PARENT_DIR / "rescaled_10A"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    data_paths = [str(p) for p in IN_DIR.glob("*.mrc")]

    target_vsize = 10
    resize_tomos(data_paths, OUT_DIR, target_vsize)

if __name__ == "__main__":
    main()
