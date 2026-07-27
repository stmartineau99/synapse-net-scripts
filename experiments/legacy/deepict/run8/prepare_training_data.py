from glob import glob
from pathlib import Path

import h5py
import torch_em
import mrcfile
from synapse_net.file_utils import read_mrc

def resize_masks(data_paths, out_dir, target_vsize):
    out_paths = []
    for p in data_paths:
        mask, source_vsize = read_mrc(p)

        scale = (
                source_vsize["z"] / (target_vsize / 10),
                source_vsize["y"] / (target_vsize / 10),
                source_vsize["x"] / (target_vsize / 10),
                )

        rescale_raw = torch_em.transform.generic.Rescale(scale, is_label=True)               
        mask = rescale_raw(mask)

        out_path = out_dir / f"{Path(p).stem}.mrc"

        with mrcfile.new(out_path, overwrite=True) as mrc:
            mrc.set_data(mask.astype("float32")) 
            mrc.voxel_size = (target_vsize, target_vsize, target_vsize)

        out_paths.append(out_path)
        print(f"Rescaled input tomogram {Path(p).stem} to {target_vsize}.")
        
    return out_paths

def main():
    # rescale sample masks
    PARENT_DIR = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/public/deepict")
    DATA_DIR = PARENT_DIR / "raw/rescale_10A"
    data_paths = [str(p) for p in DATA_DIR.glob("*.mrc")]

    MASK_DIR = PARENT_DIR / "sample_masks"
    rescale_dir = MASK_DIR / "rescale_10A"
    rescale_dir.mkdir(parents=True, exist_ok=True)
    
    mask_paths = [str(p) for p in MASK_DIR.glob("*.mrc")]
    resize_masks(mask_paths, rescale_dir, target_vsize=10)
    mask_paths = [str(p) for p in rescale_dir.glob("*.mrc")]

    out_dir = PARENT_DIR / "h5"
    for data_path, mask_path in zip(data_paths, mask_paths):
        raw = read_mrc(data_path)[0]
        mask = read_mrc(mask_path)[0]

        out_path = out_dir / f"{Path(data_path).stem}.h5"
        with h5py.File(out_path, "w") as f:
            f.create_dataset("raw", data=raw, compression="gzip")
            f.create_dataset("sample_mask", data=mask, compression="gzip")
        print(f"Tomogram {Path(data_path).stem} saved to {out_path}.")

if __name__ == "__main__":
    main()
