import h5py
import numpy as np
import torch_em
from pathlib import Path
from synapse_net.file_utils import read_voxel_size
import mrcfile

def process_sample(data_path, mask_path, out_path, target_vsize, source_vsize):

    with h5py.File(data_path, "r") as f:
        raw = f["raw"][:]
        labels = f["labels/actin"][:]

    with mrcfile.open(mask_path, permissive=True) as mrc:
        mask = np.asarray(mrc.data[:])

    scale = tuple(
        source_vsize[ax] / (target_vsize / 10) for ax in ("z", "y", "x")
    )
    raw = torch_em.transform.generic.Rescale(scale)(raw)
    labels = torch_em.transform.generic.Rescale(scale, is_label=True)(labels)
    mask = torch_em.transform.generic.Rescale(scale, is_label=True)(mask)

    labels = labels * mask.astype(labels.dtype)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("labels/actin", data=labels, compression="gzip")
        f.create_dataset("sample_mask", data=mask, compression="gzip")

    print(f"Rescaled {data_path.stem} to {target_vsize}Å.")


def main():
    DATA_ROOT = Path(
        "/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/experimental/deepict"
    )
    DATA_DIR = DATA_ROOT / "h5"
    MASK_DIR = DATA_ROOT / "sample_masks"
    RESULTS_DIR = Path(
        "/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/predictions/deepict"
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    target_vsize = 10.0  # Angstrom

    # build dict for source_vsice
    source_vsizes = {}
    RAW_DIR = DATA_ROOT / "raw"
    raw_paths = sorted(RAW_DIR.glob("*.mrc"))

    for p in raw_paths:
        source_vsizes[p.stem] = read_voxel_size(p)

    data_paths = sorted(DATA_DIR.glob("*.h5"))
    for data_path in data_paths:
        mask_path = next(MASK_DIR.glob(f"{data_path.stem}.mrc"), None)
        if mask_path is None:
            print(f"Skipping {data_path.stem}: no matching mask found.")
            continue

        out_path = RESULTS_DIR / f"{data_path.stem}.h5"
        if out_path.exists():
            print(f"Skipping {out_path.name}, already exists.")
        else:
            process_sample(data_path, mask_path, out_path, target_vsize, source_vsizes[data_path.stem])


if __name__ == "__main__":
    main()
