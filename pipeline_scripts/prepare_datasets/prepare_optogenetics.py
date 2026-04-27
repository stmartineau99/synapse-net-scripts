import h5py
import mrcfile
import numpy as np
import torch_em
from pathlib import Path
from synapse_net.file_utils import read_mrc

CROP_SHAPE = (122, 971, 971)  # (z, y, x) center crop

def center_crop(volume, shape):
    crops = tuple(
        slice((s - c) // 2, (s - c) // 2 + c)
        for s, c in zip(volume.shape, shape)
    )
    return volume[crops]

def process_sample(raw_path, label_path, mask_path, out_path, target_vsize):
    raw, source_vsize = read_mrc(raw_path)
    labels, _ = read_mrc(label_path)
    mask, _ = read_mrc(mask_path)

    scale = tuple(
        source_vsize[ax] / (target_vsize / 10) for ax in ("z", "y", "x")
    )
    raw = torch_em.transform.generic.Rescale(scale)(raw)
    labels = torch_em.transform.generic.Rescale(scale, is_label=True)(labels)
    mask = torch_em.transform.generic.Rescale(scale, is_label=True)(mask)

    raw = center_crop(raw, CROP_SHAPE)
    labels = center_crop(labels, CROP_SHAPE)
    mask = center_crop(mask, CROP_SHAPE)

    rescaled_raw_dir = raw_path.parent / "rescaled_10A"
    rescaled_raw_dir.mkdir(exist_ok=True)
    with mrcfile.new(rescaled_raw_dir / raw_path.name, overwrite=True) as mrc:
        mrc.set_data(raw.astype(np.float32))
        mrc.voxel_size = target_vsize

    with h5py.File(out_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("labels/actin", data=labels, compression="gzip")
        f.create_dataset("sample_mask", data=mask, compression="gzip")

    print(f"Rescaled {raw_path.stem} to {target_vsize}Å with shape {raw.shape}.")


def main():
    DATA_ROOT = Path(
    "/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data"
    "/experimental/optogenetics"
    )
    RAW_DIR = DATA_ROOT / "raw"
    LABEL_DIR = DATA_ROOT / "labels"
    MASK_DIR = DATA_ROOT / "sample_masks" 
    RESULTS_DIR = Path(
        "/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data"
        "/predictions/optogenetics"
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    target_vsize = 10.0  # Angstrom

    raw_paths = sorted(RAW_DIR.glob("*.mrc"))
    label_paths = sorted(LABEL_DIR.glob("*.mrc"))
    mask_paths = sorted(MASK_DIR.glob("*.mrc"))

    assert len(raw_paths) == len(label_paths) == len(mask_paths), (
        "Mismatch between number of tomos, labels, and sample masks."
    )

    for raw_path, label_path, mask_path in zip(raw_paths, label_paths, mask_paths):
        assert raw_path.stem == label_path.stem == mask_path.stem, (
            f"Name mismatch: tomo {raw_path.stem}, label {label_path.stem}"
        )
        out_path = RESULTS_DIR / f"{raw_path.stem}.h5"
        if out_path.exists():
            print(f"Skipping {out_path.name}, already exists.")
        else:
            process_sample(raw_path, label_path, mask_path, out_path, target_vsize)

if __name__ == "__main__":
    main()
