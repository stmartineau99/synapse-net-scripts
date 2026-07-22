"""Prepare Deepict actin tomograms and GT labels as training data at 10 A voxel size.

The Deepict tomograms and their actin ground-truth labels were downloaded and
reconstructed at bin 4 (~13.48 A for 00004, ~13.8 A for 00011 and 00012). The simulation
training pipeline expects a uniform 10 A voxel size, so this script rescales every
volume up to 10 A.

Per sample it:
  1. Loads the raw tomogram, and labels/actin from the cleaned h5 when GT exists
     (00004 and 00012 have GT; 00011 does not).
  2. Loads the slab sample mask and the mask center-of-mass coordinates.
  3. Center-crops raw, mask, and labels to CROP_SHAPE around the mask center of mass.
  4. Rescales each volume from its source voxel size to 10 A. read_voxel_size reports
     the source spacing in nm, and target_vsize / 10 converts the 10 A target into nm,
     so the factor reduces to source_nm / target_nm (~1.35x upsampling from bin 4).
  5. Writes the rescaled raw as an mrc (into rescaled_10A/) and an h5 holding raw,
     labels/actin (if present), and sample_mask.

Superseded by new prepare_deepict.py, which processes tomograms directly reconstructed at 10A,
and rescales the GT labels onto that grid.
"""
import json
import h5py
import mrcfile
import numpy as np
import torch_em
from pathlib import Path
from synapse_net.file_utils import read_voxel_size

CROP_SHAPE = (240, 928, 928)


def center_crop(volume, crop_shape, coords):
    cz, cy, cx = crop_shape
    sz, sy, sx = volume.shape
    center_z, center_y, center_x = coords
    z_start = np.clip(center_z - cz / 2, a_min=0, a_max=sz - cz).astype(int)
    y_start = np.clip(center_y - cy / 2, a_min=0, a_max=sy - cy).astype(int)
    x_start = np.clip(center_x - cx / 2, a_min=0, a_max=sx - cx).astype(int)
    return volume[z_start:z_start + cz, y_start:y_start + cy, x_start:x_start + cx]


def process_sample(raw_path, h5_path, mask_path, json_path, out_path, target_vsize, has_labels):
    source_vsize = read_voxel_size(raw_path)
    if has_labels:
        with h5py.File(h5_path, "r") as f:
            raw = f["raw"][:]
            labels = f["labels/actin"][:]
    else:
        with mrcfile.open(raw_path, permissive=True) as mrc:
            raw = np.asarray(mrc.data)
    with mrcfile.open(mask_path, permissive=True) as mrc:
        mask = np.asarray(mrc.data)

    with open(json_path) as f:
        info = json.load(f)
        coords = info["mask_center_of_mass_voxels"]

    raw = center_crop(raw, CROP_SHAPE, coords)
    if has_labels:
        labels = center_crop(labels, CROP_SHAPE, coords)
    mask = center_crop(mask, CROP_SHAPE, coords)

    scale = tuple(
        source_vsize[ax] / (target_vsize / 10) for ax in ("z", "y", "x")
    )
    raw = torch_em.transform.generic.Rescale(scale)(raw).astype(np.float32)
    mask = torch_em.transform.generic.Rescale(scale, is_label=True)(mask).astype(np.uint8)
    if has_labels:
        labels = torch_em.transform.generic.Rescale(scale, is_label=True)(labels).astype(np.uint8)

    rescaled_raw_dir = raw_path.parent / "rescaled_10A"
    rescaled_raw_dir.mkdir(exist_ok=True)
    with mrcfile.new(rescaled_raw_dir / raw_path.name, overwrite=True) as mrc:
        mrc.set_data(raw)
        mrc.voxel_size = target_vsize

    with h5py.File(out_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        if has_labels:
            f.create_dataset("labels/actin", data=labels, compression="gzip")
        f.create_dataset("sample_mask", data=mask, compression="gzip")

    print(f"Processed {raw_path.stem}. Rescaled shape: {raw.shape}")


def main():
    DATA_ROOT = Path(
        "/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/experimental/deepict"
    )
    RAW_DIR = DATA_ROOT / "raw"
    H5_DIR = DATA_ROOT / "h5/cleaned"
    MASK_DIR = DATA_ROOT / "sample_masks_dlm"
    OUT_DIR = Path(
        "/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/experimental/deepict/h5"
    )

    target_vsize = 10.0  # Angstrom

    raw_paths = sorted(RAW_DIR.glob("*.mrc"))

    for raw_path in raw_paths:
        h5_path = H5_DIR / f"{raw_path.stem}_cleaned.h5"
        has_labels = True if h5_path.exists() else False
        mask_path = MASK_DIR / f"{raw_path.stem}_slab_mask.mrc"
        json_path = MASK_DIR / f"{raw_path.stem}_info.json"
        out_path = OUT_DIR / f"{raw_path.stem}.h5"

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file does not exist: {mask_path}.")

        if out_path.exists():
            print(f"Skipping {out_path.name}, already exists.")
        else:
            process_sample(
                raw_path, h5_path, mask_path, json_path, out_path, target_vsize, has_labels
            )


if __name__ == "__main__":
    main()
