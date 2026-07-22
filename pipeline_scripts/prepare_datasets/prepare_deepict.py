import json
import h5py
import mrcfile
import numpy as np
import torch_em
from pathlib import Path
from synapse_net.file_utils import read_voxel_size

CROP_SHAPE = (260, 1250, 1250)
AXES = ("z", "y", "x")


def center_crop(volume, crop_shape, coords):
    cz, cy, cx = crop_shape
    sz, sy, sx = volume.shape
    center_z, center_y, center_x = coords
    z_start = np.clip(center_z - cz / 2, a_min=0, a_max=sz - cz).astype(int)
    y_start = np.clip(center_y - cy / 2, a_min=0, a_max=sy - cy).astype(int)
    x_start = np.clip(center_x - cx / 2, a_min=0, a_max=sx - cx).astype(int)
    return volume[z_start:z_start + cz, y_start:y_start + cy, x_start:x_start + cx]


def process_sample(raw_path, gt_ref_path, h5_path, mask_path, json_path, out_path, has_labels):
    raw_vsize = read_voxel_size(raw_path)
    with mrcfile.open(raw_path, permissive=True) as mrc:
        raw = np.asarray(mrc.data)
    with mrcfile.open(mask_path, permissive=True) as mrc:
        mask = np.asarray(mrc.data)

    with open(json_path) as f:
        info = json.load(f)
    com_angstrom = info["mask_center_of_mass_angstrom"]
    raw_coords = [com_angstrom[i] / (raw_vsize[ax] * 10) for i, ax in enumerate(AXES)]
    mask_coords = info["mask_center_of_mass_voxels"]

    raw = center_crop(raw, CROP_SHAPE, raw_coords).astype(np.float32)
    mask = center_crop(mask, CROP_SHAPE, mask_coords).astype(np.uint8)

    if has_labels:
        gt_vsize = read_voxel_size(gt_ref_path)
        gt_scale = tuple(gt_vsize[ax] / raw_vsize[ax] for ax in AXES)
        with h5py.File(h5_path, "r") as f:
            labels = f["labels/actin"][:]
        labels = torch_em.transform.generic.Rescale(gt_scale, is_label=True)(labels)
        labels = center_crop(labels, CROP_SHAPE, raw_coords).astype(np.uint8)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        if has_labels:
            f.create_dataset("labels/actin", data=labels, compression="gzip")
        f.create_dataset("sample_mask", data=mask, compression="gzip")

    print(f"Processed {raw_path.stem}. Raw shape: {raw.shape}")


def main():
    DATA_ROOT = Path(
        "/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/experimental/deepict"
    )
    RAW_DIR = DATA_ROOT / "raw" / "reconstructed_10A"
    GT_REF_DIR = DATA_ROOT / "raw"
    H5_DIR = DATA_ROOT / "h5" / "cleaned"
    MASK_DIR = DATA_ROOT / "sample_masks"
    OUT_DIR = DATA_ROOT / "h5" / "reconstructed_10A"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_paths = sorted(RAW_DIR.glob("*.mrc"))

    for raw_path in raw_paths:
        gt_ref_path = GT_REF_DIR / f"{raw_path.stem}.mrc"
        h5_path = H5_DIR / f"{raw_path.stem}_cleaned.h5"
        has_labels = h5_path.exists()
        mask_path = MASK_DIR / f"{raw_path.stem}_slab_mask.mrc"
        json_path = MASK_DIR / f"{raw_path.stem}_info.json"
        out_path = OUT_DIR / f"{raw_path.stem}.h5"

        if not mask_path.exists():
            raise FileNotFoundError(f"Mask file does not exist: {mask_path}.")
        if has_labels and not gt_ref_path.exists():
            raise FileNotFoundError(f"GT reference mrc does not exist: {gt_ref_path}.")

        if out_path.exists():
            print(f"Skipping {out_path.name}, already exists.")
        else:
            process_sample(
                raw_path, gt_ref_path, h5_path, mask_path, json_path, out_path, has_labels
            )


if __name__ == "__main__":
    main()
