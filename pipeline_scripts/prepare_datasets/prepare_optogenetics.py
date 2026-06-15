import os
import json
import random
import h5py
import mrcfile
import numpy as np
import torch_em
from pathlib import Path
from synapse_net.file_utils import read_mrc

CROP_SHAPE = (82, 652, 652)  # (z, y, x) in original voxels
N_TRAIN = 10
N_VAL = 2
N_TEST = 2
SEED = 42

def create_symlink(src_path: Path, dest_path: Path) -> None:
    if dest_path.is_symlink():
        print(f"Symlink already exists: {dest_path} -> {os.readlink(dest_path)}")
        return
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src_path, dest_path)
    if os.path.islink(dest_path):
        print(f"Created symlink: {dest_path} -> {os.readlink(dest_path)}")
    else:
        print("Symlink creation failed.")

def center_crop(volume, crop_shape, coords):
    cz, cy, cx = crop_shape
    sz, sy, sx = volume.shape
    center_z, center_y, center_x = coords
    z_start = np.clip(center_z - cz / 2, a_min=0, a_max=sz - cz).astype(int)
    y_start = np.clip(center_y - cy / 2, a_min=0, a_max=sy - cy).astype(int)
    x_start = np.clip(center_x - cx / 2, a_min=0, a_max=sx - cx).astype(int)

    return volume[
        z_start:z_start + cz,
        y_start:y_start + cy,
        x_start:x_start + cx,
    ]

def process_sample(raw_path, label_path, mask_path, json_path, output_path, target_vsize):
    raw, source_vsize = read_mrc(raw_path)
    labels, _ = read_mrc(label_path)
    mask, _ = read_mrc(mask_path)

    with open(json_path) as f:
        info = json.load(f)
        coords = info["mask_center_of_mass_voxels"]

    raw = center_crop(raw, CROP_SHAPE, coords)
    labels = center_crop(labels, CROP_SHAPE, coords)
    mask = center_crop(mask, CROP_SHAPE, coords)

    scale = tuple(
        source_vsize[ax] / (target_vsize / 10) for ax in ("z", "y", "x")
    )
    raw = torch_em.transform.generic.Rescale(scale)(raw).astype(np.float32)
    labels = torch_em.transform.generic.Rescale(scale, is_label=True)(labels).astype(np.uint8)
    mask = torch_em.transform.generic.Rescale(scale, is_label=True)(mask).astype(np.uint8)

    rescaled_raw_dir = raw_path.parent / "rescaled_10A"
    rescaled_raw_dir.mkdir(exist_ok=True)
    #with mrcfile.new(rescaled_raw_dir / raw_path.name, overwrite=True) as mrc:
    #    mrc.set_data(raw)
    #    mrc.voxel_size = target_vsize

    rescaled_mask_dir = mask_path.parent / "rescaled_10A"
    rescaled_mask_dir.mkdir(exist_ok=True)
    #with mrcfile.new(rescaled_mask_dir / raw_path.name, overwrite=True) as mrc:
    #    mrc.set_data(mask)
    #    mrc.voxel_size = target_vsize

    with h5py.File(output_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("labels/actin", data=labels, compression="gzip")
        f.create_dataset("sample_mask", data=mask, compression="gzip")

    print(f"Processed {raw_path.stem}. Rescaled shape: {raw.shape}")
    print(f"Processed {mask_path.stem}. Rescaled shape: {mask.shape}")


def main():
    DATA_ROOT = Path(
        "/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data"
    )
    RAW_DIR = DATA_ROOT / "experimental/optogenetics/raw"
    LABEL_DIR = DATA_ROOT / "experimental/optogenetics/labels"
    MASK_DIR = DATA_ROOT / "experimental/optogenetics/sample_masks"
    OUTPUT_DIR = DATA_ROOT / "experimental/optogenetics/h5"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    target_vsize = 10.0  # Angstrom

    raw_paths = sorted(RAW_DIR.glob("*.mrc"))

    rng = random.Random(SEED)
    rng.shuffle(raw_paths)

    splits = {
        "train": raw_paths[:N_TRAIN],
        "val": raw_paths[N_TRAIN:N_TRAIN + N_VAL],
        "test": raw_paths[N_TRAIN + N_VAL:],
        }

    for split, paths in splits.items():
        print(f"\n{split} ({len(paths)} tomograms)")
        split_dir = OUTPUT_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for raw_path in paths:
            label_path = LABEL_DIR / f"{raw_path.stem}.mrc"
            mask_path = MASK_DIR / f"{raw_path.stem}_slab_mask.mrc"
            json_path = MASK_DIR / f"{raw_path.stem}_info.json"
            output_path = split_dir / f"{raw_path.stem}.h5"

            if not label_path.exists():
                raise FileNotFoundError(f"Label file does not exist: {label_path}.")
            if not mask_path.exists():
                raise FileNotFoundError(f"Sample mask file does not exist: {mask_path}.")

            if output_path.exists():
                print(f"Skipping {output_path.name}, already exists.")
            else:
                process_sample(
                    raw_path, label_path, mask_path, json_path, output_path, target_vsize
                )

    create_symlink(OUTPUT_DIR / "test", DATA_ROOT / "predictions/opto")
if __name__ == "__main__":
    main()
