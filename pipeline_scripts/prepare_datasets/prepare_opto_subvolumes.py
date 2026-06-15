import os
import random
import h5py
import numpy as np
from pathlib import Path

KEYS = ["raw", "labels/actin", "sample_mask"]
N_QUADRANTS = 4
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.2
TEST_FRACTION = 0.1
SEED = 42


def get_quadrant_slices(y, x):
    y_mid, x_mid = y // 2, x // 2
    return [
        (slice(None, y_mid), slice(None, x_mid)),
        (slice(None, y_mid), slice(x_mid, None)),
        (slice(y_mid, None), slice(None, x_mid)),
        (slice(y_mid, None), slice(x_mid, None)),
    ]

def create_symlink(src_path: Path, dest_path: Path) -> None:
    if not src_path.exists():
        print(f"Symlink creation failed: source {src_path} does not exist.")
        return
    if dest_path.is_symlink():
        print(f"Symlink already exists: {dest_path} -> {os.readlink(dest_path)}")
        return
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    os.symlink(src_path, dest_path)
    if os.path.islink(dest_path):
        print(f"Created symlink: {dest_path} -> {os.readlink(dest_path)}")
    else:
        print("Symlink creation failed.")

def process_subvolume(input_path, output_path, q_idx):
    with h5py.File(input_path, "r") as f_in, h5py.File(output_path, "w") as f_out:
        _, y, x = f_in["raw"].shape
        sy, sx = get_quadrant_slices(y, x)[q_idx]
        for key in KEYS:
            if key not in f_in:
                continue
            data = f_in[key][:]
            chunk = data[:, sy, sx] if data.ndim == 3 else data
            f_out.create_dataset(key, data=chunk, compression="gzip")
    print(f"Processed {output_path.name}. Shape: {chunk.shape}")


def main():
    DATA_ROOT = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data")
    INPUT_DIR = DATA_ROOT / "experimental/optogenetics/h5"
    OUTPUT_DIR = DATA_ROOT / "experimental/optogenetics/h5/subvolumes"

    input_paths = sorted(INPUT_DIR.rglob("*.h5"))

    # collect all entries, assign randomly to splits
    all_entries = [[input_path, q] for input_path in input_paths for q in range(N_QUADRANTS)]

    n_train = round(len(all_entries) * TRAIN_FRACTION)
    n_val = round(len(all_entries) * VAL_FRACTION)

    rng = random.Random(SEED)
    rng.shuffle(all_entries)

    splits = {
        "train": all_entries[:n_train],
        "val": all_entries[n_train:n_train + n_val],
        "test": all_entries[n_train + n_val:]
        }

    for split, entries in splits.items():
        print(f"\n{split} ({len(entries)} subvolumes)")
        split_dir = OUTPUT_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for input_path, q_idx in entries:
            output_path = split_dir / f"{input_path.stem}_{q_idx}.h5"
            process_subvolume(input_path, output_path, q_idx)

    create_symlink(OUTPUT_DIR / "test", DATA_ROOT / "predictions/opto/subvolumes")


if __name__ == "__main__":
    main()
