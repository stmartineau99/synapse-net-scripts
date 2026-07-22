import h5py
import mrcfile
import numpy as np
from pathlib import Path


def process_sample(raw_path, label_path, mask_path, output_path):
    with mrcfile.open(raw_path, permissive=True) as mrc:
        raw = np.asarray(mrc.data).astype(np.float32)

    has_labels = label_path.exists()
    if has_labels:
        with mrcfile.open(label_path, permissive=True) as mrc:
            labels = np.asarray(mrc.data).astype(np.uint8)

    has_mask = mask_path.exists()
    if has_mask:
        with mrcfile.open(mask_path, permissive=True) as mrc:
            mask = np.asarray(mrc.data).astype(np.uint8)

    with h5py.File(output_path, "w") as f_out:
        f_out.create_dataset("raw", data=raw, compression="gzip")
        if has_labels:
            f_out.create_dataset("labels/htt", data=labels, compression="gzip")
        if has_mask:
            f_out.create_dataset("sample_mask", data=mask, compression="gzip")

    print(f"Processed {raw_path.stem}. Shape: {raw.shape}")


def main():
    DATA_ROOT = Path(
        "/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/experimental/htt"
    )
    RAW_DIR = DATA_ROOT / "raw"
    LABEL_DIR = DATA_ROOT / "labels"
    MASK_DIR = DATA_ROOT / "sample_masks"
    RESULTS_DIR = DATA_ROOT / "h5"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    raw_paths = sorted(RAW_DIR.glob("*.mrc"))

    for raw_path in raw_paths:
        label_path = LABEL_DIR / f"{raw_path.stem}.mrc"
        mask_path = MASK_DIR / f"{raw_path.stem}_slab_mask.mrc"
        output_path = RESULTS_DIR / f"{raw_path.stem}.h5"

        if output_path.exists():
            print(f"  Skipping {output_path.name}, already exists.")
            continue

        process_sample(raw_path, label_path, mask_path, output_path)


if __name__ == "__main__":
    main()
