import h5py
import numpy as np
from pathlib import Path

KEYS = ["raw", "labels/actin", "sample_mask"]
TRAIN_FRACTION = 0.7
VAL_FRACTION = 0.2
TEST_FRACTION = 0.1


def get_quadrant_slices(y, x):
    y_mid, x_mid = y // 2, x // 2
    return [
        (slice(None, y_mid), slice(None, x_mid)),
        (slice(None, y_mid), slice(x_mid, None)),
        (slice(y_mid, None), slice(None, x_mid)),
        (slice(y_mid, None), slice(x_mid, None)),
    ]


def process_subvolume(src_path, out_path, q_idx):
    with h5py.File(src_path, "r") as src, h5py.File(out_path, "w") as dst:
        _, y, x = src["raw"].shape
        sy, sx = get_quadrant_slices(y, x)[q_idx]
        for key in KEYS:
            if key not in src:
                continue
            data = src[key][:]
            chunk = data[:, sy, sx] if data.ndim == 3 else data
            dst.create_dataset(key, data=chunk, compression="gzip")
    print(f"Processed {out_path.name}. Shape: {chunk.shape}")


def main():
    DATA_ROOT = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data")
    SRC_DIR = DATA_ROOT / "predictions/deepict"
    OUT_DIR = DATA_ROOT / "experimental/deepict/subvolumes"

    src_paths = sorted(SRC_DIR.glob("*.h5"))
    quadrants = [(p, q) for p in src_paths for q in range(4)]

    n = len(quadrants)
    train_end = int(TRAIN_FRACTION * n)
    val_end = int((TRAIN_FRACTION + VAL_FRACTION) * n)
    train_idx = list(range(0, train_end))
    val_idx = list(range(train_end, val_end))
    test_idx = list(range(val_end, n))

    splits = [("train", train_idx), ("val", val_idx), ("test", test_idx)]

    for split, indices in splits:
        split_dir = OUT_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{split} ({len(indices)} subvolumes)")

        for i in indices:
            src_path, q_idx = quadrants[i]
            out_path = split_dir / f"{src_path.stem}_{q_idx}.h5"
            if out_path.exists():
                print(f"  Skipping {out_path.name}, already exists.")
                continue
            process_subvolume(src_path, out_path, q_idx)

if __name__ == "__main__":
    main()
