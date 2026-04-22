import h5py
import numpy as np
import torch_em
from pathlib import Path
from synapse_net.file_utils import read_mrc

DATA_ROOT = Path(
    "/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data"
    "/experimental/optogenetics"
)
TOMO_DIR = DATA_ROOT / "tomos"
LABEL_DIR = DATA_ROOT / "labels"
OUT_DIR = DATA_ROOT / "h5"

TARGET_VOXEL_SIZE = 10.0  # Angstrom
CROP_SHAPE = (122, 971, 971)  # (z, y, x) center crop


def center_crop(volume, shape):
    crops = tuple(
        slice((s - c) // 2, (s - c) // 2 + c)
        for s, c in zip(volume.shape, shape)
    )
    return volume[crops]


def get_rescaled_shape(tomo_path):
    _, voxel_size = read_mrc(tomo_path)
    import mrcfile
    with mrcfile.open(tomo_path, permissive=True) as m:
        orig_shape = m.data.shape
    scale = tuple(
        voxel_size[ax] / (TARGET_VOXEL_SIZE / 10) for ax in ("z", "y", "x")
    )
    return tuple(int(s * sc) for s, sc in zip(orig_shape, scale))


def inspect_shapes(tomo_paths):
    shapes = []
    for p in tomo_paths:
        shape = get_rescaled_shape(p)
        shapes.append(shape)
        print(f"  {p.name}: {shape}")
    shapes = np.array(shapes)
    print(f"\nMin shape: {tuple(shapes.min(axis=0))}")
    print(f"Max shape: {tuple(shapes.max(axis=0))}")


def process(tomo_path, label_path, out_path):
    raw, voxel_size = read_mrc(tomo_path)
    labels, _ = read_mrc(label_path)

    scale = tuple(
        voxel_size[ax] / (TARGET_VOXEL_SIZE / 10) for ax in ("z", "y", "x")
    )
    raw = torch_em.transform.generic.Rescale(scale)(raw)
    labels = torch_em.transform.generic.Rescale(scale, is_label=True)(labels)

    if CROP_SHAPE is not None:
        raw = center_crop(raw, CROP_SHAPE)
        labels = center_crop(labels, CROP_SHAPE)

    with h5py.File(out_path, "w") as f:
        f.create_dataset("raw", data=raw, compression="gzip")
        f.create_dataset("/labels/actin", data=labels, compression="gzip")

    print(f"Saved {out_path.name}  shape={raw.shape}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tomo_paths = sorted(TOMO_DIR.glob("*.mrc"))
    label_paths = sorted(LABEL_DIR.glob("*.mrc"))

    assert len(tomo_paths) == len(label_paths), (
        "Mismatch between tomos and labels"
    )
    for tp, lp in zip(tomo_paths, label_paths):
        assert tp.stem == lp.stem, f"Name mismatch: {tp.stem} vs {lp.stem}"

    print("Rescaled shapes:")
    inspect_shapes(tomo_paths)

    if CROP_SHAPE is None:
        print("\nCROP_SHAPE is None — set it and re-run to process files.")
        return

    for tomo_path, label_path in zip(tomo_paths, label_paths):
        out_path = OUT_DIR / f"{tomo_path.stem}.h5"
        if out_path.exists():
            print(f"Skipping {out_path.name}, already exists.")
            continue
        process(tomo_path, label_path, out_path)


if __name__ == "__main__":
    main()
