from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import binary_erosion

from tardis_em.dist_pytorch.utils.build_point_cloud import (
    BuildPointCloud,
    greedy_downsampling,
)


PARENT_DIR = "/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage"

DATA_DIR = None
DATA_PATHS = [f"{PARENT_DIR}/data/experimental/deepict/h5/00004.h5"]
OUTPUT_DIR = f"{PARENT_DIR}/data/predictions/deepict/csv/instances/tmp"
SEG_KEY = "labels/actin"        # use segmentations/<model> for model predictions
EROSION = 3
DOWNSAMPLE = 5


# (tag, erode, skeletonize, downsample)
CONFIGS = [
    ("skel_voxel",    False, True,  "voxel"),
    ("skel_greedy",   False, True,  "greedy"),
    ("eroded_greedy", True,  False, "greedy"),
]


def build_clouds(mask, erosion, downsample):
    mask = (mask > 0).astype(np.int8)
    eroded = binary_erosion(mask, iterations=erosion).astype(np.int8) if erosion else mask
    builder = BuildPointCloud()

    clouds = {}
    for tag, erode, skeletonize, method in CONFIGS:
        image = eroded if erode else mask
        if method == "voxel":
            _, points = builder.build_point_cloud(
                image=image, down_sampling=downsample, skeletonize=skeletonize
            )
        else:
            hd = builder.build_point_cloud(image=image, skeletonize=skeletonize)
            points = greedy_downsampling(hd, downsample)
        clouds[tag] = points
    return clouds


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    if DATA_DIR:
        data_paths = sorted(Path(DATA_DIR).glob("*.h5"))
    elif DATA_PATHS:
        data_paths = [Path(p) for p in DATA_PATHS]
    else:
        data_paths = []
    if not data_paths:
        raise FileNotFoundError("No h5 files found.")

    for data_path in data_paths:
        tomogram = data_path.stem
        with h5py.File(data_path, "r") as f:
            if SEG_KEY not in f:
                print(f"{tomogram}: '{SEG_KEY}' not found, skipping.")
                continue
            mask = f[SEG_KEY][:]

        clouds = build_clouds(mask, EROSION, DOWNSAMPLE)
        for tag, points in clouds.items():
            points_path = output_dir / f"{tomogram}_{tag}_points.npy"
            np.save(points_path, points)
            print(f"{tomogram}: {points.shape[0]} {tag} pts -> {points_path}")

    print(f"Point clouds saved in {output_dir}")


if __name__ == "__main__":
    main()
