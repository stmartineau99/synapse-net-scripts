import argparse
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

from tardis_em.dist_pytorch.utils.build_point_cloud import BuildPointCloud


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--data_paths", type=str, nargs="+", default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seg_key", type=str, required=True)
    parser.add_argument("--erosion", type=int, default=3)
    parser.add_argument("--merge_radius", type=float, default=2)
    parser.add_argument("--tag", type=str, required=True)
    return parser.parse_args()


def merge_close_points(points, radius):
    parent = np.arange(len(points))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in cKDTree(points).query_pairs(r=radius):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    clusters = {}
    for i in range(len(points)):
        clusters.setdefault(find(i), []).append(i)
    return np.array([points[idx].mean(axis=0) for idx in clusters.values()])


def stage_mask(mask, ds_path, merged_path, erosion, merge_radius):
    binary = (mask > 0).astype(np.int8)
    eroded = binary_erosion(binary, iterations=erosion).astype(np.int8)

    _, point_cloud = BuildPointCloud().build_point_cloud(
        eroded, down_sampling=5, skeletonize=False
    )
    merged = merge_close_points(point_cloud, merge_radius)

    np.save(ds_path, point_cloud)
    np.save(merged_path, merged)
    return point_cloud, merged


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if args.data_dir:
        data_paths = sorted(Path(args.data_dir).glob("*.h5"))
    elif args.data_paths:
        data_paths = [Path(p) for p in args.data_paths]
    if not data_paths:
        raise FileNotFoundError("No h5 files found.")

    for data_path in data_paths:
        tomogram = data_path.stem
        with h5py.File(data_path, "r") as f:
            if args.seg_key not in f:
                print(f"{tomogram}: '{args.seg_key}' not found, skipping.")
                continue
            mask = f[args.seg_key][:]

        name = f"{tomogram}_{args.tag}"
        ds_path = tmp_dir / f"{name}_points_ds.npy"
        merged_path = tmp_dir / f"{name}_points_merged.npy"
        point_cloud, merged = stage_mask(
            mask, ds_path, merged_path, args.erosion, args.merge_radius
        )
        print(
            f"{tomogram}: {len(point_cloud)} points after down_sample"
            f"-> {len(merged)} after merge (radius {args.merge_radius})."
        )

    print(f"Merged point clouds saved in {tmp_dir}")


if __name__ == "__main__":
    main()
