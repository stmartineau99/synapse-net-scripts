import argparse
import os

import mrcfile
import numpy as np
import napari

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_TMP = os.path.join(BASE_DIR, "tmp")

# (display label, suffix, face color) for each point cloud <name>_<suffix>.npy
POINT_CLOUDS = [
    ("skel_voxel", "skel_voxel_points", "cyan"),
    ("skel_greedy", "skel_greedy_points", "yellow"),
    ("eroded_greedy", "eroded_greedy_points", "magenta"),
]

# (display label, path relative to this script; {name} is the sample name) for each label layer
LABEL_LAYERS = [
    ("eroded3", "eroded/{name}_gt_erode3_cc26.mrc"),
    ("skeleton", "tmp/{name}_gt_skeleton.mrc"),
]


def main():
    parser = argparse.ArgumentParser(description="Overlay saved point clouds on their mask in napari.")
    parser.add_argument("--name", required=True, help="Sample name, e.g. 00004_gt.")
    parser.add_argument("--tmp_dir", default=DEFAULT_TMP, help="Directory with <name>_<suffix>.npy point clouds and <name>_mask.mrc.")
    parser.add_argument("--point_size", type=float, default=2.0, help="Point display size (default 2).")
    parser.add_argument("--mask", default=None, help="Base mask mrc (default <tmp_dir>/<name>_gt_mask.mrc).")
    args = parser.parse_args()

    mask_path = args.mask or os.path.join(args.tmp_dir, f"{args.name}_gt_mask.mrc")

    viewer = napari.Viewer()
    if os.path.exists(mask_path):
        with mrcfile.open(mask_path, permissive=True) as mrc:
            mask = np.asarray(mrc.data)
        viewer.add_image(mask, name="mask", colormap="gray", opacity=0.5)
    else:
        print(f"mask not found: {mask_path}")

    for label, rel_path in LABEL_LAYERS:
        layer_path = os.path.join(BASE_DIR, rel_path.format(name=args.name))
        if os.path.exists(layer_path):
            with mrcfile.open(layer_path, permissive=True) as mrc:
                labels = (np.asarray(mrc.data) > 0).astype(np.int32)
            viewer.add_labels(labels, name=f"{args.name}_{label}", opacity=1)
            print(f"loaded {label}: {layer_path}")
        else:
            print(f"missing {label}: {layer_path}")

    for label, suffix, face_color in POINT_CLOUDS:
        points_path = os.path.join(args.tmp_dir, f"{args.name}_{suffix}.npy")
        if os.path.exists(points_path):
            # build_point_cloud is [X,Y,Z] -> napari [Z,Y,X]
            points_zyx = np.load(points_path)[:, -3:][:, ::-1]
            viewer.add_points(points_zyx, name=f"{args.name}_{label}", size=args.point_size,
                              face_color=face_color, opacity=1, out_of_slice_display=True)
            print(f"loaded {label}: {len(points_zyx)} points from {points_path}")
        else:
            print(f"missing {label}: {points_path}")

    napari.run()


if __name__ == "__main__":
    main()
