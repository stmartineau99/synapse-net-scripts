import argparse
import os

import mrcfile
import numpy as np
import napari

DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")

# (display label, tag) for each config's <tomogram>_<tag>_instance_mask.mrc
CONFIGS = [
    ("baseline", "baseline"),   # skimage skeleton + voxel downsample, no erosion
    ("greedy_skel", "greedy_skel"),     # skel + greedy
    ("skeleton_cc", "gt_skeleton"),     # skimage skeleton + connected components
    ("greedy_eroded", "greedy"),       # erosion x3 + greedy downsample
]


def main():
    parser = argparse.ArgumentParser(description="Overlay instance masks from multiple configs over the mask in napari.")
    parser.add_argument("--tomogram", default="00004", help="Tomogram id, e.g. 00004.")
    parser.add_argument("--dir", default=DEFAULT_DIR, help="Directory with <tomogram>_<tag>_instance_mask.mrc.")
    parser.add_argument("--mask", default=None, help="Base mask mrc (default <dir>/<tomogram>_gt_mask.mrc).")
    args = parser.parse_args()

    mask_path = args.mask or os.path.join(args.dir, f"{args.tomogram}_gt_mask.mrc")

    viewer = napari.Viewer()
    if os.path.exists(mask_path):
        with mrcfile.open(mask_path, permissive=True) as mrc:
            mask = np.asarray(mrc.data)
        viewer.add_image(mask, name="mask", colormap="gray", opacity=0.5)
    else:
        print(f"mask not found: {mask_path}")

    for label, tag in CONFIGS:
        instance_path = os.path.join(args.dir, f"{args.tomogram}_{tag}_instance_mask.mrc")
        if os.path.exists(instance_path):
            with mrcfile.open(instance_path, permissive=True) as mrc:
                instances = np.asarray(mrc.data)
            viewer.add_labels(instances.astype(np.int32), name=f"{args.tomogram}_{label}")
            print(f"loaded {label}: {instance_path}")
        else:
            print(f"missing {label}: {instance_path}")

    napari.run()


if __name__ == "__main__":
    main()
