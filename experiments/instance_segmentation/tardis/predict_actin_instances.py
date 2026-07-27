import argparse
import shutil
from pathlib import Path

import h5py
import mrcfile
import numpy as np
from scipy.ndimage import binary_erosion

from tardis_em.cnn.data_processing.draw_mask import draw_instances
from tardis_em.dist_pytorch.utils.build_point_cloud import BuildPointCloud
from tardis_em.utils.export_data import to_mrc
from tardis_em.utils.predictor import GeneralPredictor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--data_paths", type=str, nargs="+", default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seg_key", type=str, required=True)
    parser.add_argument("--pixel_size", type=float, required=True)
    parser.add_argument("--dist_threshold", type=float, default=0.5)
    parser.add_argument("--filter_by_length", type=int, default=1000)
    parser.add_argument("--connect_splines", type=int, default=2500)
    parser.add_argument("--connect_cylinder", type=int, default=250)
    parser.add_argument("--erosion", type=int, default=3)
    parser.add_argument("--skeletonize", action="store_true")
    parser.add_argument("--save_instance_mask", action="store_true")
    parser.add_argument("--tag", type=str, required=True)
    return parser.parse_args()


def stage_mask(mask, save_path, pixel_size, erosion):
    mask = (mask > 0).astype(np.int8)
    eroded = binary_erosion(mask, iterations=erosion).astype(np.int8) if erosion else mask

    with mrcfile.new(save_path, overwrite=True) as mrc:
        mrc.set_data(eroded)
        mrc.voxel_size = pixel_size


def run_tardis(mrc_path, points_path, pixel_size, args, mask_path=None):
    predictor = GeneralPredictor(
        predict="Actin",
        dir_s=str(mrc_path),
        binary_mask=True,
        output_format="None_csv",
        patch_size=128,
        convolution_nn="fnet_attn",
        cnn_threshold="0.25",
        dist_threshold=args.dist_threshold,
        points_in_patch=900,
        predict_with_rotation=False,
        instances=True,
        device_s="0",
        debug=False,
        checkpoint=[None, None],
        local_only=True,
        correct_px=pixel_size,
        normalize_px=0,
        filter_by_length=args.filter_by_length,
        connect_splines=args.connect_splines,
        connect_cylinder=args.connect_cylinder,
        skeletonize=args.skeletonize,
        downsample="greedy",
        tardis_logo=False,
    )
    predictor()
    #np.save(points_path, predictor.pc_ld)

    if mask_path is not None and predictor.segments is not None:
        segments = predictor.segments.copy()
        segments[:, 1:] = segments[:, 1:] / pixel_size
        instance_mask = draw_instances(
            mask_size=predictor.org_shape,
            coordinate=segments,
            pixel_size=pixel_size,
            circle_size=80, # actin diameter in A
        )
        to_mrc(
            data=instance_mask,
            pixel_size=pixel_size,
            file_dir=str(mask_path),
            label=predictor.instance_header,
        )


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
        tmp_path = tmp_dir / f"{name}.mrc"
        stage_mask(mask, tmp_path, args.pixel_size, args.erosion)
        print(f"{tomogram}: mask saved to {tmp_path}.")

        points_path = tmp_dir / f"{name}_points.npy"
        mask_path = tmp_dir / f"{name}_instance_mask.mrc" if args.save_instance_mask else None
        print(f"{tomogram}: running instance segmentation.")
        run_tardis(tmp_path, points_path, args.pixel_size, args, mask_path=mask_path)

        src_path = tmp_dir / f"{name}_instances.csv"
        dest_path = output_dir / f"{name}_instances.csv"
        if src_path.exists():
            shutil.move(str(src_path), str(dest_path))
        else:
            print(f"TARDIS instances not found: {src_path}")

    print(f"Instances saved in {output_dir}")

if __name__ == "__main__":
    main()
