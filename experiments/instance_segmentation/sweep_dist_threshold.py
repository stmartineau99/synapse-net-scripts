"""Sweep the DIST graph-cut threshold from a single cached DIST run.

Runs TARDIS DIST once, then re-applies PropGreedyGraphCut at each threshold on the
cached graph (no DIST re-run). Lower threshold connects more fragments. An instance
mask per threshold is written to <out_dir>/dist_sweep for visual inspection.
"""
import argparse
import copy
import csv
from pathlib import Path

import h5py
import mrcfile
import numpy as np
from scipy.ndimage import binary_erosion

from tardis_em.cnn.data_processing.draw_mask import draw_instances
from tardis_em.dist_pytorch.utils.segment_point_cloud import PropGreedyGraphCut
from tardis_em.utils.export_data import to_mrc
from tardis_em.utils.predictor import GeneralPredictor
from tardis_em_analysis.filament_utils import sort_by_length

THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5]
CIRCLE_SIZE = 80


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep DIST graph-cut threshold on a single cached DIST run.")
    parser.add_argument("--data_path", type=str, required=True, help="Input h5 with the segmentation.")
    parser.add_argument("--seg_key", type=str, required=True)
    parser.add_argument("--pixel_size", type=float, required=True)
    parser.add_argument("--erosion", type=int, default=3)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    return parser.parse_args()


def stage_mask(mask, save_path, pixel_size, erosion):
    mask = (mask > 0).astype(np.int8)
    eroded = binary_erosion(mask, iterations=erosion).astype(np.int8) if erosion else mask
    with mrcfile.new(save_path, overwrite=True) as mrc:
        mrc.set_data(eroded)
        mrc.voxel_size = pixel_size


def run_dist(mrc_path, pixel_size):
    predictor = GeneralPredictor(
        predict="Actin",
        dir_s=str(mrc_path),
        binary_mask=True,
        output_format="None_csv",
        patch_size=128,
        convolution_nn="fnet_attn",
        cnn_threshold="0.25",
        dist_threshold=0.5,
        points_in_patch=900,
        predict_with_rotation=False,
        instances=True,
        device_s="0",
        debug=False,
        checkpoint=[None, None],
        local_only=True,
        correct_px=pixel_size,
        normalize_px=0,
        filter_by_length=1000,
        connect_splines=2500,
        connect_cylinder=250,
        skeletonize=False,
        downsample="greedy",
        tardis_logo=False,
    )
    predictor()
    return predictor


def segment_at_threshold(predictor, threshold):
    graph_cut = PropGreedyGraphCut(threshold=threshold, connection=2, smooth=True)
    segments = graph_cut.patch_to_segment(
        graph=copy.deepcopy(predictor.graphs),
        coord=predictor.pc_ld,
        idx=copy.deepcopy(predictor.output_idx),
        prune=5,
        sort=True,
    )
    if segments is None:
        return None
    return sort_by_length(segments)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    sweep_dir = out_dir / "dist_sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tomogram = Path(args.data_path).stem
    name = f"{tomogram}_{args.tag}"

    with h5py.File(args.data_path, "r") as f:
        mask = f[args.seg_key][:]

    staged_path = tmp_dir / f"{name}.mrc"
    stage_mask(mask, staged_path, args.pixel_size, args.erosion)
    print(f"{tomogram}: staged mask -> {staged_path}")

    predictor = run_dist(staged_path, args.pixel_size)
    print(f"{tomogram}: DIST done, sweeping {len(THRESHOLDS)} thresholds.")

    rows = []
    for threshold in THRESHOLDS:
        segments = segment_at_threshold(predictor, threshold)
        n_inst = 0 if segments is None else len(np.unique(segments[:, 0]))
        rows.append({"dist_threshold": threshold, "n_instances": n_inst})
        print(f"  dist_threshold={threshold}: {n_inst} instances")

        if segments is None:
            continue
        coord = segments.copy()
        coord[:, 1:] = coord[:, 1:] / args.pixel_size
        instance_mask = draw_instances(
            mask_size=predictor.org_shape,
            coordinate=coord,
            pixel_size=args.pixel_size,
            circle_size=CIRCLE_SIZE,
        )
        to_mrc(
            data=instance_mask,
            pixel_size=args.pixel_size,
            file_dir=str(sweep_dir / f"{name}_dist{threshold}_instance_mask.mrc"),
            label=None,
        )

    csv_path = sweep_dir / f"{name}_dist_sweep.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dist_threshold", "n_instances"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
