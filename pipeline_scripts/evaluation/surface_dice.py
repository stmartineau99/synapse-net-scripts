#!/bin/env python3
import sys
import os
import argparse
from pathlib import Path

MEMBRAIN_SEG_PATH = "/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/source/membrain-seg/src"
if MEMBRAIN_SEG_PATH not in sys.path:
    sys.path.insert(0, MEMBRAIN_SEG_PATH)

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from membrain_seg.segmentation.skeletonize import skeletonization
    from membrain_seg.benchmark.metrics import masked_surface_dice
except ImportError:
    raise ImportError("membrain_seg not found in path. Download source code:" \
    "https://github.com/teamtomo/membrain-seg/tree/main/src/membrain_seg")
    exit()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--gt_key", type=str, default="labels/actin")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--use_sample_mask", action="store_true", help="Mask prediction and GT with sample_mask.")
    parser.add_argument("--check", action="store_true", help="Visualize intermediate outputs in Napari")
    return parser.parse_args()


def load_segmentation(file_path, key):
    with h5py.File(file_path, "r") as f:
        data = f[key][:]
    return data


def evaluate_surface_dice(pred, gt, raw, check):
    gt_skeleton = skeletonization(gt == 1, batch_size=100000)
    pred_skeleton = skeletonization(pred, batch_size=100000)
    mask = gt != 2

    if check:
        import napari
        v = napari.Viewer()
        v.add_image(raw)
        v.add_labels(gt, name="gt")
        v.add_labels(gt_skeleton.astype(np.uint16), name="gt_skeleton")
        v.add_labels(pred, name="pred")
        v.add_labels(pred_skeleton.astype(np.uint16), name="pred_skeleton")
    
        napari.run()

    surf_dice, confusion_dict = masked_surface_dice(
        pred_skeleton, gt_skeleton, pred, gt, mask
    )
    return surf_dice, confusion_dict


def process_file(data_path, seg_key, gt_key, check, use_sample_mask=False):
    try:
        pred = load_segmentation(data_path, seg_key)
        gt = load_segmentation(data_path, gt_key)
        raw = load_segmentation(data_path, "raw")
        with h5py.File(data_path, "r") as f:
            mask = f["sample_mask"][:] if use_sample_mask and "sample_mask" in f else None

        gt_bin = (gt == 1).astype(np.uint8)
        pred_bin = pred.astype(np.uint8)
        if mask is not None:
            gt_bin = gt_bin * mask.astype(np.uint8)
            pred_bin = pred_bin * mask.astype(np.uint8)

        dice, confusion = evaluate_surface_dice(pred_bin, gt_bin, raw, check)
        return {
            "tomo_name": data_path.name,
            "surface_dice": dice,
            **confusion
        }

    except Exception as e:
        import traceback
        print(f"Error processing {data_path}: {e}")
        traceback.print_exc()
        return None


def collect_results(data_paths, seg_key, gt_key, check=False, use_sample_mask=False):
    results = []

    for data_path in tqdm(data_paths, desc="Processing segmentations"):
        print(data_path)

        with h5py.File(data_path, "r") as f:
            if gt_key not in f:
                print(f"GT labels not found in {data_path.stem}.")
                continue
            if seg_key not in f:
                print(f"{seg_key} not found in {data_path.stem}.")
                continue

        res = process_file(data_path, seg_key, gt_key, check, use_sample_mask)
        if res is None:
            continue

        res["input_folder"] = data_path.parent.name
        results.append(res)

    return results


def save_results(results, output_file):
    if not results:
        print("No results to save.")
        return

    new_df = pd.DataFrame(results)

    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        combined_df = existing_df[
            ~existing_df.set_index(["tomo_name", "input_folder"]).index.isin(
                new_df.set_index(["tomo_name", "input_folder"]).index
            )
        ]
        final_df = pd.concat([combined_df, new_df], ignore_index=True)
    else:
        final_df = new_df

    final_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}.")


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    gt_key = args.gt_key
    seg_key = f"segmentations/{args.model_name}"

    data_paths = sorted(Path(args.data_dir).glob("*.h5"))
    if not data_paths:
        raise FileNotFoundError("No h5 files found.")

    output_file = results_dir / f"{args.model_name}_surface_dice.csv"
    results = collect_results(data_paths, seg_key, gt_key, args.check, args.use_sample_mask)

    save_results(results, output_file)

if __name__ == "__main__":
    main()
