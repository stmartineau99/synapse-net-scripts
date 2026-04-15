#!/bin/env python3
import sys
import os
import argparse

# Add membrain-seg to Python path
MEMBRAIN_SEG_PATH = "/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/source/membrain-seg/src"
if MEMBRAIN_SEG_PATH not in sys.path:
    sys.path.insert(0, MEMBRAIN_SEG_PATH)

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.ndimage import label
from skimage.measure import regionprops
from tqdm import tqdm

from membrain_seg.benchmark.metrics import masked_surface_dice


class SoftSkeletonize(torch.nn.Module):
    def __init__(self, num_iter: int = 5):
        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, input_: torch.Tensor):
        if len(input_.shape) == 4:
            p1 = -F.max_pool2d(-input_, (3, 1), (1, 1), (1, 0))
            p2 = -F.max_pool2d(-input_, (1, 3), (1, 1), (0, 1))
            return torch.min(p1, p2)
        elif len(input_.shape) == 5:
            p1 = -F.max_pool3d(-input_, (3, 1, 1), (1, 1, 1), (1, 0, 0))
            p2 = -F.max_pool3d(-input_, (1, 3, 1), (1, 1, 1), (0, 1, 0))
            p3 = -F.max_pool3d(-input_, (1, 1, 3), (1, 1, 1), (0, 0, 1))
            return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, input_: torch.Tensor):
        if len(input_.shape) == 4:
            return F.max_pool2d(input_, (3, 3), (1, 1), (1, 1))
        elif len(input_.shape) == 5:
            return F.max_pool3d(input_, (3, 3, 3), (1, 1, 1), (1, 1, 1))

    def soft_open(self, input_: torch.Tensor):
        return self.soft_dilate(self.soft_erode(input_))

    def soft_skel(self, input_: torch.Tensor):
        input1 = self.soft_open(input_)
        skel = F.relu(input1)
        for j in range(self.num_iter):
            input_ = self.soft_erode(input_)
            input1 = self.soft_open(input_)
            delta = F.relu(input_ - input1)
            skel = skel + F.relu(delta - skel * delta)
        return skel

    def forward(self, input_: torch.Tensor):
        return self.soft_skel(input_)

def skeletonize(arr):
    soft_skel = SoftSkeletonize(num_iter=5)
    t = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        skel = soft_skel(t)
    return skel.squeeze().numpy()


def load_segmentation(file_path, key):
    with h5py.File(file_path, "r") as f:
        data = f[key][:]
    return data


def evaluate_surface_dice(pred, gt, raw, check):
    gt_skeleton = skeletonize(gt.astype(np.float32))
    pred_skeleton = skeletonize(pred.astype(np.float32))
    mask = gt != 2

    if check:
        import napari
        v = napari.Viewer()
        v.add_image(gt_skeleton, name="gt_skeleton")
        napari.run()

    surf_dice, confusion_dict = masked_surface_dice(
        pred_skeleton, gt_skeleton, pred, gt, mask
    )
    return surf_dice, confusion_dict


def process_file(pred_path, gt_path, seg_key, gt_key, check,
                 min_bb_shape=(64, 384, 384), min_thinning_size=0,
                 global_eval=False):
    try:
        pred = load_segmentation(pred_path, seg_key)
        gt = load_segmentation(gt_path, gt_key)
        raw = load_segmentation(gt_path, "raw")

        if global_eval:
            gt_bin = (gt == 1).astype(np.uint8)
            pred_bin = pred.astype(np.uint8)

            dice, confusion = evaluate_surface_dice(pred_bin, gt_bin, raw, check)
            return [{
                "tomo_name": os.path.basename(pred_path),
                "gt_component_id": -1,
                "surface_dice": dice,
                **confusion
            }]

        labeled_gt, _ = label(gt == 1)
        props = regionprops(labeled_gt)
        results = []

        for prop in props:
            if prop.area < min_thinning_size:
                continue

            comp_id = prop.label
            bbox_start = prop.bbox[:3]
            bbox_end = prop.bbox[3:]
            bbox = tuple(slice(start, stop) for start, stop in zip(bbox_start, bbox_end))

            pad_width = [
                max(min_shape - (sl.stop - sl.start), 0) // 2
                for sl, min_shape in zip(bbox, min_bb_shape)
            ]

            expanded_bbox = tuple(
                slice(
                    max(sl.start - pw, 0),
                    min(sl.stop + pw, dim)
                )
                for sl, pw, dim in zip(bbox, pad_width, gt.shape)
            )

            gt_crop = (labeled_gt[expanded_bbox] == comp_id).astype(np.uint8)
            pred_crop = pred[expanded_bbox].astype(np.uint8)
            raw_crop = raw[expanded_bbox]

            try:
                dice, confusion = evaluate_surface_dice(pred_crop, gt_crop, raw_crop, check)
            except Exception as e:
                print(f"Error computing Dice for GT component {comp_id} in {pred_path}: {e}")
                continue

            result = {
                "tomo_name": os.path.basename(pred_path),
                "gt_component_id": comp_id,
                "surface_dice": dice,
                **confusion
            }
            results.append(result)

        return results

    except Exception as e:
        print(f"Error processing {pred_path}: {e}")
        return []


def collect_results(input_folder, gt_folder, model_name, check=False,
                    min_bb_shape=(64, 384, 384), min_thinning_size=0,
                    global_eval=False):
    results = []
    seg_key = f"/segmentations/{model_name}"
    gt_key = "/labels/actin"
    input_folder_name = os.path.basename(os.path.normpath(input_folder))

    for fname in tqdm(os.listdir(input_folder), desc="Processing segmentations"):
        if not fname.endswith(".h5"):
            continue

        pred_path = os.path.join(input_folder, fname)
        print(pred_path)
        gt_path = os.path.join(gt_folder, fname)

        if not os.path.exists(gt_path):
            print(f"Warning: Ground truth file not found for {fname}")
            continue

        file_results = process_file(
            pred_path, gt_path, seg_key, gt_key, check,
            min_bb_shape=min_bb_shape,
            min_thinning_size=min_thinning_size,
            global_eval=global_eval
        )

        for res in file_results:
            res["input_folder"] = input_folder_name
            results.append(res)

    return results


def save_results(results, output_file):
    new_df = pd.DataFrame(results)

    if os.path.exists(output_file):
        existing_df = pd.read_excel(output_file)
        combined_df = existing_df[
            ~existing_df.set_index(["tomo_name", "input_folder", "gt_component_id"]).index.isin(
                new_df.set_index(["tomo_name", "input_folder", "gt_component_id"]).index
            )
        ]
        final_df = pd.concat([combined_df, new_df], ignore_index=True)
    else:
        final_df = new_df

    final_df.to_excel(output_file, index=False)
    print(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute surface dice per GT component or globally for actin segmentations."
    )
    parser.add_argument("--input_folder", "-i", required=True, help="Folder with predicted segmentations (.h5)")
    parser.add_argument("--gt_folder", "-gt", required=True, help="Folder with ground truth segmentations (.h5)")
    parser.add_argument("--model_name", "-m", required=True, help="Model name string used in prediction key")
    parser.add_argument("--check", action="store_true", help="Visualize intermediate outputs in Napari")
    parser.add_argument("--global_eval", action="store_true", help="If set, compute global surface dice instead of per-component")

    args = parser.parse_args()

    min_bb_shape = (64, 384, 384)
    min_thinning_size = 0

    suffix = "global" if args.global_eval else "per_gt_component"

    output_file = f"./evaluation_results/{args.model_name}_surface_dice_{suffix}.xlsx"
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    results = collect_results(
        args.input_folder,
        args.gt_folder,
        args.model_name,
        args.check,
        min_bb_shape=min_bb_shape,
        min_thinning_size=min_thinning_size,
        global_eval=args.global_eval
    )

    save_results(results, output_file)

if __name__ == "__main__":
    main()
