#!/usr/bin/env python
"""Export a binary mask from an HDF5 volume to a standalone MRC file.

The deepict masks live inside the per-tomogram h5 files, so they have to be written out before they can
be handed to anyone working outside this project. The MRC voxel size is set from --pixel_size, so the
spacing travels with the file and the diagnostics scripts do not have to be told it again.

Usage:
    python export_actin_mask.py --out_path <path/to/00004_actin_mask.mrc>
"""
import argparse
from pathlib import Path

import h5py
import numpy as np
import mrcfile

BASE = "/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data"
MASK_PATH = f"{BASE}/experimental/deepict/h5/00004.h5"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mask_path", type=str, default=MASK_PATH, help="Source .h5 volume.")
    parser.add_argument("--seg_key", type=str, default="labels/actin", help="HDF5 key of the binary mask.")
    parser.add_argument("--out_path", type=str, required=True, help="Destination .mrc file.")
    parser.add_argument("--pixel_size", type=float, default=10.0, help="Voxel size in A for the MRC header.")
    return parser.parse_args()


def load_mask(mask_path, seg_key):
    with h5py.File(mask_path, "r") as f:
        if seg_key not in f:
            raise KeyError(f"'{seg_key}' not in {mask_path}")
        return np.asarray(f[seg_key][:])


def write_mrc(mask, out_path, pixel_size):
    binary = (mask > 0).astype(np.uint8)
    with mrcfile.new(out_path, overwrite=True) as mrc:
        mrc.set_data(binary)
        mrc.voxel_size = pixel_size
    return binary


def main():
    args = parse_args()
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mask = load_mask(args.mask_path, args.seg_key)
    binary = write_mrc(mask, out_path, args.pixel_size)
    print(f"{args.seg_key} of {Path(args.mask_path).name}: shape {binary.shape}, "
          f"foreground {int(binary.sum())} voxels, {args.pixel_size:.1f} A -> {out_path}")


if __name__ == "__main__":
    main()
