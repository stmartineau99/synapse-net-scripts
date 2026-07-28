#!/usr/bin/env python
"""Score saved skeletons with the same metrics used for the teasar sweep.

Runs in `synapse-net`. Reads a manifest written by `sweep_kimimaro.py`, loads each npz, and
applies `evaluate` from `sweep_skeleton_params.py`, so both methods are measured by identical
code.
"""
import csv
import argparse
from pathlib import Path

import numpy as np
import h5py

from sweep_skeleton_params import CROP_PATH, FIELDS, evaluate


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=str, nargs="+", required=True)
    parser.add_argument("--mask_path", type=str, default=CROP_PATH)
    parser.add_argument("--seg_key", type=str, default="labels/actin")
    parser.add_argument("--pixel_size", type=float, default=10.0)
    parser.add_argument("--direction_span", type=int, default=10)
    parser.add_argument("--min_branch_angle", type=float, default=20.0)
    parser.add_argument("--min_daughter_length", type=float, default=0.0)
    parser.add_argument("--tick_length", type=float, default=200.0)
    parser.add_argument("--join_dist", type=float, default=50.0)
    parser.add_argument("--spur_length", type=float, default=200.0)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    with h5py.File(args.mask_path, "r") as f:
        mask = np.asarray(f[args.seg_key][:]) > 0
    foreground = np.argwhere(mask).astype(np.float64) * args.pixel_size

    config_fields = ["scale", "constant", "fix_branching", "fill_holes", "max_paths"]
    fields = config_fields + [f for f in FIELDS if f not in ("scale", "constant")]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for manifest in args.manifest:
            for entry in csv.DictReader(open(manifest)):
                row = {key: entry.get(key) for key in config_fields}
                row["status"] = entry["status"]
                row["seconds"] = entry["seconds"]
                if entry["status"] != "ok":
                    writer.writerow(row)
                    continue
                data = np.load(entry["npz"])
                try:
                    row.update(evaluate(data["vertices"], data["edges"], foreground, args))
                except Exception as error:
                    row["status"] = f"{type(error).__name__}: {error}"[:120]
                writer.writerow(row)
                handle.flush()
                print(f"scale={row['scale']} const={row['constant']} "
                      f"fb={row['fix_branching']} fh={row['fill_holes']} "
                      f"mp={row['max_paths']} deg3={row.get('deg3')} "
                      f"inst>200={row.get('instances_gt200')} "
                      f"real={row.get('length_in_real_um')}um "
                      f"cov_max={row.get('cov_max')} {row['status']}", flush=True)
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
