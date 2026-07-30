#!/usr/bin/env python
"""Raw TEASAR skeleton quality on one binary filament mask, before and after tick removal.

Skeletonises the mask, then reports how many degree-3 nodes are plausible filament crossings rather than
skeletonisation spurs, judged by whether all three arms exceed --spur_length. No other graph cleaning is
applied: no degree-3 or degree-4 splitting and no joining, so the numbers describe what the skeletoniser
plus tick removal produce on their own.

Run in the synapse-net environment, which is where bioimage_cpp lives:
    micromamba run -n synapse-net python example_teasar_diagnostics.py --mask_path <mask.mrc>

Add --crop to work on the centred --crop_size box instead of the full volume.
"""
import argparse

import numpy as np
import mrcfile

from bioimage_cpp.skeleton import remove_ticks, teasar

from skeleton_stats import center_crop, print_rows, skeleton_stats, write_csv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mask_path", type=str, required=True, help="Binary mask as an .mrc volume.")
    parser.add_argument("--pixel_size", type=float, default=10.0, help="Voxel spacing in A passed to teasar.")
    parser.add_argument("--scale", type=float, default=0.0, help="teasar invalidation-radius scale.")
    parser.add_argument("--constant", type=float, default=70.0, help="teasar invalidation radius in A.")
    parser.add_argument("--number_of_threads", type=int, default=8)
    parser.add_argument("--tick_length", type=float, default=200.0,
                        help="Max dead-end branch length in A that remove_ticks prunes.")
    parser.add_argument("--spur_length", type=float, default=200.0,
                        help="An arm at or below this length in A is a spur, not a real branch.")
    parser.add_argument("--crop", action="store_true", help="Use the centred crop instead of the full volume.")
    parser.add_argument("--crop_size", type=int, default=200, help="Edge length of the centred crop in voxels.")
    parser.add_argument("--out_csv", type=str, default=None, help="Optional path for the measurements CSV.")
    return parser.parse_args()


def read_mask(mask_path):
    with mrcfile.open(mask_path, permissive=True) as mrc:
        return np.asarray(mrc.data) > 0


def main():
    args = parse_args()
    mask = read_mask(args.mask_path)
    volume = "full"
    if args.crop:
        origin = center_crop(mask.shape, args.crop_size)
        mask = mask[tuple(slice(o, o + args.crop_size) for o in origin)]
        volume = f"crop{args.crop_size}"
    foreground = int(mask.sum())
    print(f"{args.mask_path}: {volume}, shape {mask.shape}, foreground {foreground} voxels, "
          f"{args.pixel_size:.1f} A")

    spacing = (args.pixel_size,) * 3
    vertices, edges, radii = teasar(mask.astype(np.uint8), spacing=spacing, scale=args.scale,
                                    constant=args.constant, number_of_threads=args.number_of_threads)
    ticked_vertices, ticked_edges, _ = remove_ticks(vertices, edges, args.tick_length, radii=radii)

    rows = []
    for stage, verts, edgs in (("raw", vertices, edges),
                               (f"remove_ticks {args.tick_length:.0f} A", ticked_vertices, ticked_edges)):
        row = skeleton_stats(verts, edgs, args.spur_length)
        row.update(method="teasar", volume=volume, stage=stage, foreground=foreground)
        rows.append(row)

    print_rows(rows, args.spur_length)
    if args.out_csv:
        write_csv(rows, args.out_csv)
        print(f"\n-> {args.out_csv}")


if __name__ == "__main__":
    main()
