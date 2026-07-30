#!/usr/bin/env python
"""Raw kimimaro skeleton quality on one binary filament mask, before and after tick removal.

The kimimaro counterpart of example_teasar_diagnostics.py, reporting the same measurements so the two can
be compared. Separate script because kimimaro and bioimage_cpp live in different environments.

Run in the tardis-em environment, which is where kimimaro lives:
    micromamba run -n tardis-em python example_kimimaro_diagnostics.py --mask_path <mask.mrc>

Add --crop to work on the centred --crop_size box instead of the full volume.

--constant defaults higher than the teasar script uses. `scale` and `const` mean the same thing in both
libraries, but kimimaro invalidates a sphere where teasar invalidates a cube, so the same radius does not
invalidate the same volume and the two need different values to behave comparably.
"""
import argparse

import numpy as np
import mrcfile

from skeleton_stats import center_crop, print_rows, skeleton_stats, write_csv


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mask_path", type=str, required=True, help="Binary mask as an .mrc volume.")
    parser.add_argument("--pixel_size", type=float, default=10.0, help="Voxel spacing in A passed as anisotropy.")
    parser.add_argument("--scale", type=float, default=0.0, help="kimimaro invalidation-radius scale.")
    parser.add_argument("--constant", type=float, default=140.0, help="kimimaro invalidation radius in A.")
    parser.add_argument("--tick_length", type=float, default=200.0,
                        help="Max dead-end branch length in A that postprocess prunes.")
    parser.add_argument("--spur_length", type=float, default=200.0,
                        help="An arm at or below this length in A is a spur, not a real branch.")
    parser.add_argument("--fix_branching", action="store_true",
                        help="Enable kimimaro's fix_branching. Off by default; see skeletonize().")
    parser.add_argument("--crop", action="store_true", help="Use the centred crop instead of the full volume.")
    parser.add_argument("--crop_size", type=int, default=200, help="Edge length of the centred crop in voxels.")
    parser.add_argument("--out_csv", type=str, default=None, help="Optional path for the measurements CSV.")
    return parser.parse_args()


def read_mask(mask_path):
    with mrcfile.open(mask_path, permissive=True) as mrc:
        return np.asarray(mrc.data) > 0


def skeletonize(mask, spacing, scale, constant, fix_branching):
    """Raw kimimaro skeleton.

    `fix_branching` defaults off, against kimimaro's own recommendation, because it is designed for trees. It
    zeroes the edge weights along each traced path, so every later path is routed to the nearest already
    traced path rather than along its own course. In a dense network of separate filaments that turns a
    contact into a branch. Measured on the 200 crop it adds junctions and removes real crossings, and it
    runs one Dijkstra per path instead of one per skeleton, which costs minutes rather than seconds on the
    full volume. See ../teasar/benchmark/README.md.
    """
    import kimimaro
    params = {
        "scale": scale, "const": constant,
        "pdrf_scale": 100000, "pdrf_exponent": 4,
        "soma_detection_threshold": float("inf"),
        "soma_acceptance_threshold": float("inf"),
        "soma_invalidation_scale": 1.0, "soma_invalidation_const": 0.0,
    }
    result = kimimaro.skeletonize(
        mask.astype(np.uint32), teasar_params=params, anisotropy=spacing,
        object_ids=[1], dust_threshold=0, progress=False, fix_branching=fix_branching,
        fix_borders=False, fill_holes=False, parallel=1,
    )
    return result[1] if isinstance(result, dict) else result


def remove_ticks(skeleton, tick_length):
    """dust_threshold stays 0 so this prunes ticks only; the default would also drop small components."""
    import kimimaro
    return kimimaro.postprocess(skeleton, dust_threshold=0.0, tick_threshold=tick_length)


def as_graph(skeleton):
    return (np.asarray(skeleton.vertices, dtype=np.float64),
            np.asarray(skeleton.edges, dtype=np.int64))


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
          f"{args.pixel_size:.1f} A, invalidation radius {args.constant:.0f} A, "
          f"fix_branching {'on' if args.fix_branching else 'off'}")

    spacing = (args.pixel_size,) * 3
    skeleton = skeletonize(mask, spacing, args.scale, args.constant, args.fix_branching)
    ticked = remove_ticks(skeleton, args.tick_length)

    rows = []
    for stage, skel in (("raw", skeleton),
                        (f"remove_ticks {args.tick_length:.0f} A", ticked)):
        vertices, edges = as_graph(skel)
        row = skeleton_stats(vertices, edges, args.spur_length)
        row.update(method="kimimaro", volume=volume, stage=stage, foreground=foreground)
        rows.append(row)

    print_rows(rows, args.spur_length)
    if args.out_csv:
        write_csv(rows, args.out_csv)
        print(f"\n-> {args.out_csv}")


if __name__ == "__main__":
    main()
