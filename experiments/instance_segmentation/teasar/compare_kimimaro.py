#!/usr/bin/env python
"""Compare bioimage-cpp teasar against kimimaro on one crop, junction quality only.

Run once per method, in the matching environment:
    micromamba run -n synapse-net python compare_kimimaro.py --method teasar
    micromamba run -n tardis-em   python compare_kimimaro.py --method kimimaro
"""
import argparse

import numpy as np
import h5py

MASK_PATH = ("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/experimental/"
             "deepict/h5/00004.h5")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=("teasar", "kimimaro"), required=True)
    parser.add_argument("--mask_path", type=str, default=MASK_PATH)
    parser.add_argument("--seg_key", type=str, default="labels/actin")
    parser.add_argument("--crop", type=int, default=200)
    parser.add_argument("--upsample", type=int, default=1)
    parser.add_argument("--pixel_size", type=float, default=10.0)
    parser.add_argument("--scale", type=float, default=0.0)
    parser.add_argument("--constant", type=float, default=70.0)
    parser.add_argument("--spur_length", type=float, default=200.0,
                        help="An arm at or below this length (A) is a spur. A junction whose "
                             "arms all exceed it is a plausible crossing.")
    return parser.parse_args()


def center_crop(shape, size):
    """Origin of the centred box of the given size."""
    return tuple(max(0, (extent - size) // 2) for extent in shape)


def upsample(crop, factor):
    """Cubic-spline resample to a finer grid, then re-threshold."""
    from scipy.ndimage import zoom
    if factor == 1:
        return crop
    return zoom(crop.astype(np.float32), factor, order=3) >= 0.5


def run_teasar(crop, spacing, scale, constant):
    from bioimage_cpp.skeleton import teasar
    return teasar(crop.astype(np.uint8), spacing=spacing, scale=scale,
                  constant=constant, number_of_threads=8)[:2]


def run_kimimaro(crop, spacing, scale, constant):
    import kimimaro
    params = {
        "scale": scale, "const": constant,
        "pdrf_scale": 100000, "pdrf_exponent": 4,
        "soma_detection_threshold": float("inf"),
        "soma_acceptance_threshold": float("inf"),
        "soma_invalidation_scale": 1.0, "soma_invalidation_const": 0.0,
    }
    result = kimimaro.skeletonize(
        crop.astype(np.uint32), teasar_params=params, anisotropy=spacing,
        object_ids=[1], dust_threshold=0, progress=False, fix_branching=True,
        fix_borders=False, fill_holes=False, parallel=1,
    )
    skeleton = result[1] if isinstance(result, dict) else result
    return np.asarray(skeleton.vertices, dtype=np.float64), np.asarray(skeleton.edges, dtype=np.int64)


def build_adjacency(n_nodes, edges):
    src = np.concatenate([edges[:, 0], edges[:, 1]])
    dst = np.concatenate([edges[:, 1], edges[:, 0]])
    order = np.argsort(src, kind="stable")
    dst = dst[order]
    degrees = np.bincount(src, minlength=n_nodes)
    indptr = np.zeros(n_nodes + 1, dtype=np.int64)
    np.cumsum(degrees, out=indptr[1:])
    return indptr, dst, degrees


def walk_arm(node, first, indptr, dst, degrees, vertices):
    prev, cur = node, int(first)
    length = float(np.linalg.norm(vertices[cur] - vertices[node]))
    while degrees[cur] == 2:
        s, e = indptr[cur], indptr[cur + 1]
        nbrs = dst[s:e]
        nxt = int(nbrs[0]) if int(nbrs[0]) != prev else int(nbrs[1])
        length += float(np.linalg.norm(vertices[nxt] - vertices[cur]))
        prev, cur = cur, nxt
    return cur, length


def report(vertices, edges, args):
    n = len(vertices)
    indptr, dst, degrees = build_adjacency(n, edges)
    seg = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    deg3 = np.where(degrees == 3)[0]
    print(f"vertices {n}, edges {len(edges)}, length {seg.sum() / 1e4:.1f} um")
    print(f"degree 1 {int((degrees == 1).sum())}, degree 3 {len(deg3)}, "
          f"degree 4 {int((degrees == 4).sum())}")
    if not len(deg3):
        print("no degree-3 nodes")
        return
    print(f"degree-3 per 100 um of skeleton: {100 * len(deg3) / (seg.sum() / 1e4):.1f}")

    shortest, all_long, has_spur = [], 0, 0
    for node in deg3:
        arms = [walk_arm(int(node), dst[k], indptr, dst, degrees, vertices)
                for k in range(indptr[node], indptr[node + 1])]
        lengths = [a[1] for a in arms]
        shortest.append(min(lengths))
        if min(lengths) > args.spur_length:
            all_long += 1
        if any(degrees[end] == 1 and L <= args.spur_length for end, L in arms):
            has_spur += 1
    shortest = np.array(shortest)
    print("shortest arm per junction:")
    for q in (25, 50, 75, 90):
        print(f"  p{q}: {np.percentile(shortest, q):7.1f} A")
    print(f"all arms > {args.spur_length:.0f} A, plausible crossing: "
          f"{all_long} ({100 * all_long / len(deg3):.1f}%)")
    print(f"has dead-end spur <= {args.spur_length:.0f} A: "
          f"{has_spur} ({100 * has_spur / len(deg3):.1f}%)")


def main():
    args = parse_args()
    with h5py.File(args.mask_path, "r") as f:
        mask = np.asarray(f[args.seg_key][:]) > 0
    origin = center_crop(mask.shape, args.crop)
    slices = tuple(slice(o, o + args.crop) for o in origin)
    crop = mask[slices]
    print(f"method {args.method}, crop origin {origin} size {crop.shape}, "
          f"foreground {int(crop.sum())}")

    if args.upsample != 1:
        expected = int(crop.sum()) * args.upsample ** 3
        crop = upsample(crop, args.upsample)
        print(f"upsampled {args.upsample}x cubic: shape {crop.shape} "
              f"foreground {int(crop.sum())} "
              f"({100 * int(crop.sum()) / expected:.1f}% of volume-preserving)")

    spacing = (args.pixel_size / args.upsample,) * 3
    runner = run_teasar if args.method == "teasar" else run_kimimaro
    vertices, edges = runner(crop, spacing, args.scale, args.constant)
    report(vertices, np.asarray(edges, dtype=np.int64), args)


if __name__ == "__main__":
    main()
