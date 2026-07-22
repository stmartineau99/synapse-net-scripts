import argparse
import csv
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

from bioimage_cpp.graph import connected_components
from bioimage_cpp.skeleton import clean_filament_graph, skeleton_to_graph, teasar


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--data_paths", type=str, nargs="+", default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seg_key", type=str, required=True)
    parser.add_argument("--pixel_size", type=float, required=True)
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--constant", type=float, default=30)
    parser.add_argument("--number_of_threads", type=int, default=8)
    parser.add_argument("--direction_span", type=int, default=10)
    parser.add_argument("--min_through_angle", type=float, default=170.0)
    parser.add_argument("--min_branch_angle", type=float, default=30.0)
    parser.add_argument("--tick_length", type=float, default=50.0)
    parser.add_argument("--join_dist", type=float, default=50.0)
    parser.add_argument("--min_join_angle", type=float, default=175.0)
    return parser.parse_args()


def skeletonize(mask, pixel_size, scale, constant, number_of_threads):
    mask = (mask > 0).astype(np.uint8)
    spacing = (pixel_size, pixel_size, pixel_size)
    return teasar(
        mask,
        spacing=spacing,
        scale=scale,
        constant=constant,
        number_of_threads=number_of_threads,
    )


def order_component(component, adjacency):
    endpoints = [n for n in component if len(adjacency[n]) == 1]
    start = endpoints[0] if endpoints else component[0]
    order = [start]
    visited = {start}
    current = start
    while True:
        nexts = [n for n in adjacency[current] if n not in visited]
        if not nexts:
            break
        current = nexts[0]
        visited.add(current)
        order.append(current)
    order.extend(n for n in component if n not in visited)
    return order


def instance_rows(vertices, edges, labels):
    adjacency = defaultdict(list)
    for a, b in edges:
        adjacency[int(a)].append(int(b))
        adjacency[int(b)].append(int(a))

    nodes_by_label = defaultdict(list)
    for index, label in enumerate(labels):
        nodes_by_label[int(label)].append(index)

    rows = []
    for label in sorted(nodes_by_label):
        for index in order_component(nodes_by_label[label], adjacency):
            x, y, z = vertices[index]
            rows.append((label, float(x), float(y), float(z)))
    return rows


def write_csv(csv_path, rows):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "X", "Y", "Z"])
        writer.writerows(rows)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.data_dir:
        data_paths = sorted(Path(args.data_dir).glob("*.h5"))
    elif args.data_paths:
        data_paths = [Path(p) for p in args.data_paths]
    else:
        data_paths = []
    if not data_paths:
        raise FileNotFoundError("No h5 files found.")

    for data_path in data_paths:
        tomogram = data_path.stem
        with h5py.File(data_path, "r") as f:
            if args.seg_key not in f:
                print(f"{tomogram}: '{args.seg_key}' not found, skipping.")
                continue
            mask = f[args.seg_key][:]

        print(f"{tomogram}: running teasar skeletonization.")
        raw_vertices, raw_edges, raw_radii = skeletonize(
            mask, args.pixel_size, args.scale, args.constant, args.number_of_threads
        )
        vertices, edges, radii = clean_filament_graph(
            raw_vertices,
            raw_edges,
            radii=raw_radii,
            direction_span=args.direction_span,
            min_through_angle=args.min_through_angle,
            min_branch_angle=args.min_branch_angle,
            tick_length=args.tick_length,
            join_dist=args.join_dist,
            min_join_angle=args.min_join_angle,
        )
        labels = connected_components(skeleton_to_graph(vertices, edges))

        tomo_dir = output_dir / tomogram
        tomo_dir.mkdir(parents=True, exist_ok=True)

        npz_kwargs = dict(vertices=vertices, edges=edges, labels=labels)
        if radii is not None:
            npz_kwargs["radii"] = radii
        np.savez_compressed(tomo_dir / f"{tomogram}_instances.npz", **npz_kwargs)

        write_csv(tomo_dir / f"{tomogram}_instances.csv", instance_rows(vertices, edges, labels))

        print(f"{tomogram}: {len(np.unique(labels))} instances -> {tomo_dir}")

    print(f"Instances saved in {output_dir}")


if __name__ == "__main__":
    main()
