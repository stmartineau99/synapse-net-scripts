"""Flag filament instances that were not split at degree-3 or degree-4 nodes.

An instance whose skeleton graph has a node of degree >= 3 is a branch (3) or crossing (4)
that clean_filament_graph did not split, merging two filaments under one ID. Report the
graph degree, the junction turning angle, and the spatial gap, then write clean CSVs with
the suspects removed so the persistence-length fit can rerun without them.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--pattern", type=str, default="*/*_instances.csv")
    parser.add_argument("--min_length", type=float, default=350.0)
    parser.add_argument("--tangent_window", type=int, default=5)
    parser.add_argument("--gap_ratio", type=float, default=5.0)
    parser.add_argument("--angle_deg", type=float, default=60.0)
    parser.add_argument("--tag", type=str, default=None)
    return parser.parse_args()


def load_ordered(csv_path):
    df = pd.read_csv(csv_path)
    return [(int(fid), g[["X", "Y", "Z"]].to_numpy(np.float64)) for fid, g in df.groupby("ID")]


def graph_max_degree(npz_path):
    if not npz_path.exists():
        return None
    data = np.load(npz_path)
    labels = data["labels"]
    degree = np.zeros(len(labels), dtype=np.int64)
    for a, b in data["edges"]:
        degree[int(a)] += 1
        degree[int(b)] += 1
    max_by_label = {}
    for idx, label in enumerate(labels):
        label = int(label)
        max_by_label[label] = max(max_by_label.get(label, 0), int(degree[idx]))
    return max_by_label


def filament_length(coords):
    return float(np.linalg.norm(np.diff(coords, axis=0), axis=1).sum())


def smoothed_tangents(coords, window):
    t = np.gradient(coords, axis=0)
    if window > 1 and len(coords) >= window:
        kernel = np.ones(window) / window
        t = np.column_stack([np.convolve(t[:, d], kernel, mode="same") for d in range(3)])
    norms = np.linalg.norm(t, axis=1, keepdims=True)
    return t / np.where(norms == 0.0, 1.0, norms)


def geometry(coords, window):
    steps = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    median_step = float(np.median(steps)) if steps.size else 0.0
    gap_ratio = float(steps.max() / median_step) if median_step > 0 else np.nan
    tangents = smoothed_tangents(coords, window)
    dots = np.clip(np.einsum("ij,ij->i", tangents[:-1], tangents[1:]), -1.0, 1.0)
    max_angle = float(np.degrees(np.arccos(dots)).max()) if len(dots) else 0.0
    return gap_ratio, max_angle


def summarize(name, values):
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return f"{name} none"
    return f"{name} median={np.median(v):.1f} p90={np.percentile(v, 90):.1f} max={v.max():.1f}"


def write_clean_csv(csv_path, filaments):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "X", "Y", "Z"])
        for fid, coords in filaments:
            for x, y, z in coords:
                writer.writerow([fid, float(x), float(y), float(z)])


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_rows = []
    for csv_path in sorted(Path(args.data_dir).glob(args.pattern)):
        tomogram = csv_path.stem.replace("_instances", "")
        max_degree = graph_max_degree(csv_path.with_suffix(".npz"))

        kept, clean, tomo_rows = [], [], []
        for fid, coords in load_ordered(csv_path):
            if len(coords) < 2 or filament_length(coords) < args.min_length:
                continue
            gap_ratio, max_angle = geometry(coords, args.tangent_window)
            degree = max_degree.get(fid) if max_degree is not None else None
            if degree is not None:
                suspect = degree >= 3
            else:
                suspect = (np.isfinite(gap_ratio) and gap_ratio > args.gap_ratio) or max_angle > args.angle_deg
            row = {
                "tomogram": tomogram, "id": fid, "length": filament_length(coords),
                "max_degree": degree if degree is not None else -1,
                "gap_ratio": gap_ratio, "max_angle": max_angle, "suspect": int(suspect),
            }
            tomo_rows.append(row)
            kept.append((fid, coords))
            if not suspect:
                clean.append((fid, coords))

        report_rows.extend(tomo_rows)
        n, n_clean = len(kept), len(clean)
        n_suspect = n - n_clean
        degrees = [r["max_degree"] for r in tomo_rows]
        n_d3 = sum(d == 3 for d in degrees)
        n_d4 = sum(d >= 4 for d in degrees)
        susp_angles = [r["max_angle"] for r in tomo_rows if r["suspect"]]
        susp_gaps = [r["gap_ratio"] for r in tomo_rows if r["suspect"]]
        print(f"{tomogram}: {n} filaments, {n_suspect} suspect "
              f"({100 * n_suspect / max(n, 1):.0f}%), degree3={n_d3} degree>=4={n_d4}")
        print(f"  suspect {summarize('angle[deg]', susp_angles)} | {summarize('gap_ratio', susp_gaps)}")

        tomo_dir = out_dir / tomogram
        tomo_dir.mkdir(parents=True, exist_ok=True)
        write_clean_csv(tomo_dir / f"{tomogram}_instances.csv", clean)

    suffix = f"_{args.tag}" if args.tag else ""
    with open(out_dir / f"merged_report{suffix}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
        writer.writeheader()
        writer.writerows(report_rows)
    print(f"clean CSVs and merged_report{suffix}.csv written to {out_dir}")


if __name__ == "__main__":
    main()
