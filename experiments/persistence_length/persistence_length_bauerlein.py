"""Persistence length by the Bäuerlein et al. (Cell 2017) method.

Reference: https://github.com/FJBauerlein/Huntington
Port for comparison with persistence_length.py. It uses the middle tangent as reference,
the unsigned acute angle (MATLAB subspace, so cos stays in [0, 1]), and a through-origin
log fit up to the P90 length cutoff.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--pattern", type=str, default="*_instances.csv")
    parser.add_argument("--sampling_dist", type=float, default=5.0)
    parser.add_argument("--min_length", type=float, default=350.0)
    parser.add_argument("--tag", type=str, default=None)
    return parser.parse_args()


def load_filaments(csv_path):
    df = pd.read_csv(csv_path)
    return [g[["X", "Y", "Z"]].to_numpy(dtype=np.float64) for _, g in df.groupby("ID")]


def resample(coords, sampling_dist):
    seg = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    s_new = np.arange(0.0, s[-1], sampling_dist)
    resampled = np.column_stack([np.interp(s_new, s, coords[:, k]) for k in range(3)])
    return resampled, float(s[-1])


def point_tangents(coords):
    t = np.gradient(coords, axis=0)
    return t / np.linalg.norm(t, axis=1, keepdims=True)


def cos_profile(filaments, sampling_dist, min_length):
    lengths = []
    by_offset = {}
    for coords in filaments:
        resampled, length = resample(coords, sampling_dist)
        if length < min_length or len(resampled) < 3:
            continue
        lengths.append(length)
        tangents = point_tangents(resampled)
        mid = len(tangents) // 2
        v1 = tangents[mid]
        for i in range(len(tangents)):
            offset = abs(i - mid)
            by_offset.setdefault(offset, []).append(abs(float(np.dot(v1, tangents[i]))))

    if not by_offset:
        return np.array([]), np.array([])
    max_offset = max(by_offset)
    cos_av = np.array([np.mean(by_offset[d]) if d in by_offset else np.nan
                       for d in range(max_offset + 1)])
    return cos_av, np.array(lengths)


def fit_through_origin(s, y):
    slope = np.sum(s * y) / np.sum(s * s)
    residual = y - slope * s
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - np.sum(residual ** 2) / ss_tot if ss_tot > 0 else np.nan
    return slope, r2


def plot_fits(curves, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for sample, lp, s, y, slope in curves:
        label = sample if not np.isfinite(lp) else f"{sample}  Lp={lp / 10000.0:.2f} µm"
        line = ax.plot(s / 10.0, y, ".", ms=3, label=label)[0]
        ax.plot(s / 10.0, slope * s, "-", color=line.get_color())
    ax.set_xlabel("distance [nm]")
    ax.set_ylabel("log <cos θ>")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_length_distribution(lengths_by_sample, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for sample, lengths in lengths_by_sample.items():
        ax.hist(np.asarray(lengths) / 10.0, bins=60, histtype="step",
                label=f"{sample} (n={len(lengths)})")
    ax.set_xlabel("filament length [nm]")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    data_paths = sorted(Path(args.data_dir).glob(args.pattern))
    rows = []
    curves = []
    lengths_by_sample = {}
    for data_path in data_paths:
        cos_av, lengths = cos_profile(load_filaments(data_path), args.sampling_dist, args.min_length)
        if lengths.size == 0:
            print(f"{data_path.name}: no filaments above {args.min_length:.0f} A.")
            continue
        lengths_by_sample[data_path.stem] = lengths

        p90 = float(np.percentile(lengths, 90))
        s = np.arange(len(cos_av)) * args.sampling_dist
        valid = np.isfinite(cos_av) & (cos_av > 0.0) & (s <= p90)
        if valid.sum() < 2:
            continue
        y = np.log(cos_av[valid])
        slope, r2 = fit_through_origin(s[valid], y)
        lp = -1.0 / slope if slope < 0.0 else np.nan
        curves.append((data_path.stem, lp, s[valid], y, slope))

        rows.append({
            "sample": data_path.stem,
            "n_filaments": len(lengths),
            "mean_length": float(np.mean(lengths)),
            "p90_length": p90,
            "persistence_length": lp,
            "r_squared": r2,
        })
        lp_um = lp / 10000.0 if np.isfinite(lp) else np.nan
        print(f"{data_path.name}: {len(lengths)} filaments, Lp = {lp_um:.2f} um, R^2 = {r2:.3f}")

    suffix = f"_{args.tag}" if args.tag else ""
    if lengths_by_sample:
        plot_length_distribution(lengths_by_sample, results_dir / f"length_distribution_bauerlein{suffix}.png")
    if not rows:
        print("No samples fit.")
        return

    with open(results_dir / f"persistence_length_bauerlein{suffix}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    plot_fits(curves, results_dir / f"persistence_length_bauerlein{suffix}.png")


if __name__ == "__main__":
    main()
