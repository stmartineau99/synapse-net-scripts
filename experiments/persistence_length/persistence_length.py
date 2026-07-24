"""Measure the persistence length of filaments from instance CSVs.

Each input CSV is one sample, with columns ID, X, Y, Z and coordinates in Angstrom. Rows that share an ID form one
filament. For each sample the script resamples every filament at a sampling distance, computes the tangent
autocorrelation, and fits log<cos θ> against contour distance to get the persistence length Lp. It writes a results CSV,
a log-linear fit plot, and a filament-length histogram to the results directory.

Usage:
    python persistence_length.py --data_dir <path/to/data_dir> --results_dir <path/to/results_dir> [options]

Based on the method of Bäuerlein et al. (Cell 2017).
Reference implementation: https://github.com/FJBauerlein/Huntington
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory of per-sample instance CSVs with coordinates in Angstrom (ID, X, Y, Z).")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Output directory for the results CSV and plots.")
    parser.add_argument("--pattern", type=str, default="*_instances.csv",
                        help="Glob pattern for input CSV files in data_dir.")
    parser.add_argument("--sampling_dist", type=float, default=5.0,
                        help="Filament resampling distance in A.")
    parser.add_argument("--min_length", type=float, default=350.0,
                        help="Minimum filament length in A to include.")
    parser.add_argument("--tangent_window", type=int, default=5,
                        help="Number of adjacent points to average per tangent.")
    return parser.parse_args()

def load_filaments(csv_path):
    """Return a list of (N, 3) coordinate arrays, one per filament ID in the CSV."""
    df = pd.read_csv(csv_path)

    filaments = []
    for fid, group in df.groupby("ID"):
        coords = group[["X", "Y", "Z"]].to_numpy(dtype=np.float64)

        filaments.append(coords)

    return filaments

def resample_filaments(filaments, sampling_dist, min_length):
    resampled = []
    filament_lengths = []
    for coords in filaments:
        segment_lengths = np.linalg.norm(np.diff(coords, axis=0), axis=1)
        s = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        if s[-1] < min_length:
            continue
        filament_lengths.append(s[-1])
        
        s_new = np.arange(0, s[-1], sampling_dist)

        x_new = np.interp(s_new, s, coords[:, 0])
        y_new = np.interp(s_new, s, coords[:, 1])
        z_new = np.interp(s_new, s, coords[:, 2])

        resampled.append(np.vstack([x_new, y_new, z_new]).T)

    return resampled, np.array(filament_lengths)

def unit_tangents(coords, window):
    """Unit tangent per point, averaged over `window` adjacent points to reduce aliasing."""
    t = np.gradient(coords, axis=0)
    if window > 1:
        kernel = np.ones(window) / window
        t = np.column_stack([np.convolve(t[:, d], kernel, mode="same") for d in range(t.shape[1])])
    norms = np.linalg.norm(t, axis=1, keepdims=True)
    return t / np.where(norms == 0.0, 1.0, norms)


def tangent_autocorr(filaments, max_lag=None, window=5):
    tangents = [unit_tangents(c, window) for c in filaments if len(c) >= 2]

    if max_lag is None:
        max_lag = max(len(t) - 1 for t in tangents)

    sums = np.zeros(max_lag + 1)
    counts = np.zeros(max_lag + 1)
    for t in tangents:
        for k in range(min(max_lag, len(t) - 1) + 1):
            dots = np.einsum("ij,ij->i", t[: len(t) - k], t[k:])
            sums[k] += dots.sum()
            counts[k] += dots.size

    corr = np.full(max_lag + 1, np.nan)
    valid = counts > 0
    corr[valid] = sums[valid] / counts[valid]

    return corr

def fit_persistence_length(corr, sampling_dist):
    """Fit log<cos θ> = ln(A) - s / Lp with a free intercept. Return (Lp in A, R², curve);
    Lp NaN if degenerate. curve is (s, log<cos θ>, fitted line) for plotting."""
    lags = np.arange(len(corr))
    s = lags * sampling_dist
    valid = np.isfinite(corr) & (corr > 0.0)
    s = s[valid]
    y = np.log(corr[valid])
    if s.size < 2:
        return np.nan, np.nan, None

    slope, intercept = np.polyfit(s, y, 1)
    yhat = slope * s + intercept
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - np.sum((y - yhat) ** 2) / ss_tot if ss_tot > 0 else np.nan
    lp = -1.0 / slope if slope < 0.0 else np.nan
    return lp, r2, (s, y, yhat)


def plot_fits(curves, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for sample, s, y, yhat in curves:
        line = ax.plot(s / 10.0, y, ".", ms=3, label=sample)[0]
        ax.plot(s / 10.0, yhat, "-", color=line.get_color())
    ax.set_xlabel("distance [nm]")
    ax.set_ylabel("log <cos θ>")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def filament_length(coords):
    if len(coords) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(coords, axis=0), axis=1).sum())


def plot_length_distribution(lengths_by_sample, out_path):
    fig, ax = plt.subplots(figsize=(7, 4))
    for sample, lengths in lengths_by_sample.items():
        lengths_nm = np.asarray(lengths, dtype=np.float64) / 10.0
        ax.hist(lengths_nm, bins=60, histtype="step",
                label=f"{sample} (n={len(lengths_nm)})")
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

    data_dir = Path(args.data_dir)
    data_paths = sorted(data_dir.glob(args.pattern))
    rows = []
    curves = []
    lengths_by_sample = {}
    for data_path in data_paths:
        filaments = load_filaments(data_path)
        lengths_by_sample[data_path.stem] = [filament_length(c) for c in filaments]

        filaments, lengths = resample_filaments(filaments, args.sampling_dist, args.min_length)
        if len(filaments) == 0:
            print(f"{data_path.name}: no filaments above the min length of {args.min_length:.0f} A.")
            continue

        p90 = float(np.percentile(lengths, 90))
        max_lag = int(p90 / args.sampling_dist)

        corr = tangent_autocorr(filaments, max_lag, args.tangent_window)
        lp, r2, curve = fit_persistence_length(corr, args.sampling_dist)
        if curve is not None:
            curves.append((data_path.stem, *curve))

        rows.append({
            "sample": data_path.stem,
            "n_filaments": len(filaments),
            "mean_length": float(np.mean(lengths)),
            "p90_length": p90,
            "persistence_length": lp,
            "r_squared": r2,
        })
        lp_um = lp / 10000.0 if np.isfinite(lp) else np.nan
        print(f"{data_path.name}: {len(filaments)} filaments, Lp = {lp_um:.2f} um, R^2 = {r2:.3f}")

    if lengths_by_sample:
        plot_length_distribution(lengths_by_sample, results_dir / "length_distribution.png")

    if len(rows) == 0:
        print("No samples with filaments above the min length.")
        return

    csv_path = results_dir / "persistence_length.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    plot_fits(curves, results_dir / "persistence_length.png")


if __name__ == "__main__":
    main()
