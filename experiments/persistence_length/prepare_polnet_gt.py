"""Pool PolNet ground-truth actin filaments into one ID,X,Y,Z CSV.

Read the per-tomo motif lists in a PolNet simulation_dir, keep the actin rows
(Code == --code), group each tomo's points by Polymer into filaments, and write one pooled
CSV. Coordinates are already in Angstrom. The output is intermediate; the default out_dir is
under /tmp.
"""

import argparse
import csv
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="/tmp/polnet_gt")
    parser.add_argument("--code", type=str, default="tube")
    parser.add_argument("--tag", type=str, default="polnet")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    motif_paths = sorted((Path(args.sim_dir) / "motif_lists").glob("tomo_motif_list_*.csv"))
    if not motif_paths:
        raise FileNotFoundError(f"no motif lists in {args.sim_dir}/motif_lists")

    rows = []
    next_id = 0
    for motif_path in motif_paths:
        df = pd.read_csv(motif_path, sep="\t")
        actin = df[df["Code"] == args.code]
        for _, group in actin.groupby("Polymer", sort=True):
            for x, y, z in group[["X", "Y", "Z"]].to_numpy(dtype=float):
                rows.append((next_id, x, y, z))
            next_id += 1

    out_path = out_dir / f"{args.tag}_instances.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "X", "Y", "Z"])
        writer.writerows(rows)
    print(f"{next_id} filaments from {len(motif_paths)} tomos -> {out_path}")


if __name__ == "__main__":
    main()
