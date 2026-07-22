import argparse
import re
from pathlib import Path

import numpy as np
import tifffile

TARGET = 3708  # square edge matching the cryoet-portal aligned tilt series (_sq alignment)


def square_movie(movie):
    if movie.shape[-2] < movie.shape[-1]:
        movie = np.rot90(movie, k=3, axes=(-2, -1))
    y, x = movie.shape[-2:]
    y0, x0 = (y - TARGET) // 2, (x - TARGET) // 2
    return movie[..., y0:y0 + TARGET, x0:x0 + TARGET]


def process_run(run_dir, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for tif_path in sorted(run_dir.glob("*.tif")):
        tifffile.imwrite(out_dir / tif_path.name, square_movie(tifffile.imread(tif_path)))
    for mdoc_path in run_dir.glob("*.mdoc"):
        text = re.sub(r"ImageSize = .*", f"ImageSize = {TARGET} {TARGET}", mdoc_path.read_text(), count=1)
        (out_dir / mdoc_path.name).write_text(text)


def main():
    parser = argparse.ArgumentParser(
        description="Rotate landscape movie frames to portrait and centre-crop to a square, matching the cryoet-portal aligned tilt series.")
    parser.add_argument("--frames_dir", type=Path, required=True, help="Directory with <run>/ subdirs of raw movie tifs + mdoc.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory to write squared frames to (<run>/ subdirs).")
    parser.add_argument("--runs", nargs="+", default=None, help="Run subdir names to process. Defaults to all subdirs of frames_dir.")
    args = parser.parse_args()

    runs = args.runs or sorted(d.name for d in args.frames_dir.iterdir() if d.is_dir())
    for run in runs:
        process_run(args.frames_dir / run, args.output_dir / run)
        print(f"{run}: squared frames -> {args.output_dir / run}")


if __name__ == "__main__":
    main()
