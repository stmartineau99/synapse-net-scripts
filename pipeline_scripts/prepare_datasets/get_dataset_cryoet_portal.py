import argparse
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

from cryoet_data_portal import Client, Dataset

PORTAL_URL = "https://files.cryoetdataportal.cziscience.com"
IMOD_CMD_NEWSTACK = "newstack"
IMOD_CMD_TILT = "tilt"
IMOD_CMD_TRIMVOL = "trimvol"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a dataset from the CryoET Data Portal and reconstruct tomograms at a given binning with IMOD. Optionally download frames for half map reconstruction.")
    parser.add_argument("dataset_id", type=int, help="CryoET Data Portal dataset ID.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Directory for downloaded data.")
    parser.add_argument("--binning", type=int, default=1, help="Binning factor for reconstruction.")
    parser.add_argument("--runs", nargs="+", default=None,
                        help="Run names to process; defaults to all runs in the dataset.")
    parser.add_argument("--download_tiltseries", action="store_true",
                        help="Download tilt series and alignment data.")
    parser.add_argument("--download_frames", action="store_true", default=None,
                        help="Download raw movie frames, gain reference, and mdoc.")
    parser.add_argument("--reconstruct", action="store_true",
                        help="Reconstruct tomograms from downloaded tilt series with IMOD.")
    parser.add_argument("--tomogram_dir", type=Path, default=None,
                        help="Copy reconstructed tomograms to this directory.")
    return parser.parse_args()


def _download(url, dest_path):
    if not dest_path.exists():
        urllib.request.urlretrieve(url, dest_path)


def _run_imod(cmd, log_path):
    if shutil.which(cmd[0]) is None:
        raise RuntimeError(f"IMOD binary {cmd[0]!r} not found on PATH.")
    with open(log_path, "a") as log_file:
        subprocess.run(cmd, stdout=log_file, stderr=log_file, check=True)


def download_dataset(dataset_id, output_dir, runs=None, tiltseries=True, frames=False):
    client = Client()
    dataset = Dataset.get_by_id(client, dataset_id)
    run_dirs = []
    for run in dataset.runs:
        if runs is not None and run.name not in runs:
            continue
        if tiltseries:
            run_dir = output_dir / "tiltseries" / run.name
            run_dir.mkdir(parents=True, exist_ok=True)
            for tilt_series in run.tiltseries:
                tilt_series.download_mrcfile(dest_path=str(run_dir))
                for alignment in tilt_series.alignments:
                    metadata_path = run_dir / "alignment_metadata.json"
                    _download(alignment.https_alignment_metadata, metadata_path)
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                    _download(f"{PORTAL_URL}/{metadata['tiltseries_path']}",
                              run_dir / "tiltseries_metadata.json")
                    for file_path in metadata["files"]:
                        _download(f"{PORTAL_URL}/{file_path}", run_dir / Path(file_path).name)
            run_dirs.append(run_dir)
        if frames:
            if not run.frames:
                print(f"No frames available for: {run.name}")
            else:
                frames_dir = output_dir / "frames" / run.name
                frames_dir.mkdir(parents=True, exist_ok=True)
                for frame in run.frames:
                    _download(frame.https_frame_path, frames_dir / Path(frame.https_frame_path).name)
                for gain_file in run.gain_files:
                    _download(gain_file.https_file_path, frames_dir / Path(gain_file.https_file_path).name)
                for acquisition_file in run.frame_acquisition_files:
                    _download(acquisition_file.https_mdoc_path, frames_dir / Path(acquisition_file.https_mdoc_path).name)
        print(f"Finished downloading: {run.name}")
    return run_dirs


def reconstruct_run(run_dir, binning):
    with open(run_dir / "alignment_metadata.json") as f:
        alignment = json.load(f)
    if alignment.get("format") != "IMOD":
        raise ValueError(f"{run_dir}: alignment format {alignment.get('format')!r} is not IMOD.")

    with open(run_dir / "tiltseries_metadata.json") as f:
        pixel_size = float(json.load(f)["pixel_spacing"])

    xf_path = run_dir / Path(alignment["alignment_path"]).name
    tlt_path = run_dir / Path(alignment["tilt_path"]).name
    raw_path = next(p for p in sorted(run_dir.glob("*.mrc"))
                    if p.name not in ("aligned.mrc", "tilt_output.mrc", "reconstructed.mrc"))

    thickness = round(alignment["volume_dimension"]["z"] / (pixel_size * binning))
    aligned_path = run_dir / "aligned.mrc"
    tilt_output_path = run_dir / "tilt_output.mrc"
    tomogram_path = run_dir / "reconstructed.mrc"
    log_path = run_dir / "imod.log"

    _run_imod([IMOD_CMD_NEWSTACK, "-input", raw_path, "-output", aligned_path,
               "-xform", xf_path, "-bin", str(binning)], log_path)
    _run_imod([IMOD_CMD_TILT, "-input", aligned_path, "-output", tilt_output_path,
               "-TILTFILE", tlt_path, "-THICKNESS", str(thickness),
               "-RADIAL", "0.35,0.035", "-FalloffIsTrueSigma", "1"], log_path)
    _run_imod([IMOD_CMD_TRIMVOL, "-rx", tilt_output_path, tomogram_path], log_path)
    tilt_output_path.unlink()
    return tomogram_path


def main():
    args = parse_args()

    if args.download_tiltseries or args.download_frames:
        run_dirs = download_dataset(args.dataset_id, args.output_dir, args.runs,
                                    tiltseries=args.download_tiltseries,
                                    frames=args.download_frames)
    else:
        run_dirs = sorted(p for p in (args.output_dir / "tiltseries").glob("*") if p.is_dir())
        if args.runs is not None:
            run_dirs = [p for p in run_dirs if p.name in args.runs]

    if not args.reconstruct:
        return

    if args.tomogram_dir is not None:
        args.tomogram_dir.mkdir(parents=True, exist_ok=True)

    for run_dir in run_dirs:
        tomogram_path = reconstruct_run(run_dir, args.binning)
        print(f"Reconstructed: {tomogram_path}")
        if args.tomogram_dir is not None:
            shutil.copy2(tomogram_path, args.tomogram_dir / f"{run_dir.name}.mrc")


if __name__ == "__main__":
    main()
