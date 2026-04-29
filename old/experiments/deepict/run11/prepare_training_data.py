import numpy as np
from synapse_net.file_utils import read_mrc
import mrcfile 
import h5py
from pathlib import Path

def main():
    parent_dir = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/simulation/deepict_dataset_4")
    out_dir = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/training/deepict_dataset_4")

    ntomos = 20

    for c in range(4):
        if c == 3:
            splits = [
                ("train", list(range(0, int(0.8 * ntomos)))),
                ("val",   list(range(int(0.8 * ntomos), ntomos))),
            ]
        else:
            splits = [
                ("train", list(range(0, int(0.7 * ntomos)))),
                ("val",   list(range(int(0.7 * ntomos), int(0.9 * ntomos)))),
                ("test",  list(range(int(0.9 * ntomos), ntomos))),
            ]

        tomo_dir = parent_dir / f"train_dir_{c}" / "faket_tomograms"
        label_dir = parent_dir / f"simulation_dir_{c}" / "tomos"

        for split, indices in splits:
            out_dir_split = out_dir / split
            out_dir_split.mkdir(parents=True, exist_ok=True)

            print(f"\nWriting {split} split for condition {c} ({len(indices)} tomograms)...")

            for i in indices:
                tomo_path = tomo_dir / f"tomogram_{c}_{i}_faket.mrc"
                if not tomo_path.exists():
                    raise FileNotFoundError(f"Missing tomo: {tomo_path}.")

                label_path = label_dir / f"tomo_actin_mask_{i}.mrc"
                if not label_path.exists():
                    raise FileNotFoundError(f"Missing labels: {label_path}.")

                out_path = out_dir_split / f"tomogram_{c}_{i}.h5"
                if out_path.exists():
                    print(f"Skipping {out_path}, file already exists.")
                    continue

                tomo = read_mrc(tomo_path)[0]
                labels = read_mrc(label_path)[0]

                with h5py.File(out_path, "w") as f:
                    f.create_dataset("raw", data=tomo, compression="gzip")
                    f.create_dataset("/labels/actin", data=labels, compression="gzip")

                print(f"tomogram_{c}_{i} saved to {out_path}.")


if __name__ == "__main__":
    main()