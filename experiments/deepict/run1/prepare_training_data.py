import numpy as np
from synapse_net.file_utils import read_mrc
import mrcfile 
import h5py

from pathlib import Path

def main():
    parent_dir = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/simulated/deepict_dataset_1")
    out_dir = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/training/deepict_dataset_1")

    conditions = [0, 1, 2]
    train_idx = list(range(0, 16))
    val_idx   = list(range(16, 20))

    for c in conditions:
        tomo_dir = parent_dir / f"train_dir_{c}" / "faket_tomograms"
        label_dir = parent_dir / f"simulation_dir_{c}" / "tomos"

        for split, indices in [("train", train_idx), ("val", val_idx)]:
            out_dir_split = out_dir / split
            out_dir_split.mkdir(parents=True, exist_ok=True)

            for i in indices: 
                tomo_path = tomo_dir / f"tomogram_{c}_{i}_faket.mrc"

                if not tomo_path.exists():
                    raise FileNotFoundError(f"Missing tomo: {tomo_path}.")
              
                label_path = label_dir / f"tomo_lbls_{i}.mrc"

                if not label_path.exists():
                    raise FileNotFoundError(f"Missing labels: {label_path}.")
                
                out_path = out_dir_split / f"tomogram_{c}_{i}.h5"
                tomo = read_mrc(tomo_path)[0]
                labels = read_mrc(label_path)[0]

                # label mapping: actin -> 1, others -> 0
                if c in (0, 1):
                    actin_label = (labels == 2).astype(np.uint8)
                elif c == 2:
                    actin_label = (labels == 5).astype(np.uint8)

                with h5py.File(out_path, "w") as f:
                    f.create_dataset("raw", data=tomo, compression="gzip")
                    f.create_dataset("/labels/actin", data=actin_label, compression="gzip")
            
if __name__ == "__main__":
    main()
