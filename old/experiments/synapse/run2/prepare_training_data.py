import numpy as np
from synapse_net.file_utils import read_mrc
import mrcfile 
import h5py
from pathlib import Path

def main():
    parent_dir = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/simulation/synapse_dataset_1")
    out_dir = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/training/synapse_dataset_1")
    
    conditions = [0, 1]
    ntomos = 25
    # train/val/test 80/20/10 
    train_idx = list(range(0, int(0.7*ntomos)))
    val_idx = list(range(int(0.7*ntomos), int(0.9*ntomos)))
    test_idx = list(range(int(0.9*ntomos), ntomos))

    for c in conditions:
        tomo_dir = parent_dir / f"train_dir_{c}" / "faket_tomograms"
        label_dir = parent_dir / f"simulation_dir_{c}" / "tomos"

        for split, indices in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            print(f"\nWriting {split} split for condition {c} ({len(indices)} tomograms)...")
            out_dir_split = out_dir / split
            out_dir_split.mkdir(parents=True, exist_ok=True)

            for i in indices: 
                tomo_path = tomo_dir / f"tomogram_{c}_{i}_faket.mrc"

                if not tomo_path.exists():
                    raise FileNotFoundError(f"Missing tomo: {tomo_path}.")
              
                label_path = label_dir / f"tomo_actin_mask_{i}.mrc"

                if not label_path.exists():
                    raise FileNotFoundError(f"Missing labels: {label_path}.")
                
                out_path = out_dir_split / f"tomogram_{c}_{i}.h5"
                tomo = read_mrc(tomo_path)[0]
                labels = read_mrc(label_path)[0]

                if not out_path.exists():
                    with h5py.File(out_path, "w") as f:
                        f.create_dataset("raw", data=tomo, compression="gzip")
                        f.create_dataset("/labels/actin", data=labels, compression="gzip")

                print(f"tomogram_{c}_{i} saved to {out_path}.")
            
if __name__ == "__main__":
    main()
