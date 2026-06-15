import numpy as np
from synapse_net.training import supervised_training
from torch_em.data.sampler import MinForegroundSampler
from pathlib import Path

def actin_supervised_training(train_paths, val_paths, out_dir):
    """Train a network for actin segmentation.
    """
    
    patch_shape = (64, 384, 384) 
    sampler = MinForegroundSampler(min_fraction=0.025, p_reject=0.95)

    supervised_training(
        name="actin-deepict-run6",
        label_key="/labels/actin",
        patch_shape=patch_shape,
        train_paths=train_paths,
        val_paths=val_paths,
        sampler=sampler,
        batch_size=4,
        lr=1e-4,
        n_iterations=25000,
        save_root=str(out_dir),
        check=False
    )

def main():
    parent_dir = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/training/deepict_dataset_3")
    train_dir = parent_dir / "train"
    val_dir = parent_dir / "val"

    train_paths = [str(p) for p in train_dir.glob("*.h5")]
    val_paths = [str(p) for p in val_dir.glob("*.h5")]

    out_dir = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/training/out/deepict/run6")
    actin_supervised_training(train_paths, val_paths, out_dir)


if __name__ == "__main__":
    main()
