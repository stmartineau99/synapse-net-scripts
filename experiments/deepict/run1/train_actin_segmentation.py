import numpy as np
from synapse_net.training import supervised_training
from torch_em.data.sampler import MinForegroundSampler
from pathlib import Path

def actin_supervised_training(train_paths, val_paths, ckpt_dir):
    """Train a network for actin segmentation.
    """
    # TODO maybe decrease batch size
    patch_shape = (64, 384, 384) 
    sampler = MinForegroundSampler(min_fraction=0.025, p_reject=0.95)

    supervised_training(
        name="actin-deepict",
        label_key="/labels/actin",
        patch_shape=patch_shape,
        train_paths=train_paths,
        val_paths=val_paths,
        sampler=sampler,
        save_root=str(ckpt_dir),
    )


def main():
    parent_dir = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/training")
    train_dir = parent_dir / "train"
    val_dir = parent_dir / "val"

    train_paths = [str(p) for p in train_dir.glob("*.h5")]
    val_paths = [str(p) for p in val_dir.glob("*.h5")]

    ckpt_dir = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/training/checkpoints/deepict/run1")
    actin_supervised_training(train_paths, val_paths, ckpt_dir)


if __name__ == "__main__":
    main()
