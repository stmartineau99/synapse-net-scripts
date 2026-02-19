import os
from synapse_net.training.domain_adaptation import mean_teacher_adaptation
from pathlib import Path

def actin_adaptation(train_paths, val_paths, source_ckpt, out_dir):

    patch_shape = (64, 384, 384)

    #TODO need to resize tomos to 10A before training 
    #TODO sample mask, also resize?

    mean_teacher_adaptation(
        name="actin-deepict-adapted-run2",
        unsupervised_train_paths=train_paths,
        unsupervised_val_paths=val_paths,
        raw_key="data",
        patch_shape=patch_shape,
        batch_size=4,
        save_root=out_dir,
        source_checkpoint=source_ckpt,
        confidence_threshold=0.75,
    )


def main():
    parent_dir = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data")

    deepict_dir = parent_dir / "public/deepict"
    train_paths = [str()]
    val_paths = [str()]

    source_ckpt = parent_dir / "training/out/deepict/run2/checkpoints/actin-deepict-run2"
    out_dir = parent_dir / "training/out/deepict/run2"

    actin_adaptation()


if __name__ == "__main__":
    main()
