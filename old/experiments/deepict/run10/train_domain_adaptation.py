from pathlib import Path
from synapse_net.training.domain_adaptation import mean_teacher_adaptation
from torch_em.data.sampler import MinForegroundSampler

def actin_adaptation(train_paths, val_paths, model_path, out_dir):
    patch_shape = (64, 384, 384)
    patch_sampler = MinForegroundSampler(min_fraction=0.95)

    mean_teacher_adaptation(
        name="actin-deepict-adapted-run10",
        unsupervised_train_paths=train_paths,
        unsupervised_val_paths=val_paths,
        raw_key="raw",
        patch_shape=patch_shape,
        save_root=out_dir,
        source_checkpoint=model_path,
        confidence_threshold=0.75,
        batch_size=2,
        lr=2e-4,
        train_sample_mask_paths=train_paths,
        val_sample_mask_paths=val_paths,
        sample_mask_key="sample_mask",
        patch_sampler=patch_sampler,
        check=False
    )

def main():
    PARENT_DIR = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/public/deepict/h5")
    train_paths = sorted([str(p) for p in PARENT_DIR.glob("00004.h5")])
    val_paths = sorted([str(p) for p in PARENT_DIR.glob("00011.h5")])

    model_path = "/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/training/out/deepict/run9/checkpoints/actin-deepict-run9"

    out_dir = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/training/out/deepict/run9")
    actin_adaptation(train_paths, val_paths, model_path, out_dir)


if __name__ == "__main__":
    main()