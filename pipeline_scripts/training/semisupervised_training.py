import random
import h5py
import configargparse
from pathlib import Path
from synapse_net.training.semisupervised_training import semisupervised_training
from torch_em.data.sampler import MinForegroundSampler

LABELED_FRACTION_SEED = 42

BACKBONE_MODEL_TYPES = {
    "sam": "vit_b_em_organelles",
    "sam2": "hvit_b",
    "dinov3": "vit_b",
}

def has_label_key(path, label_key):
    with h5py.File(path, "r") as f:
        return label_key in f


def parse_args():
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.TomlConfigParser(
            ["run_info", "semisupervised_learning"]
        )
    )
    parser.add_argument("--config", is_config_file_arg=True, help="Path to TOML config file.")

    # run_info
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--synthetic_dataset", type=str, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--run", type=int, required=True)

    # semisupervised_learning
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--source_checkpoint", type=str, default=None)
    parser.add_argument("--backbone", type=str, default=None, choices=list(BACKBONE_MODEL_TYPES))
    parser.add_argument("--train_glob", type=str, default="*.h5")
    parser.add_argument("--val_glob", type=str, default="*.h5")
    parser.add_argument("--supervised_train_glob", type=str, required=True)
    parser.add_argument("--supervised_val_glob", type=str, required=True)
    parser.add_argument("--raw_key", type=str, default="raw")
    parser.add_argument("--label_key", type=str, default="/labels/actin")
    parser.add_argument("--patch_shape", type=int, nargs=3, default=[64, 256, 256])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_iterations", type=int, default=10_000)
    parser.add_argument("--teacher_warmup_iterations", type=int, default=1000)
    parser.add_argument("--confidence_threshold", type=float, default=0.9)
    parser.add_argument("--labeled_fraction", type=float, default=1.0)
    parser.add_argument("--use_sample_mask", action="store_true", default=False)
    parser.add_argument("--sample_mask_key", type=str, default="sample_mask")
    parser.add_argument("--check", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path(args.data_root) / "experimental" / args.dataset / "h5"

    if args.check:
        args.batch_size = 1

    run_name = f"actin-{args.dataset}-run{args.run}"
    out_dir = Path(args.data_root) / "training" / "out" / args.dataset / f"run{args.run}"

    unsupervised_train_paths = sorted(str(p) for p in data_dir.glob(args.train_glob))
    unsupervised_val_paths = sorted(str(p) for p in data_dir.glob(args.val_glob))

    supervised_train_paths = sorted(
        str(p) for p in data_dir.glob(args.supervised_train_glob)
        if has_label_key(p, args.label_key)
    )
    supervised_val_paths = sorted(
        str(p) for p in data_dir.glob(args.supervised_val_glob)
        if has_label_key(p, args.label_key)
    )
    if not supervised_train_paths:
        raise FileNotFoundError(f"No labelled files found in {data_dir} for key '{args.label_key}'.")

    if args.labeled_fraction < 1.0:
        rng = random.Random(LABELED_FRACTION_SEED)
        n_train = max(1, int(len(supervised_train_paths) * args.labeled_fraction))
        n_val = max(1, int(len(supervised_val_paths) * args.labeled_fraction))
        supervised_train_paths = sorted(rng.sample(supervised_train_paths, n_train))
        supervised_val_paths = sorted(rng.sample(supervised_val_paths, n_val))

    supervised_sampler = MinForegroundSampler(min_fraction=0.025, p_reject=0.95)

    if args.use_sample_mask:
        unsupervised_sampler = MinForegroundSampler(min_fraction=0.5)
        train_mask_paths, val_mask_paths = unsupervised_train_paths, unsupervised_val_paths
    else:
        unsupervised_sampler, train_mask_paths, val_mask_paths = None, None, None

    print(f"Running semisupervised training for run {args.run}.")

    semisupervised_training(
        name=run_name,
        unsupervised_train_paths=unsupervised_train_paths,
        unsupervised_val_paths=unsupervised_val_paths,
        supervised_train_paths=supervised_train_paths,
        supervised_val_paths=supervised_val_paths,
        label_key=args.label_key,
        patch_shape=tuple(args.patch_shape),
        save_root=str(out_dir),
        raw_key=args.raw_key,
        confidence_threshold=args.confidence_threshold,
        batch_size=args.batch_size,
        lr=args.lr,
        n_iterations=args.n_iterations,
        teacher_warmup_iterations=args.teacher_warmup_iterations,
        source_checkpoint=args.source_checkpoint,
        supervised_sampler=supervised_sampler,
        unsupervised_sampler=unsupervised_sampler,
        train_mask_paths=train_mask_paths,
        val_mask_paths=val_mask_paths,
        sample_mask_key=args.sample_mask_key,
        backbone=args.backbone,
        model_type=BACKBONE_MODEL_TYPES.get(args.backbone),
        check=args.check,
    )


if __name__ == "__main__":
    main()
