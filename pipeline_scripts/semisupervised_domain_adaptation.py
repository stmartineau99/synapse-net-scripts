import h5py
import configargparse
from pathlib import Path
from synapse_net.training.domain_adaptation import mean_teacher_adaptation
from torch_em.data.sampler import MinForegroundSampler


def has_label_key(path, label_key):
    with h5py.File(path, "r") as f:
        return label_key in f


def parse_args():
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.TomlConfigParser(
            ["run_info", "semisupervised_domain_adaptation"]
        )
    )
    parser.add_argument(
        "--config", is_config_file_arg=True, help="Path to TOML config file."
    )

    # run_info
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--synthetic_dataset", type=str, required=True)
    parser.add_argument("--real_dataset", type=str, required=True)
    parser.add_argument("--run", type=int, required=True)

    # semisupervised_domain_adaptation
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--source_checkpoint", type=str, default=None)
    parser.add_argument("--train_glob", type=str, default="*.h5")
    parser.add_argument("--val_glob", type=str, default="*.h5")
    parser.add_argument("--raw_key", type=str, default="raw")
    parser.add_argument("--sample_mask_key", type=str, default="sample_mask")
    parser.add_argument("--patch_shape", type=int, nargs=3, default=[64, 384, 384])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_iterations", type=int, default=50_000)
    parser.add_argument("--check", action="store_true", default=False)
    parser.add_argument("--supervised_train_glob", type=str, default="*.h5")
    parser.add_argument("--supervised_val_glob", type=str, default="*.h5")
    parser.add_argument("--raw_key_supervised", type=str, default="raw")
    parser.add_argument("--label_key", type=str, default="/labels/actin")

    return parser.parse_args()


def main():
    args = parse_args()

    run_name = f"actin-{args.real_dataset}-run{args.run}-ssda"
    data_dir = Path(args.data_dir) if args.data_dir else Path(args.data_root) / "experimental" / args.real_dataset / "h5"
    out_dir = (
        Path(args.data_root) / "training" / "out"
        / args.real_dataset / run_name
    )
    if args.check:
        args.batch_size = 1

    if args.source_checkpoint:
        source_checkpoint = args.source_checkpoint
    else:
        supervised_out = (
            Path(args.data_root) / "training" / "out"
            / args.real_dataset / f"run{args.run}"
        )
        source_checkpoint = str(supervised_out / "checkpoints" / run_name)

    train_paths = sorted(str(p) for p in data_dir.glob(args.train_glob))
    val_paths = sorted(str(p) for p in data_dir.glob(args.val_glob))

    # unsupervised_sampler = MinForegroundSampler(min_fraction=0.95)

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
    supervised_kwargs = dict(
        supervised_train_paths=supervised_train_paths,
        supervised_val_paths=supervised_val_paths,
        label_key=args.label_key,
        raw_key_supervised=args.raw_key_supervised,
    )

    mean_teacher_adaptation(
        name=run_name,
        unsupervised_train_paths=train_paths,
        unsupervised_val_paths=val_paths,
        raw_key=args.raw_key,
        patch_shape=tuple(args.patch_shape),
        save_root=str(out_dir),
        source_checkpoint=source_checkpoint,
        confidence_threshold=0.75,
        batch_size=args.batch_size,
        lr=args.lr,
        n_iterations=args.n_iterations,
        # train_mask_paths=train_paths,
        # val_mask_paths=val_paths,
        # sample_mask_key=args.sample_mask_key,
        # unsupervised_sampler=unsupervised_sampler,
        supervised_sampler=False,
        check=args.check,
        **supervised_kwargs,
    )


if __name__ == "__main__":
    main()
