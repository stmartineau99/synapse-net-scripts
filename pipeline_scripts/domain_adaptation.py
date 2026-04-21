import configargparse
from pathlib import Path
from synapse_net.training.domain_adaptation import mean_teacher_adaptation
from torch_em.data.sampler import MinForegroundSampler

def parse_args():
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.TomlConfigParser(
            ["run_info", "domain_adaptation"]
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

    # domain_adaptation
    parser.add_argument("--source_checkpoint", type=str, default=None)
    parser.add_argument("--train_glob", type=str, default="*.h5")
    parser.add_argument("--val_glob", type=str, default="*.h5")
    parser.add_argument("--raw_key", type=str, default="raw")
    parser.add_argument("--sample_mask_key", type=str, default="sample_mask")
    parser.add_argument("--patch_shape", type=int, nargs=3, default=[64, 384, 384])
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--check", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = parse_args()

    run_name = f"actin-{args.real_dataset}-run{args.run}"
    data_dir = Path(args.data_root) / "experimental" / args.real_dataset / "h5"
    out_dir = (
        Path(args.data_root) / "training" / "out"
        / args.real_dataset / f"run{args.run}-adapted"
    )
    n_iterations = 100_000 // args.batch_size

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

    patch_sampler = MinForegroundSampler(min_fraction=0.95)

    mean_teacher_adaptation(
        name=f"{run_name}-adapted",
        unsupervised_train_paths=train_paths,
        unsupervised_val_paths=val_paths,
        raw_key=args.raw_key,
        patch_shape=tuple(args.patch_shape),
        save_root=str(out_dir),
        source_checkpoint=source_checkpoint,
        confidence_threshold=0.75,
        batch_size=args.batch_size,
        lr=args.lr,
        n_iterations=n_iterations,
        train_sample_mask_paths=train_paths,
        val_sample_mask_paths=val_paths,
        sample_mask_key=args.sample_mask_key,
        patch_sampler=patch_sampler,
        check=args.check,
    )


if __name__ == "__main__":
    main()
