import configargparse
import torch_em.loss
from pathlib import Path
from synapse_net.training import supervised_training
from torch_em.data.sampler import MinForegroundSampler


def parse_args():
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.TomlConfigParser(
            ["run_info", "supervised_training"]
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

    # supervised_training
    parser.add_argument("--label_key", type=str, default="/labels/actin")
    parser.add_argument(
        "--patch_shape", type=int, nargs=3, default=[64, 256, 256]
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--loss_fn", type=str, default=None)
    parser.add_argument("--conditions", type=int, nargs="*", default=None)
    parser.add_argument("--check", action="store_true", default=False)

    return parser.parse_args()


def filter_by_conditions(paths, conditions):
    def condition_of(p):
        return int(Path(p).stem.split("_")[1])
    return [p for p in paths if condition_of(p) in conditions]


def main():
    args = parse_args()

    run_name = f"actin-{args.real_dataset}-run{args.run}"
    train_data_dir = Path(args.data_root) / "training" / args.synthetic_dataset
    out_dir = (
        Path(args.data_root) / "training" / "out"
        / args.real_dataset / f"run{args.run}"
    )
    n_iterations = 100_000 // args.batch_size

    train_paths = sorted(
        str(p) for p in (train_data_dir / "train").glob("*.h5")
    )
    val_paths = sorted(
        str(p) for p in (train_data_dir / "val").glob("*.h5")
    )

    if args.conditions:
        train_paths = filter_by_conditions(train_paths, args.conditions)
        val_paths = filter_by_conditions(val_paths, args.conditions)

    sampler = MinForegroundSampler(min_fraction=0.025, p_reject=0.95)
    loss_fn = getattr(torch_em.loss, args.loss_fn)() if args.loss_fn else None

    supervised_training(
        name=run_name,
        label_key=args.label_key,
        patch_shape=tuple(args.patch_shape),
        train_paths=train_paths,
        val_paths=val_paths,
        sampler=sampler,
        batch_size=args.batch_size,
        lr=args.lr,
        loss_fn=loss_fn,
        n_iterations=n_iterations,
        save_root=str(out_dir),
        check=args.check,
    )


if __name__ == "__main__":
    main()
