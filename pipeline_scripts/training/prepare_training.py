import h5py
import configargparse
from pathlib import Path
from synapse_net.file_utils import read_mrc


def parse_args():
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.TomlConfigParser(
            ["run_info", "prepare_training"]
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

    # prepare_training
    parser.add_argument("--n_conditions", type=int, default=1)
    parser.add_argument("--n_tomos", type=int, required=True)
    parser.add_argument("--conditions", type=int, nargs="*", default=None)
    parser.add_argument("--train_fraction", type=float, default=0.7)
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--raw_key", type=str, default="raw")
    parser.add_argument("--label_key", type=str, default="/labels/actin")

    return parser.parse_args()


def main():
    args = parse_args()

    TOMO_PATTERN = "train_dir_{c}/faket_tomograms/tomogram_{c}_{i}_faket.mrc"
    LABEL_PATTERN = "simulation_dir_{c}/tomos/tomo_actin_mask_{i}.mrc"

    sim_dir = Path(args.data_root) / "simulation" / args.synthetic_dataset
    out_dir = Path(args.data_root) / "training" / args.synthetic_dataset
    conditions = args.conditions if args.conditions else list(range(args.n_conditions))

    n = args.n_tomos
    train_end = int(args.train_fraction * n)
    val_end = int((args.train_fraction + args.val_fraction) * n)
    train_idx = list(range(0, train_end))
    val_idx = list(range(train_end, val_end))
    test_idx = list(range(val_end, n))

    splits = [("train", train_idx), ("val", val_idx), ("test", test_idx)]

    for c in conditions:
        for split, indices in splits:
            split_dir = out_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            print(f"\nCondition {c} — {split} ({len(indices)} tomograms)")

            for i in indices:
                tomo_path = sim_dir / TOMO_PATTERN.format(c=c, i=i)
                label_path = sim_dir / LABEL_PATTERN.format(c=c, i=i)

                if not tomo_path.exists():
                    raise FileNotFoundError(f"Missing tomo: {tomo_path}")
                if not label_path.exists():
                    raise FileNotFoundError(f"Missing label: {label_path}")

                out_path = split_dir / f"tomogram_{c}_{i}.h5"
                if out_path.exists():
                    print(f"  Skipping {out_path.name}, already exists.")
                    continue

                tomo = read_mrc(tomo_path)[0]
                labels = read_mrc(label_path)[0]

                with h5py.File(out_path, "w") as f:
                    f.create_dataset(args.raw_key, data=tomo, compression="gzip")
                    f.create_dataset(args.label_key, data=labels, compression="gzip")

                del tomo, labels

                print(f"  Saved {out_path.name}")


if __name__ == "__main__":
    main()
