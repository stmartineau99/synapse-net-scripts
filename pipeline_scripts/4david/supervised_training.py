from pathlib import Path

import configargparse

from synapse_net.training import supervised_training
from synapse_net.training.supervised_training import _parse_input_files


def parse_args():
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.TomlConfigParser(["supervised_training"])
    )
    parser.add_argument("--config", is_config_file_arg=True, help="Path to TOML config file.")

    parser.add_argument("--name", type=str, required=True, help="Name for the model checkpoint.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where the checkpoint will be saved.")  # noqa
    parser.add_argument("--train_dir", type=str, required=True, dest="train_folder", help="Directory with training raw tomograms.")  # noqa
    parser.add_argument("--train_label_dir", type=str, required=True, dest="label_folder", help="Directory with training labels.")  # noqa
    parser.add_argument("--val_dir", type=str, default=None, dest="val_folder", help="Directory with validation raw tomograms. If not given, train_dir is split using val_fraction.")  # noqa
    parser.add_argument("--val_label_dir", type=str, default=None, dest="val_label_folder", help="Directory with validation labels. Required if val_dir is given.")  # noqa
    parser.add_argument("--val_fraction", type=float, default=0.2, help="Fraction of train_dir held out for validation. Ignored if val_dir is given.")  # noqa
    parser.add_argument("--raw_pattern", type=str, default="*.mrc", dest="image_file_pattern", help="Glob pattern for raw tomogram files.")  # noqa
    parser.add_argument("--label_pattern", type=str, default="*.mrc", dest="label_file_pattern", help="Glob pattern for label files.")  # noqa
    parser.add_argument("--patch_shape", type=int, nargs=3, default=[64, 256, 256])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n_iterations", type=int, default=10_000)
    parser.add_argument("--check", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = parse_args()
    if args.check:
        args.batch_size = 1
    args.raw_key = None
    args.label_key = None

    train_paths, train_label_paths, val_paths, val_label_paths, raw_key, label_key = _parse_input_files(args)

    supervised_training(
        name=args.name,
        train_paths=train_paths,
        val_paths=val_paths,
        train_label_paths=train_label_paths,
        val_label_paths=val_label_paths,
        raw_key=raw_key,
        label_key=label_key,
        patch_shape=tuple(args.patch_shape),
        batch_size=args.batch_size,
        lr=args.lr,
        n_iterations=args.n_iterations,
        save_root=args.output_dir,
        check=args.check,
    )


if __name__ == "__main__":
    main()
