# CLAUDE.md

Guidance for Claude Code in the `synapse-net-scripts` repository.

`/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/` is the parent directory of all the
user's projects, and is the form every config and sbatch script uses. The same tree is also
mounted at `/mnt/vast-nhr/projects/nim00020/sage/`. Two paths, one directory; neither is
wrong. That parent has its own `CLAUDE.md` describing the source subprojects and the
simulation pipeline.

## Asking

Ask before searching when the question is about intent, currency, or anything outside this
repository. These are cheap for the user to answer and expensive to infer:

- which run, dataset, or checkpoint a task means
- whether a script, config, or result is still current
- what an experiment was for, or what comes next
- what has already been run
- which reading of an ambiguous request is intended

The trigger is a budget, not a feeling: if the answer needs more than about one tool call and
it falls in that list, ask. Do not spend five calls building a probabilistic answer to
something the user can state outright. Batch questions into one prompt rather than a stream.

Search, do not ask, for what the code states: a value in a config, what a function does,
which script writes which path.

Trust what this file states rather than re-deriving it. The HDF5 layout and instance CSV
columns under Pipeline are a contract the user maintains. Do not open an h5 to check that
`raw`, `labels/<structure>`, or `sample_mask` are present; they are.

## Purpose

Train and evaluate 3D U-Net models that segment filaments in cryo-ET tomograms, actin first
and other filament types later. Models train on synthetic Polnet/Faket data by supervised
training, then transfer to experimental data by mean-teacher domain adaptation in either
unsupervised (USDA) or semisupervised (SSDA) mode. Semantic segmentations are converted to 
instances, which feed downstream analysis.

Library code lives in `source/synapse-net` and `source/torch-em` for training, and
`source/bioimage-cpp` for image processing tasks such as instance segmentation. This repository
holds only the scripts that call them.

## Experimental datasets

The short name is what appears everywhere: `configs/<dataset>/`,
`data/experimental/<dataset>/`, and the `real_dataset` key.

| Short name | Full name | State |
|---|---|---|
| `deepict` | | 10 A pixel size. The current focus. |
| `opto` | optogenetics | saved for later |
| `synapse` | | saved for later |
| `htt` | huntingtin | saved for later |

Assume `deepict` unless the task names another. The other three have config directories,
prepare scripts, and run histories, but none of it is current; do not read them on spec.

## Environment

Run every Python command in `synapse-net`, not in `super`:

```bash
micromamba run -n synapse-net python <script>
```

Do not install packages. Give the user the install command when one is needed.

## Directories

| Path | Holds |
|---|---|
| `pipeline_scripts/` | the maintained scripts, in `prepare_datasets/`, `inference/`, `analysis/`, plus the config-driven top level |
| `configs/` | one TOML per run, grouped by experiment dataset |
| `slurm_scripts/` | job submission |
| `experiments/` | experiment logs and exploratory work, each with its own README |
| `evaluation_results/`, `notebooks/`, `slurm_logs/`, `slurm_metrics/` | outputs and scratch |

All data lives under `data_root`, set in `[run_info]`, and nowhere else. It is `<parent>/data`.

## Pipeline

Synthetic branch:

| Script | Reads | Writes |
|---|---|---|
| `prepare_training.py` | `data/simulation/<synthetic_dataset>/` tomograms and masks | `data/training/<synthetic_dataset>/{train,val,test}/*.h5` |
| `supervised_training.py` | `data/training/<synthetic_dataset>/{train,val}/*.h5` | `data/training/out/<real_dataset>/run<N>/checkpoints/` |

Experimental branch:

| Script | Reads | Writes |
|---|---|---|
| `prepare_<dataset>.py` | experimental mrc and its ground truth | `data/experimental/<dataset>/h5/*.h5` |
| `prepare_<dataset>_subvolumes.py` | `data/experimental/<dataset>/h5/*.h5` | `.../h5/subvolumes/{train,val,test}/` |
| `domain_adaptation.py` | experimental data an h5, plus a supervised checkpoint as the source | `data/training/out/<real_dataset>/run<N>/checkpoints/` |

Inference and analysis:

| Script | Reads | Writes |
|---|---|---|
| `predict_actin.py` | `raw` in an h5, plus a checkpoint | `predictions/<model>` and `segmentations/<model>` back into the same h5 |
| `predict_actin_instances.py` | `segmentations/<model>` in an h5 | `<out>/<tomogram>/<tomogram>_instances.{npz,csv}` |
| `analysis/` | to be extended in the future |

HDF5 layout from the prepare scripts: `raw` float32, `labels/<structure>` uint8, `sample_mask`
uint8, all gzip compressed.

Instance CSV columns are `ID,X,Y,Z`. One `ID` is one filament, coordinates are in
`--pixel_size` units, and rows within an `ID` are ordered along the filament.

## Configuration

Configs are TOML, read by `configargparse.TomlConfigParser`. Each key maps one-to-one onto a
command line flag of the consuming script.

| Section | Read by |
|---|---|
| `[run_info]` | every config-driven script; holds `data_root`, `synthetic_dataset`, `real_dataset`, `run` |
| `[prepare_training]` | `prepare_training.py` |
| `[supervised_training]` | `supervised_training.py` |
| `[domain_adaptation]` | `domain_adaptation.py` |

A script declares the sections it reads and rejects unknown keys in them, so a key that no
script reads is an error rather than an annotation.

`run<N>` identifies one experiment. The number must agree in the config path
`configs/<dataset>/<dataset>_run<N>.toml`, the `run` key inside it, the output tree
`data/training/out/<real_dataset>/run<N>/`, and the model name `<structure>-<real_dataset>-run<N>`.
The counter is per dataset. Never reuse a number.

Domain adaptation mode is inferred, not passed. `supervised_train_glob` present means SSDA,
absent means USDA.

Not every script is config driven. Scripts in `prepare_datasets/` are dataset-specific and have hardcoded paths and module-level constansts. 

## SLURM

`slurm_scripts/submit_training_pipeline.sh` is the main training pipeline entrypoint: chains 
prepare dataset, supervised training, and domain adaptation with `--dependency=afterok`, and submits a
`collect_slurm_metrics.py` job after each stage. Don't submit jobs unless the user requests. Edit `RUN_PREPARE`,
`RUN_SUPERVISED`, and `RUN_DA` variables at the top of that file.

| Partition | Use |
|---|---|
| `large96s` | CPU work: dataset preparation, downloads, analysis, metrics |
| `grete:shared` with `-G A100` | training and heavy inference |
| `grete:interactive` with a `1g.20gb` MIG slice and `--qos=2h` | short prediction jobs |

For testing and data exploration (inference and analysis), use the
`interactive-node` skill and run with `srun`.
