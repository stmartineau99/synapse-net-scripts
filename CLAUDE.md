# CLAUDE.md

Guidance for Claude Code in the `synapse-net-scripts` repository. Read `./CONTEXT.md` first; it states the current task.

## Purpose

Train and evaluate 3D U-Net models that segment filaments in cryo-ET tomograms. Models train on
synthetic Polnet/Faket data by supervised training, then transfer to experimental data by
mean-teacher domain adaptation in either unsupervised (USDA) or semisupervised (SSDA) mode.
Semantic segmentations are converted to instances, which feed downstream analysis.

Library code lives in `source/synapse-net` and `source/torch-em`. This repository holds only
the scripts that call them.

## Experimental datasets

The short name is referenced by `./configs/<dataset>/`, `data/experimental/<dataset>/`, and the
`dataset` key. Some short names are aliases for a full name.

- `deepict`
- `opto`, alias for optogenetics
- `synapse`
- `htt`, alias for huntingtin

## Environment

Run every Python command in the `synapse-net-backbones` micromamba environment, not in `super`.

## Directories

| Path | Holds |
|---|---|
| `pipeline_scripts/` | the maintained scripts, in `prepare_datasets/`, `inference/`, `analysis/`, plus the config-driven top level |
| `configs/` | one TOML per run, grouped by experiment dataset |
| `slurm_scripts/` | job submission: `submit.py`, resource presets in `profiles/`, the old sbatch scripts in `deprecated/` |
| `experiments/` | experiment logs and exploratory work, each with its own README |
| `evaluation_results/`, `notebooks/`, `slurm_logs/`, `slurm_metrics/` | outputs and scratch |

All data lives under `data_root`, set in `[run_info]`, and nowhere else. It is `<parent>/data`.

## Pipeline

Synthetic branch:

| Script | Reads | Writes |
|---|---|---|
| `prepare_training.py` | `data/simulation/<synthetic_dataset>/` tomograms and masks | `data/training/<synthetic_dataset>/{train,val,test}/*.h5` |
| `supervised_training.py` | `data/training/<synthetic_dataset>/{train,val}/*.h5` | `data/training/out/<dataset>/run<N>/checkpoints/` |

Experimental branch:

| Script | Reads | Writes |
|---|---|---|
| `prepare_<dataset>.py` | raw data mrc and labels | `data/experimental/<dataset>/h5/*.h5` |
| `domain_adaptation.py` | data in h5, plus a supervised checkpoint as the source | `data/training/out/<dataset>/run<N>/checkpoints/` |

Inference and analysis:

| Script | Reads | Writes |
|---|---|---|
| `predict_actin.py` | `raw` in an h5, plus a checkpoint | `predictions/<model>` and `segmentations/<model>` into the same h5 |
| `predict_actin_instances.py` | `segmentations/<model>` in an h5 | `<out>/<tomogram>/<tomogram>_instances.csv` |

HDF5 layout from the prepare scripts: `raw` float32, `labels/<structure>` uint8, `sample_mask`
uint8, all gzip compressed. This layout is a contract the user maintains.

## Configuration

Configs are TOML, read by `configargparse.TomlConfigParser`. Each key maps onto a CLI flag of
the consuming script.

| Section | Read by |
|---|---|
| `[run_info]` | every config-driven script; holds `data_root`, `synthetic_dataset`, `dataset`, `run` |
| `[prepare_training]` | `prepare_training.py` |
| `[supervised_training]` | `supervised_training.py` |
| `[domain_adaptation]` | `domain_adaptation.py` |
| `[semisupervised_learning]` | `semisupervised_training.py` |
| `[predict_actin]` | `predict_actin.py` |
| `[jobs]` | `slurm_scripts/submit.py` |

`run<N>` identifies one experiment. The number must agree in the config path
`./configs/<dataset>/<dataset>_run<N>.toml`, the `run` key inside it, the output tree
`data/training/out/<dataset>/run<N>/`, and the model name `<structure>-<dataset>-run<N>`.
The counter is per dataset. Never reuse a number.

Domain adaptation mode is inferred, not passed. `supervised_train_glob` present means SSDA,
absent means USDA.

Not every script is config driven. Scripts in `prepare_datasets/` are dataset-specific and have
hardcoded paths and module-level constants.

## Submitting jobs

A run config states how its jobs are submitted. `[jobs].env` is the micromamba env. Each
`[jobs.<name>]` table has a `script`, relative to the repository root, and a `slurm_profile`, a file
in `slurm_scripts/profiles/`. Optional keys: `env`, which overrides `[jobs].env`, and
`slurm = {...}`, merged over the profile. A script's flags go in its own section, not in the job
table. See `configs/template.toml`.

```bash
python slurm_scripts/submit.py configs/deepict/deepict_run39.toml --dry_run
python slurm_scripts/submit.py configs/deepict/deepict_run39.toml
python slurm_scripts/submit.py <config> --job predict_actin
```

Group several jobs in one config only when they form a chain, for example train and then
predict. Without `--job`, a single job is submitted alone, and several jobs are submitted as a
chain in config order, each with `--dependency=afterok` on the previous one. `--job <name>`
submits only that job, to redo a failed stage or to test one. The generated `.sh` and the job
output go to `slurm_logs/<config>_<job>_<timestamp>.{sh,out}`. Loader checks (`--check`) are not
submitted. The user runs them directly on the Jupyter desktop.

The scripts in `prepare_datasets/` and `analysis/` are not config driven and keep their own
`slurm_scripts/sbatch_*.sh`.
