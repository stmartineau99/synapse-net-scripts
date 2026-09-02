# U-Net Training Tutorial

Train and run a filament segmentation model on your own mrc tomograms.

## Requirements

Installation instructions for `synapse_net`:

First clone the repository: 
```
git clone https://github.com/computational-cell-analytics/synapse-net
```

Then setup a new conda environment. **Make sure to do this on a GPU node so that CUDA is configured correctly.**

```
cd synapse-net 
conda env create -f environment.yaml
```

Activate the environment and install `synapse-net` using pip:
```
conda activate synapse-net
pip install .
```

## Input data

Input raw tomograms and label volumes are both `.mrc` files. `raw` and `labels` (and `masks`, used for restricting the output during inference) are paired by sorted order within each directory, so use matching filenames across directories, for example:

```
data/
├── train/
│   ├── raw/
│   │   ├── tomo_01.mrc
│   │   └── tomo_02.mrc
│   └── labels/
│       ├── tomo_01.mrc
│       └── tomo_02.mrc
└── val/
    ├── raw/
    │   └── tomo_03.mrc
    └── labels/
        └── tomo_03.mrc
```

## Training

```
python supervised_training.py --config configs/example.toml
```

`configs/example.toml` is a ready-to-edit template. 

| Key | Meaning |
|---|---|
| `name` | Name of the trained model checkpoint. |
| `output_dir` | Directory where the checkpoint will be saved. |
| `train_dir` | Directory with training raw tomograms. |
| `train_label_dir` | Directory with training labels. |
| `val_dir` | Directory with validation raw tomograms. |
| `val_label_dir` | Directory with validation labels. Required if `val_dir` is given. |
| `val_fraction` | Fraction of `train_dir` held out for validation. Ignored if `val_dir` is given. Default `0.2`. |
| `patch_shape` | Training patch shape `[z, y, x]`. Default `[64, 256, 256]`. |
| `batch_size` | Training batch size. |
| `lr` | Learning rate. |
| `n_iterations` | Number of training iterations. |

Before a real training run use `--check` to visualize a few samples from the data loaders and confirm that the labels line up with the raw data. 

The model checkpoint can be found at `<output_dir>/checkpoints/<name>`.

## Inference

```
python inference.py 
    --checkpoint /path/to/output_dir/checkpoints/name \
    --input_dir /path/to/tomograms \
    --output_dir /path/to/segmentations
```

Optional arguments:

| Flag | Meaning |
|---|---|
| `--label_dir` | Directory with ground truth mrc labels. 
| `--mask_dir` | Directory with mrc masks, used to restrict the prediction. |
| `--threshold` | Threshold for binarizing the foreground prediction. Default `0.5`. |

One `.h5` file is created per input tomogram, written to `output_dir`, containing:

- `raw` - input tomogram.
- `predictions/<model>` - raw foreground prediction.
- `segmentations/<model>` - thresholded segmentation.
- `labels/gt` - ground truth labels.

If `--label_dir` is given, per-tomogram metrics are saved to `metrics.csv`, including dice, precision, and recall. 
