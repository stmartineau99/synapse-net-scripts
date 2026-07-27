# Opto Experiments

All runs train an actin segmentation model on optogenetics cryo-ET data, with checkpoint and output saved to `data/training/out/opto/run{N}/`.

Config files: `configs/opto/opto_run{N}.toml`  
Training script: `pipeline_scripts/supervised_training.py`  
SLURM script: `sbatch slurm_scripts/sbatch_supervised_training.sh configs/opto/opto_run{N}.toml`

---

## Run 1

**Dataset:** `opto_dataset_0` (20 tomograms, 3 conditions)  
**Conditions:** [0, 1, 2]  
**Train:** 42 tomograms (14 per condition)  
**Val:** 12 tomograms (4 per condition)  
**Test:** 6 tomograms (2 per condition)  
**Loss:** DiceLoss (default)  
**lr:** 4e-4 | **batch_size:** 4 | **n_iterations:** 25,000

Baseline supervised run on `opto_dataset_0`.

---

## Run 2

**Type:** SL  
**Source checkpoint:** none  
**Data:** `experimental/optogenetics/h5`  
**lr:** 2e-4 | **batch_size:** 2 | **n_iterations:** 10,000

Supervised training on real optogenetics data from scratch. Strong baseline for comparison with future DA runs.

---

## Run 3

**Type:** USDA  
**Source checkpoint:** `actin-opto-run1`  
**Data:** `experimental/optogenetics/h5`  
**lr:** 1e-4 | **batch_size:** 1 | **n_iterations:** 10,000

Paired with run4 to compare SSDA vs. USDA from the same starting checkpoint.

---

## Run 4

**Type:** SSDA  
**Source checkpoint:** `actin-opto-run1`  
**Data:** `experimental/optogenetics/h5`  
**Train:** 1 tomogram | **Val:** 1 tomogram  
**lr:** 1e-4 | **batch_size:** 1 | **n_iterations:** 10,000 | **labeled_fraction:** 0.1

SSDA counterpart to run3. Uses 10% of labeled tomograms (1 train, 1 val) for the supervised component.

---

## Run 9

**Type:** SL warmup  
**Source checkpoint:** `actin-opto-run1`  
**Data:** `experimental/optogenetics/h5/subvolumes`  
**lr:** 1e-4 | **batch_size:** 2 | **n_iterations:** 1,000

Supervised fine-tuning of the synthetic-trained model on real subvolumes. Teacher warmup for subsequent SSDA run.

---

## Run 10

**Type** SL Synthetic
**Dataset:** `opto_dataset_1` (15 tomograms, 3 conditions)  
**Conditions:** [0, 1, 2]  
**Train:** 36 tomograms (12 per condition)  
**Val:** 9 tomograms (3 per condition)  
**Loss:** DiceLoss (default)  
**lr:** 1e-4 | **batch_size:** 4 | **n_iterations:** 10,000

Baseline supervised run on `opto_dataset_1`.
