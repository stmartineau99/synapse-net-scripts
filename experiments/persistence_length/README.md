# Persistence length: method comparison

Compare two estimators of actin persistence length (Lp) on deepict tomograms 00004 and
00012, using the TARDIS instance point clouds produced by `predict_actin_instances.py`.

## Scripts

- `pipeline_scripts/analysis/persistence_length.py` (production): all-pairs tangent
  autocorrelation. It averages the signed tangent dot product over every point pair at each
  contour separation, then fits `log<cos θ>` through the origin. `Lp = -1 / slope`.
- `persistence_length_bauerlein.py` (this folder, WIP): a port of Bäuerlein et al.
  (Cell 2017). For each filament it correlates the middle tangent with the tangent at each
  offset along both arms. It uses the unsigned acute angle, so `cos θ` stays in [0, 1]. It
  applies the same through-origin fit, restricted to separations below the P90 filament
  length.

Both scripts read `ID,X,Y,Z` CSVs, resample to `--sampling_dist` (5 A), and drop filaments
shorter than `--min_length` (350 A = 35 nm).

## Run

```
python pipeline_scripts/analysis/persistence_length.py \
  --data_dir data/predictions/deepict/instances --pattern "*/*_instances.csv" \
  --results_dir experiments/persistence_length --tag deepict --min_length 350
python experiments/persistence_length/persistence_length_bauerlein.py \
  --data_dir data/predictions/deepict/instances --pattern "*/*_instances.csv" \
  --results_dir experiments/persistence_length --tag deepict --min_length 350
```

## Results (min_length 35 nm)

Persistence length in µm:

| sample | n filaments | all-pairs Lp | Bäuerlein Lp | Bäuerlein R² |
|--------|-------------|--------------|--------------|--------------|
| 00004  | 1301        | 1.24         | 1.84         | -19.8        |
| 00012  | 1834        | 0.47         | 0.69         | -18.1        |

Filament length in nm:

| sample | mean | p90 |
|--------|------|-----|
| 00004  | 225  | 520 |
| 00012  | 130  | 260 |

Plots: `persistence_length_deepict.png`, `persistence_length_bauerlein_deepict.png`,
`length_distribution_deepict.png`, `length_distribution_bauerlein_deepict.png`.

## Interpretation

The Bäuerlein estimator gives a higher Lp than the all-pairs estimator on both tomograms.
The unsigned angle raises `<cos θ>` at each separation. This flattens the decay and
increases the fitted Lp. The strongly negative R² shows that the zero-intercept line fits
this curve poorly, so the higher Lp is not well constrained.

Both estimators stay well below the reported stress-fiber value of 3.7 µm. The measurement
is length-limited. The longest filaments reach about 1.8 µm, which is shorter than a 3.7 µm
Lp, so the tangent correlation decays only a little over the observable range. The limit is
skeleton fragmentation, not the fit.
