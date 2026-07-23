# Persistence length: Deepict

Compare two persistence-length (Lp) estimators on deepict tomograms 00004 and 00012. R²
reports the fit quality of each method.

## Input

Both methods read the instance point clouds written by `predict_actin_instances.py`, one
CSV per tomogram (`<tomogram>_instances.csv`) with columns `ID,X,Y,Z`:

- `ID` is a connected-component label; one filament per `ID`.
- `X,Y,Z` are skeleton graph vertex coordinates in `--pixel_size` units (10 A).
- Rows within an `ID` are ordered along the filament.

`predict_actin_instances.py` builds the clouds from the binary segmentation by teasar
skeletonization, `clean_filament_graph` (split, prune, join), and `connected_components`.

## Methods

Method 1 — port of Bäuerlein et al. (Cell 2017)
(`persistence_length_bauerlein.py`):

- Reference tangent is each filament's middle tangent; correlate it with the tangent at each
  offset along both arms.
- Uses the unsigned acute angle, so `cos θ` stays in [0, 1].
- Fits `log<cos θ>` through the origin (intercept fixed at 0), up to the P90 filament length.

Method 2 — robust version (`pipeline_scripts/analysis/persistence_length.py`):

- All-pairs autocorrelation: average the signed tangent dot product over every point pair at
  each separation.
- The tangent at each point is averaged over `--tangent_window` adjacent points (default 5)
  to reduce aliasing.
- Fits `log<cos θ> = ln(A) - s / Lp` with a free intercept, up to the P90 filament length
  (via `max_lag`).

Both resample to `--sampling_dist` (5 A) and drop filaments shorter than `--min_length`
(350 A = 35 nm).

## Results

`n` is the number of filaments kept after the length cutoff and used in the fit.
Persistence length in µm, with fit R². Method 1 is the through-origin fit (Bäuerlein);
Method 2 is the free-intercept fit.

Cutoff 350 A (35 nm):

| sample | n    | Method 1 Lp | R²    | Method 2 Lp | R²    |
|--------|------|-------------|-------|-------------|-------|
| 00004  | 1301 | 1.84        | -19.8 | 3.41        | 0.953 |
| 00012  | 1834 | 0.69        | -18.1 | 1.00        | 0.987 |

Cutoff 2000 A (200 nm):

| sample | n   | Method 1 Lp | R²   | Method 2 Lp | R²    |
|--------|-----|-------------|------|-------------|-------|
| 00004  | 500 | 2.87        | -6.8 | 2.71        | 0.971 |
| 00012  | 309 | 1.39        | -3.7 | 0.94        | 0.995 |


Both fit plots share axes (log <cos θ> vs distance in nm), so they are directly comparable.

Method 1 (Bäuerlein):

![Method 1 fit](results/persistence_length_bauerlein_deepict.png)

Method 2 (robust):

![Method 2 fit](results/persistence_length_deepict.png)

Filament length distribution:

![Filament length distribution](results/length_distribution_deepict.png)

## Interpretation

Method 2 (robust) fits well at both cutoffs (R² 0.95 to 0.99). Method 1
(Bäuerlein) fits poorly (negative R²); it improves at the 2000 A cutoff but stays below
zero. Forcing the intercept to 0 does not match the measured decay, so the Method 1 Lp is
not trustworthy on this data.

For Method 2, 00004 gives Lp 3.41 µm at 350 A and 2.71 µm at 2000 A, near the reported
stress-fiber value of 3.7 µm. 00012 gives about 0.9 to 1.0 µm. Lp exceeds the longest
filament (about 1.8 µm), so the fit extrapolates beyond the measured range. Method 2 fits
better than Method 1, but check the merged filaments below before trusting the value.

## Merged filaments

`detect_merged_filaments.py` flags instances that `clean_filament_graph` did not split. An
instance whose skeleton graph has a node of degree >= 3 is a branch (3) or crossing (4) that
merges two filaments under one ID. The script reports the graph degree, the maximum turning
angle between consecutive tangents, and `gap_ratio` (the largest step over the median step),
then writes clean CSVs with the suspects removed.

At the 350 A cutoff:

| sample | filaments | suspect | degree 3 | degree 4 |
|--------|-----------|---------|----------|----------|
| 00004  | 1301      | 11      | 8        | 3        |
| 00012  | 1834      | 9       | 0        | 9        |

The two failure modes separate by geometry:

- Degree-4 crossings show a near-180° turn and a large spatial gap (gap_ratio up to 525),
  because the ordered walk jumps to the second, separated filament.
- Degree-3 branches show a moderate turn (76-97°) with a small gap.

Effect on the Method 2 fit (350 A), suspects removed:

| sample | Lp all | R² all | Lp clean | R² clean |
|--------|--------|--------|----------|----------|
| 00004  | 3.41   | 0.95   | 13.94    | 0.77     |
| 00012  | 1.00   | 0.99   | 3.05     | 0.84     |

Only about 1% of instances are merged. They are among the longest (merged length up to about
1800 nm), and their near-180° reversals inject spurious anti-correlation. Removing them
changes Lp several-fold and lowers R². The fit is ill-conditioned: the true Lp is far larger
than the observable filament length, so Lp is not robustly determined. Screen for merged
filaments before trusting any value.

Method 2 fit with suspects removed. The decay is shallow, especially 00004, which barely
drops over 500 nm, so the slope and Lp are weakly constrained.

![Method 2 fit, suspects removed](results/persistence_length_deepict_clean.png)
