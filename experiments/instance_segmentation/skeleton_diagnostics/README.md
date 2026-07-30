# Skeleton Diagnostics

Data: deepict 00004 labels

- Produced by `example_teasar_diagnostics.py` and `example_kimimaro_diagnostics.py`, which share their
  measurements through `skeleton_stats.py`.
- Centered 200x200x200 crop, which is where the optimization happens.

| parameter | TEASAR | kimimaro |
|---|---|---|
| scale | 0.0 | 0.0 |
| const | 70 A | 140 A |
| tick length | 200 A | 200 A |

## Results

**TEASAR** (raw skeleton -> `remove_ticks`)

| measurement | raw | ticks removed |
|---|---|---|
| n_components | 72 | 71 |
| total length | 16.72 µm | 15.90 µm |
| degree 1 | 340 | 229 |
| degree 3 | 198 | 87 |
| degree 4 | 0 | 0 |
| n_spurs | 110 (55.6%) | 0 |
| real junctions | 22 (11.1%) | 37 (42.5%) |

**kimimaro** (raw skeleton)

| measurement | `fix_branching` on | `fix_branching` off |
|---|---|---|
| n_components | 71 | 71 |
| total length | 16.31 µm | 16.51 µm |
| degree 1 | 283 | 268 |
| degree 3 | 141 | 126 |
| degree 4 | 0 | 0 |
| n_spurs | 67 (47.5%) | 44 (34.9%) |
| real junctions | 21 (14.9%) | 24 (19.0%) |

Definitions

- `n_spurs`: a degree-3 node with at least one dead-end arm at or below 200 A. It goes to 0 after `remove_ticks`.
- `real junctions`: a degree-3 node whose three arms all exceed 200 A, so it is a plausible crossing rather than a skeletonization artifact.
- refer to `skeleton_stats.py` for more details

Notes

- `scale` and `const` mean the same thing in both libraries, but kimimaro invalidates a sphere where
  TEASAR invalidates a cube, so the same radius does not invalidate the same volume. The kimimaro value
  was chosen by visual inspection, not by optimisation, so the table compares each method at its own
  working setting rather than at a shared parameter.
- The degree-3 nodes that survive tick removal without being real junctions have a short arm running to
  another junction rather than to a dead end, which is why pruning cannot reach them.
  `view_failed_junctions.py` visualizes that population.

# TEASAR Fixes

Both are in `source/bioimage-cpp/include/bioimage_cpp/skeleton/teasar.hxx`.

The two fixes (see below for implementation details) are cumulative, and each change should reproduce the numbers from the kimimaro table.

1. reproduce column 1 of the above kimimaro table, `fix_branching` turned on
2. reproduce column 2 of the above kimimaro table, `fix_branching` turned off

Reproduce these numbers on the full volume after testing on the crop: 

| degree 3 | crop | full volume |
|---|---|---|
| TEASAR | 198 | 9633 |
| kimimaro `fix_branching` on | 141 | 6494 |
| kimimaro `fix_branching` off | 126 | 2518 |

**1. Invalidate a sphere instead of a cube.**
  - Lines 398 to 410 loop over the axis aligned box returned by `invalidation_bounds` and invalidate every
    voxel in it, with no distance test.
  - `invalidation_bounds` at line 125 computes only per axis half widths, so the region is a cube whose
    corners reach `sqrt(3)` further than the nominal radius.
  - Adding a radius check inside the innermost loop makes it a sphere, which is what kimimaro does.
  - A few lines inside an existing loop, so much cheaper to try than the second fix.

**2. Give the branch fixing behavior an off switch.**
  - TEASAR always does what kimimaro calls `fix_branching`, with no way to disable it.
  - Line 384 zeroes the path cost along every traced path, `pdrf[voxel] = 0.0`.
  - Line 362 routes each new target to `skeleton_voxels`, the set of already traced voxels, so a new path
    stops as soon as it reaches existing skeleton.
  - Lines 366 to 378 reuse the vertex it lands on, which is where the junction appears.
  - A contact between two separate filaments therefore becomes a branch. Correct for a tree, wrong for a
    dense network of separate filaments.
  - kimimaro's alternative is at `trace.py:154`: build one parental field from the root and extract every
    path by pointer hopping, so paths never snap to each other.
  - This is the larger of the two effects, and a structural change to how paths are traced rather than a
    local edit.

Reproduce with:

```
python example_teasar_diagnostics.py --mask_path 00004_gt_mask.mrc --crop
python example_kimimaro_diagnostics.py --mask_path 00004_gt_mask.mrc --crop --fix_branching
python example_kimimaro_diagnostics.py --mask_path 00004_gt_mask.mrc --crop
```

Drop `--crop` from either diagnostic to measure the full volume instead of the centred 200-voxel box.
