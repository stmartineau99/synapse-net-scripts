# Skeleton benchmark

Fixed input: centred 200^3 crop of `labels/actin` from 00004, as `crop200.h5`. Postprocessing held at
the production setting and never swept, so instance counts describe the regime the pipeline runs. CSVs
in `data/predictions/deepict/csv/instances/teasar/benchmark/results/`.

## Summary

**Sphere invalidation beats cube invalidation.**

- `bioimage-cpp` invalidates an axis-aligned cube while kimimaro invalidates a sphere, and kimimaro's
  own docstring calls the cube the faster approximation.
- Compared at matched total skeleton length, so neither is favoured by having more skeleton, the sphere
  gives far fewer spurious junctions and more usable filaments.
- The sphere also tolerates a larger invalidation radius before it starts clipping neighbours, because
  a cube reaches `sqrt(3)` further into its corners than its nominal radius.

full volume, matched skeleton length

|  | teasar cube | kimimaro sphere |
|---|---|---|
| total skeleton | 313.37 um | 313.02 um |
| degree-3 | 6,526 | 2,518 |
| crossings | 236 | 329 |
| instances > 200 A | 1,470 | 1,849 |

**Tuning the invalidation radius helps marginally but cannot reach the goal.**

- `scale` and `constant` determine the cube radius or "reach"
- Reach is a smooth trade with no knee, raising eliminates fragments but ablates real filaments

| reach | degree-3 | filament length lost |
|---|---|---|
| 42 A | 892 | none |
| 70 A | 198 | 2% |
| 91 A | 114 | 6% |
| 140 A | 38 | 24% |
| 300 A | 3 | 44% |

**kimimaro settings.**

- It completes the full volume in 94 s at `const=140` but stalls at `const=70`, so the documented stall
  is a small-radius problem rather than a scale problem.
- `fix_branching=0` beats the default on both objectives.