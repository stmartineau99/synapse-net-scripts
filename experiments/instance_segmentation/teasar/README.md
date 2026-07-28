# TEASAR Graph Cleaning Parameters

What was learned about each `clean_filament_graph` parameter. Numbers are for tomogram 00004 
of `deepict`, 10 A pixel size, TEASAR skeleton of 237,275 nodes.

## direction_span

Current 10. Tangent vector smoothing distance. More accurate angle measurements result in 
better splitting. Stable values are 8-10. 
See `00004_labels_actin_skeleton_span_sweep.csv`.

## min_branch_angle

Current 20, lowered from 30, because many real branches were found in the 20-25 range.

0 is not an off switch. The comparison rejects below the threshold, so 0 separates every
degree-3 junction. 181 does disable the split, which makes a useful control.

That control shows the degree-3 split causes 975 of the 1,742 fragments under 200 A at
`tick_length=50`.

## tick_length

Raised to 200 A, because 50 A produced too many fragments.

| tick_length (A) | degree-3 nodes | instances | fragments <= 200 A | filaments > 200 A |
|---|---|---|---|---|
| 0 | 9633 | 9596 | 7854 | 1742 |
| 25 | 8228 | 9505 | 7790 | 1715 |
| 50 | 2084 | 3356 | 1742 | 1614 |
| 75 | 1815 | 3086 | 1506 | 1580 |
| 100 | 1718 | 2988 | 1411 | 1577 |
| 150 | 1375 | 2643 | 1105 | 1538 |
| 200 | 1253 | 2521 | 1010 | 1511 |
| 300 | 1106 | 2374 | 962 | 1412 |

The falling `> 200 A` count is merging, not loss: total contour length in those filaments
stays at 298 um across the whole sweep (`length_in_real`), so raising the threshold yields
fewer and longer filaments.

A floor of 767 pre-existing fragments is untouchable at any value; those are left to a
downstream length filter.

## join_dist and min_join_angle

Current 50 and 175. No measured change so far.

## min_through_angle

Removed. It was already depcrecated, so every degree-4 crossing always split.