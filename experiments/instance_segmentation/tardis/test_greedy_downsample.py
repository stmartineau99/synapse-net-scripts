import time
from pathlib import Path

import mrcfile
import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

from tardis_em.dist_pytorch.utils.build_point_cloud import BuildPointCloud
from tardis_em.dist_pytorch.utils.utils import VoxelDownSampling

ROOT = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/predictions/deepict/csv/instances/tmp")
MRC = ROOT / "00004_gt_mask.mrc"
OUT_PC = ROOT / "00004_gt_points_greedy.npy"
D = 5.0
EROSION = 3


def greedy_min_distance(coord, d):
    """Keep points so that every kept point is >= d from all others (Poisson-disk)."""
    tree = cKDTree(coord)
    alive = np.ones(len(coord), dtype=bool)
    keep = []
    for i in range(len(coord)):
        if not alive[i]:
            continue
        keep.append(i)
        nbrs = tree.query_ball_point(coord[i], d)
        alive[nbrs] = False
        alive[i] = False
    return coord[keep]


def nn_stats(coord, label):
    t = cKDTree(coord)
    d, _ = t.query(coord, k=2)
    nn = d[:, 1]
    print(f"  [{label}] n={len(coord)} NN vox: "
          f"min={nn.min():.2f} p10={np.percentile(nn,10):.2f} median={np.median(nn):.2f} "
          f"mean={nn.mean():.2f} p90={np.percentile(nn,90):.2f} max={nn.max():.2f} | "
          f"frac<3={100*(nn<3).mean():.1f}% frac>8={100*(nn>8).mean():.1f}%")


with mrcfile.open(MRC) as mrc:
    mask = np.asarray(mrc.data) > 0
print(f"mask: shape={mask.shape} nonzero={int(mask.sum())}")

eroded = mask
for _ in range(EROSION):
    eroded = binary_erosion(eroded)
print(f"after erosion x{EROSION}: {int(eroded.sum())} nonzero ({100*eroded.sum()/mask.sum():.1f}% of mask)")

pc_hd = np.asarray(
    BuildPointCloud().build_point_cloud(image=eroded, skeletonize=False)
)[:, -3:].astype(np.float64)
print(f"pc_hd (TARDIS build_point_cloud, skeletonize=False): {len(pc_hd)} points")

baseline = np.asarray(VoxelDownSampling(voxel=5, labels=False, KNN=True)(coord=pc_hd))[:, -3:]
nn_stats(baseline, "baseline VoxelDownSampling(5)")

t0 = time.time()
greedy = greedy_min_distance(pc_hd, D)
print(f"greedy d={D}: {len(greedy)} points kept in {time.time()-t0:.1f}s "
      f"({100*len(greedy)/len(pc_hd):.1f}% of pc_hd)")
nn_stats(greedy, f"greedy d={D}")

np.save(OUT_PC, greedy)
print(f"Saved point cloud to {OUT_PC}")
