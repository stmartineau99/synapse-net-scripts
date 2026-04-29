from glob import glob
from pathlib import Path
import h5py
import numpy as np
import torch
import torch_em
from torch_em.loss.dice import dice_score
from torch_em.loss.cldice import cldice_score
from synapse_net.inference.actin import segment_actin

def predict_actin(data_paths, model_path):
    model_name = Path(model_path).stem

    dice_scores = []
    cldice_scores = []

    for p in data_paths:
        with h5py.File(p, "r") as f: 
            raw = f["raw"][:]
            gt = f["labels"]["actin"][:]

        seg, pred = segment_actin(raw, model_path, verbose=True, return_predictions=True)
        
        # convert to [1, 1, D, H, W] tensors
        pred_t = torch.from_numpy(pred).float().unsqueeze(0).unsqueeze(0)
        gt_t = torch.from_numpy(gt).float().unsqueeze(0).unsqueeze(0)

        d = dice_score(pred_t, gt_t, invert=False, channelwise=False).item()
        cld = cldice_score(pred_t, gt_t, invert=False).item()

        print(f"{Path(p).stem}: dice_score {d:.4f}, cldice_score {cld:.4f}")

        dice_scores.append(d)
        cldice_scores.append(cld)

        with h5py.File(p, "a") as f:
            f.create_dataset(f"segmentations/{model_name}", data=seg, compression="gzip")

    print("---Summary---")
    print(f"mean dice: {np.mean(dice_scores):.4f}")
    print(f"mean cldice: {np.mean(cldice_scores):.4f}")   

def main():
    DATA_DIR = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/training/deepict_dataset_3/test")
    model_path = "/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/training/out/deepict/run6/checkpoints/actin-deepict-run6"

    data_paths = sorted(DATA_DIR.glob("*.h5"))

    predict_actin(data_paths, model_path)

if __name__ == "__main__":
    main()
