import os
from glob import glob
from pathlib import Path

import imageio.v3 as imageio
import napari
from elf.io import open_file


def main(view):
    PARENT_DIR = Path("/projects/extern/nhr/nhr_ni/nim00020/dir.project/sage/data/public/deepict/results")

    data_paths = [str(p) for p in PARENT_DIR.glob("*.h5")]

    model_name = "actin-deepict-run2"

    for ff in data_paths:
        fname = Path(ff).stem

        with open_file(ff, "r") as f:
            raw = f["raw"][:]

        with open_file(ff, "r") as f:
            pred = f["predictions"][f"{model_name}"][:]
            seg = f["segmentations"][f"{model_name}"][:]

        if view:
            v = napari.Viewer()
            v.add_image(raw)
            v.add_image(pred)
            v.add_labels(seg)
            v.title = fname
            napari.run()
        #else:
        #    imageio.imwrite(os.path.join(output_root, f"{fname}.tif"), seg, compression="zlib")


if __name__ == "__main__":

    main(view=True)