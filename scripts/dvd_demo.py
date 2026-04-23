from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from dvd.dvd.development import DVDTransformer, DVDConfig # DVD Data Transformer (main API)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
AGES: List[int] = [1, 4, 16, 64, 256]   # ages in months
IMG_SIZE: int = 256                     # resize target (px)
CFG = DVDConfig() 
# Note: If your input images are not normalized to [0, 1], consider set 'by_percentile=True' in DVDConfig() to percentile-based thresholding, which adapts to the image’s actual intensity distribution.

# Input / output paths
ASSETS_DIR = Path("assets/example_stimuli")
IMAGE_PATHS = [
    ASSETS_DIR / "example_1.jpeg",
    ASSETS_DIR / "example_2.jpeg",
]
OUT_DIR = Path("results/dvd_demo_output")
OUT_PATH = OUT_DIR / "dvd_demo_output.pdf"

# Helper: load an image as [1, 3, H, W] float tensor in [0, 1]
def load_tensor(fp: Path) -> torch.Tensor:
    img = Image.open(fp).convert("RGB")
    img.thumbnail((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.asarray(img).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).float()

# Main: build demo
def make_demo(paths: List[Path], outfile: Path) -> None:
    dvdt = DVDTransformer(CFG)
    tensors = [load_tensor(p) for p in paths]

    rows, cols = len(tensors), len(AGES)
    fig, ax = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

    for r, img_t in enumerate(tensors):
        for c, age in enumerate(AGES):
            out = dvdt(img_t.clone(), months=age)              # DVD data transformation
            vis = out.squeeze(0).permute(1, 2, 0).numpy().clip(0, 1)
            ax[r, c].imshow(vis)
            ax[r, c].axis("off")
            if r == 0:
                ax[r, c].set_title(f"{age} mo", fontsize=12)

    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=300)
    print(f"Saved {outfile.resolve()}")


make_demo(IMAGE_PATHS, OUT_PATH)