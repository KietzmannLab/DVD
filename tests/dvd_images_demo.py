#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Visualize RGB images across DVD developmental ages."""

import argparse
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from dvd_scale_free import DVDTransformer


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
AGES: List[int] = [1, 4, 16, 64, 256]
IMG_SIZE: int = 256


# ------------------------------------------------------------
# Input / output paths
# ------------------------------------------------------------
ASSETS_DIR = Path(__file__).resolve().parents[1] / "assets/example_stimuli"
RGB_PATHS: List[Path] = [
    ASSETS_DIR / "example_1.jpeg",
    ASSETS_DIR / "example_2.jpeg",
]

OUT_DIR = Path("results/dvd_demo_output")


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _pil_resize_square(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), resample=Image.LANCZOS)


def load_tensor_rgb(fp: Path, size: int) -> torch.Tensor:
    """Load RGB image as [1,3,H,W] float in [0,1]."""
    img = Image.open(fp).convert("RGB")
    img = _pil_resize_square(img, size)
    arr = np.asarray(img, dtype=np.float32) / 255.0      # [H,W,3]
    arr = np.transpose(arr, (2, 0, 1))                   # [3,H,W]
    return torch.from_numpy(arr).unsqueeze(0)            # [1,3,H,W]


def _vis_rgb(img_b3hw: torch.Tensor) -> np.ndarray:
    vis = img_b3hw[0].permute(1, 2, 0).detach().cpu().numpy()
    return np.clip(vis, 0.0, 1.0)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
@torch.no_grad()
def make_demo(outfile: Path, device: str) -> None:
    dvdt = DVDTransformer()
    caption = "DVD | V = 1 | Adult anchor: auto"

    items = [load_tensor_rgb(p, IMG_SIZE).to(device) for p in RGB_PATHS]
    row_labels = [p.name for p in RGB_PATHS]

    rows = len(items)
    cols = len(AGES)

    fig, ax = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))

    if rows == 1:
        ax = np.expand_dims(ax, axis=0)
    if cols == 1:
        ax = np.expand_dims(ax, axis=1)

    for i, x in enumerate(items):
        for j, age in enumerate(AGES):
            out = dvdt(x, months=age)

            ax[i, j].imshow(_vis_rgb(out))
            ax[i, j].axis("off")

            if j == 0:
                ax[i, j].set_ylabel(row_labels[i], fontsize=9)

            if i == 0:
                ax[i, j].set_title(f"{age} mo", fontsize=12)

    fig.suptitle(caption, fontsize=10, y=0.995)

    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=300)
    plt.close(fig)
    print(f"Saved {outfile.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    make_demo(args.output or OUT_DIR / "rgb_dvd_demo.pdf", args.device)
