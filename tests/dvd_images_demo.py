#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
debug_dvd_demo.py
~~~~~~~~~~~~~~~~~
Visualize how RGB images change across developmental ages (months)
using the current Scale-Free DVDTransformer implementation.

Run:
    python debug_dvd_demo.py
"""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from dvd_scale_free.dvd_scale_free.development import DVDTransformer, DVDConfig


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
AGES: List[int] = [1, 4, 16, 64, 256]
IMG_SIZE: int = 256
USE_GPU: bool = True

CFG = DVDConfig(
    # Global switches
    apply_blur=1,
    apply_color=1,
    apply_contrast=1,
    verbose=False,

    # Whether flip the operation: decomposition of output behavior (Note: blur_mode need to be "freq" not "gaussian" model)
    # normal case:  decomposition_mode="keep", color_decomposition_mode="blend" , blur_mode="gaussian" or  "freq"
    # flip case:  decomposition_mode="remove", color_decomposition_mode="chroma_remove", blur_mode="freq"
    decomposition_mode="keep", # "keep" | "remove" | "both"
    decomposition_clamp=True,
    color_decomposition_mode="blend", # "blend"| "chroma_remove"  | "chroma_keep" | "gray_base" 

    # Input size
    image_size=IMG_SIZE,
    resize_input=True,
    fully_random=False,

    # Contrast sensitivity behavior
    cs_progress_remap="log",               # "none" | "power" | "log"
    cs_progress_mode="logsf_to_amplitude",
    cs_logspan_start=5e-3, # Main hyperparameter in "log" mode (default). Sets the minimum starting point of the log-scale range. Smaller values correspond to higher contrast amplitude threshold, and therefore lower early visual contrast fidelity.
    cs_to_amplitude_gamma=1.0,

    # CPD / viewing geometry
    cs_fov_deg=15.0,
    cs_sf_min_cpd=0.0,
    cs_sf_max_cpd=30.0,
    cs_logsf_min_cpd=0.1,
    cs_anchor_half_width_cpd=0.5,

    # FFT / masking behavior
    cs_restrict_to_band=True,
    cs_demean_for_fft=True,
    cs_keep_dc=True,
    cs_band_range_mode="endpoints",
    cs_anchor_half_width_bins=2,
    cs_trim_low_pct=99.0,
    cs_trim_high_pct=1.0,
    cs_use_soft_mask=True,
    cs_soft_tau_log10=0.10,
    cs_keep_min=1e-4,
    cs_keep_max=1.0,
    cs_eps=1e-12,

    # Scale-free parameterization
    use_scale_free_geometry=True,
    scale_free_reference_image_size=224,
    scale_free_reference_fov_deg=15.0,

    # Visual acuity behavior
    blur_mode="gaussian",                  # "gaussian" | "freq"
    acuity_apply_with_cs=True,
    acuity_cpd_at_20_20=30.0,
    acuity_soft_transition_cpd=1.0,
    acuity_use_scale_free_cutoff=True,

    # Chromatic sensitivity behavior
    apply_threshold_color=False,
)


# ------------------------------------------------------------
# Input / output paths
# ------------------------------------------------------------
ASSETS_DIR = Path("assets/example_stimuli")
RGB_PATHS: List[Path] = [
    ASSETS_DIR / "example_1.jpeg",
    ASSETS_DIR / "example_2.jpeg",
]

OUT_DIR = Path("results/dvd_demo_output")
OUT_PATH = OUT_DIR / (
    f"rgb_dvd_demo_"
    f"{CFG.blur_mode}_"
    f"b{CFG.apply_blur}_c{CFG.apply_color}_cs{CFG.apply_contrast}_"
    f"mode_{CFG.decomposition_mode}_clamp{int(CFG.decomposition_clamp)}_color_decomp{CFG.color_decomposition_mode}"
    f"cslog{CFG.cs_logspan_start:g}_"
    f"size{CFG.image_size}.pdf"
)


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
def make_demo(outfile: Path) -> None:
    device = torch.device("cuda" if (USE_GPU and torch.cuda.is_available()) else "cpu")
    dvdt = DVDTransformer(CFG)

    if CFG.decomposition_mode == "both":
        raise ValueError(
            'debug_dvd_demo.py expects a single image tensor output. '
            'Set decomposition_mode to "keep" or "remove", not "both".'
        )

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
            out = dvdt(x.clone(), months=age)

            ax[i, j].imshow(_vis_rgb(out))
            ax[i, j].axis("off")

            if j == 0:
                ax[i, j].set_ylabel(row_labels[i], fontsize=9)

            if i == 0:
                ax[i, j].set_title(f"{age} mo", fontsize=12)

    caption = (
            f"RGB DVD demo | "
            f"cs band=[{CFG.cs_sf_min_cpd:g}, {CFG.cs_sf_max_cpd:g}] cpd, "
            f"FOV={CFG.cs_fov_deg:g}°, "
            f"{'soft' if CFG.cs_use_soft_mask else 'hard'}(tau={CFG.cs_soft_tau_log10:g}), "
            f"range={CFG.cs_band_range_mode}, "
            f"blur={CFG.blur_mode}, "
            f"b{CFG.apply_blur}_c{CFG.apply_color}_cs{CFG.apply_contrast}, "
            f"cs_logspan_start={CFG.cs_logspan_start:g}"
        )
    fig.suptitle(caption, fontsize=10, y=0.995)

    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=300)
    print(f"Saved {outfile.resolve()}")


if __name__ == "__main__":
    make_demo(OUT_PATH)