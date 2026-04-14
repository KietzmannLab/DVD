# -*- coding: utf-8 -*-
"""
Developmental Visual Diet (DVD) simulator | Scale-free version : core API

This file intentionally keeps only the main user-facing logic:
  - DVDTransformer
  - _apply_transformations
  - generate_age_months_curve

All implementation details are moved into helpers/dvd_utils.py
"""

import random
import torch
import kornia.filters as KF
from dataclasses import dataclass
from typing import List, Union, Optional, Sequence
from dvd_scale_free.dvd_scale_free.helpers.dvd_utils import DVDTransformerBase

# ============================================================
# Configuration
# ============================================================
@dataclass
class DVDConfig:
    """Configuration for DVDTransformer."""

    # ----------------------------
    # Global switches
    # ----------------------------
    apply_blur: int = 1
    apply_color: int = 1
    apply_contrast: int = 1
    verbose: bool = False

    # ----------------------------
    # Size & order control
    # ----------------------------
    image_size: int = 224
    resize_input: bool = True
    resize_mode: str = "bilinear"
    resize_antialias: bool = True

    fully_random: bool = False
    age_months_curve: Optional[Sequence[float]] = None

    # ------------------------------------------------------------
    # Contrast sensitivity (cs) behavior
    # ------------------------------------------------------------
    # Mapping developmental contrast sensitivity to the contrast-amplitude threshold in the frequency domain:
    # This controls how developmental contrast sensitivity is translated into how much contrast information is kept over training.
    # In practice, the main hyperparameter worth tuning is usually `cs_logspan_start`;
    # most other settings can stay at their defaults.
    cs_progress_remap: str = "log"                 # "none" | "power" | "log"
    cs_progress_mode: str = "logsf_to_amplitude"

    cs_logspan_start: float = 5e-3                 # Main hyperparameter in "log" mode (default). Sets the minimum starting point of the log-scale range. Smaller values correspond to higher contrast amplitude threshold, and therefore lower early visual contrast fidelity.
    cs_to_amplitude_gamma: float = 1.0             # Only used in "power" mode; usually can be left at the default.

    # CPD / viewing geometry
    cs_fov_deg: float = 15.0
    cs_sf_min_cpd: float = 0.0
    cs_sf_max_cpd: float = 30.0
    cs_logsf_min_cpd: float = 0.1
    cs_anchor_half_width_cpd: Optional[float] = 0.5

    # FFT / masking behavior
    cs_restrict_to_band: bool = True
    cs_demean_for_fft: bool = True
    cs_keep_dc: bool = True

    cs_band_range_mode: str = "endpoints"
    cs_anchor_half_width_bins: int = 2

    cs_trim_low_pct: float = 99.0
    cs_trim_high_pct: float = 1.0

    cs_use_soft_mask: bool = True
    cs_soft_tau_log10: float = 0.10

    cs_keep_min: float = 1e-4
    cs_keep_max: float = 1.0
    cs_eps: float = 1e-12

    # More scale-free frequency parameterization
    use_scale_free_geometry: bool = True
    scale_free_reference_image_size: int = 224
    scale_free_reference_fov_deg: float = 15.0

    cs_sf_min_frac_nyq: Optional[float] = None
    cs_sf_max_frac_nyq: Optional[float] = None
    cs_logsf_min_frac_nyq: Optional[float] = None
    cs_anchor_half_width_frac_nyq: Optional[float] = None

    # ------------------------------------------------------------
    # Visual acuity behavior
    # ------------------------------------------------------------
    blur_mode: str = "gaussian" # "gaussian" or "freq"
    acuity_apply_with_cs: bool = True
    acuity_cpd_at_20_20: float = 30.0
    acuity_soft_transition_cpd: float = 0.5
    acuity_use_scale_free_cutoff: bool = True

    # ------------------------------------------------------------
    # Chromatic sensitivity behavior
    # ------------------------------------------------------------
    apply_threshold_color: bool = False

    # ------------------------------------------------------------
    # Cutoff decomposition controls
    # ------------------------------------------------------------
    # Mainly for visualisation and analysis.
    # Controls whether to return the information preserved by the cutoff ("keep")
    # or the complementary information filtered out ("remove").
    # This is used for contrast sensitivity and only for frequency-domain visual acuity (i.e. when blur_mode is not "gaussian").
    decomposition_mode: str = "keep"   # "keep" | "remove" | "both"
    decomposition_clamp: bool = True   # Whether to clamp outputs after filtering. For ANN training, clamping is usually preferred to keep values in a valid image range. For exact decomposition analysis, set this to False so that: kept + removed == original  (up to numerical precision)
    color_decomposition_mode: str = "blend" # "blend" | "gray_base" | "chroma_keep" | "chroma_remove" -> color decomposition is different from contrast sensitivty & visual acuity


# ============================================================
# DVD Transformation
# ============================================================
class DVDTransformer(DVDTransformerBase):
    """
    DVD Data Transformer (main API)

    Accepts:
      - [C,H,W] or [B,C,H,W] float in [0,1], C in {1,3,4}
      - or a list of such tensors

    Returns same container type/shape as input.
    """

    def __init__(self, config: Optional[DVDConfig] = None):
        super().__init__(config if config is not None else DVDConfig())

    def __call__(self, image: Union[torch.Tensor, List[torch.Tensor]], months: int):
        return self._apply_transformations(image, months)

    # ============================================================
    # Core transformation
    # ============================================================
    def _sample_age(self, age_m: float) -> float:
        cfg = self.cfg
        if cfg.fully_random and cfg.age_months_curve:
            return float(random.choice(cfg.age_months_curve))
        return float(age_m)

    def _apply_transformations(self, image: Union[torch.Tensor, List[torch.Tensor]], age_months: int):
        cfg = self.cfg

        def _process_one_tensor(img_in: torch.Tensor, age_m: float) -> torch.Tensor:
            img, squeeze = self._standardize_input(img_in)
            img = self._maybe_resize(img)

            if cfg.verbose:
                B, C, H, W = img.shape
                print(f"[DVD] B={B} C={C} {H}x{W} mean={img.mean().item():.4f} std={img.std().item():.4f}")

            acuity_cpd_for_cs = None

            # --- Visual acuity ---
            if cfg.apply_blur:
                age_use = self._sample_age(age_m)

                if cfg.blur_mode == "gaussian":
                    assert cfg.decomposition_mode not in ["remove", "both"], ('decomposition_mode="remove" or "both" for visual acuity is only supported in frequency-domain mode, so blur_mode must not be "gaussian".')

                    _, _, _, W = img.shape
                    blur_sigma = float(self.get_early_visual_acuity(age_use)) * (float(W) / 224.0)
                    if blur_sigma > 0:
                        k = int(8 * blur_sigma)
                        kernel_size = k + (1 if k % 2 == 0 else 0)
                        img = KF.gaussian_blur2d(
                            img,
                            (kernel_size, kernel_size),
                            (blur_sigma, blur_sigma),
                            border_type="reflect",
                        )
                else:
                    _, _, _, W = img.shape
                    acuity_cpd_raw = self.get_early_visual_acuity_cpd(age_use, image_size=W)
                    acuity_cutoff = self._resolve_acuity_cutoff_for_grid(acuity_cpd_raw, W=W)

                    if cfg.apply_contrast and cfg.acuity_apply_with_cs:
                        acuity_cpd_for_cs = acuity_cutoff
                    else:
                        img = self._apply_visual_acuity_freq_cutoff(img, cutoff_cpd=acuity_cutoff, return_mode=cfg.decomposition_mode, clamp=cfg.decomposition_clamp)

            # --- Contrast sensitivity ---
            if cfg.apply_contrast:
                age_use = self._sample_age(age_m)
                keep_frac = float(self.get_cs_sensitivity_development(age_use))
                keep_frac = float(max(cfg.cs_keep_min, min(cfg.cs_keep_max, keep_frac)))

                img = self._apply_cs_sensitivity_cutoff(
                    img,
                    keep_frac=keep_frac,
                    acuity_cpd=acuity_cpd_for_cs,
                    return_mode=cfg.decomposition_mode, 
                    clamp=cfg.decomposition_clamp,
                )

            # --- Color development ---
            if cfg.apply_color:
                age_use = self._sample_age(age_m)
                chroma = float(self.get_color_sensitivity(age_use))
                img = self._apply_color_development(img, chroma, return_mode=cfg.color_decomposition_mode)

            if squeeze:
                img = img.squeeze(0)
            return img

        if isinstance(image, list):
            return [_process_one_tensor(x, float(age_months)) for x in image]
        return _process_one_tensor(image, float(age_months))


# ============================================================
# Utilities
# ============================================================
def generate_age_months_curve(
    total_epochs: int,
    len_train_loader: int,
    months_per_epoch: float,
    shuffle: bool = False,
    seed: int = None,
    mid_phase: bool = False,
) -> List[float]:
    """
    Generate the sequence of age_months based on epochs and number of batches.
    """
    curve = [
        ep * months_per_epoch + b * months_per_epoch / len_train_loader
        for ep in range(total_epochs)
        for b in range(len_train_loader)
    ]

    if mid_phase:
        first = sorted(curve[::2], reverse=True)
        second = curve[1::2]
        curve = first + second

    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(curve)

    return curve