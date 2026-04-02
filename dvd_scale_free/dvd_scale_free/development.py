# dvd/dvd/development.py
# -*- coding: utf-8 -*-
"""
Developmental Visual Diet (DVD) simulator | Scale-free version : three-stage pipeline

Supports inputs with C in {1, 3, 4}:
  - C=1  : single-channel (depth map / grayscale). Color stage is a no-op.
  - C=3  : RGB. All stages apply to RGB.
  - C=4  : RGBD. Color stage applies ONLY to first 3 (RGB); blur/acuity + contrast apply to ALL 4.

Accepted shapes:
  - [C,H,W] or [B,C,H,W]  (C ∈ {1,3,4})
  - or a list of those tensors

Expected value range: float in [0,1].

Design notes
------------
This file preserves the original behavior while speeding up the implementation.

Main speed optimizations:
- vectorized FFT filtering across ALL channels
- fewer Python loops in contrast / acuity stages
- reduced repeated tensor allocations
- more aggressive caching of helper tensors / constants
"""

# ============================================================
# Standard library
# ============================================================
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

# ============================================================
# Third-party
# ============================================================
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import kornia.color as kc
import kornia.filters as KF

# ============================================================
# Project
# ============================================================
from dvd_scale_free.dvd_scale_free.helpers.csf_barten_1999 import csf_relative_gain


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
    apply_threshold_color: bool = False

    # ----------------------------
    # Size control
    # ----------------------------
    image_size: int = 224
    resize_input: bool = True
    resize_mode: str = "bilinear"          # "nearest"|"bilinear"|"bicubic"|...
    resize_antialias: bool = True

    fully_random: bool = False
    age_months_curve: Optional[Sequence[float]] = None
    verbose: bool = False

    # ----------------------------
    # Geometry / view
    # ----------------------------
    contrast_fov_deg: float = 15.0

    # ------------------------------------------------------------
    # Key: Developmental progress -> log-SF -> amplitude boundary mapping
    # Key hyper-paramete to tune, others can just stay in default
    # ------------------------------------------------------------
    contrast_progress_mode: str = "logsf_to_amplitude"  # "logsf_to_amplitude" | "linear_amplitude"
    contrast_progress_remap: str = "log"                # "none" | "power" | "log"

    contrast_progress_logspan_start: float = 5e-3  # if mapping in "log" mode (default) : Minimal starting point of the log-scale range; determines the mapping between contrast amplitude in the frequency domain and the spatial domain. Smaller values correspond to higher initial amplitude, thus lower fidelity at begining.
    contrast_to_amplitude_gamma: float = 1.0 # if mapping in "power" mode

    contrast_use_csf_for_threshold_readout: bool = False

    # ------------------------------------------------------------
    # Legacy CPD parameters 
    # ------------------------------------------------------------
    contrast_sf_min_cpd: float = 0.0
    contrast_sf_max_cpd: float = 30.0  # Snellen 20/20 acuity ≈ 30 CPD
    contrast_logsf_min_cpd: float = 0.1 
    contrast_anchor_half_width_cpd: Optional[float] = 0.5

    # ------------------------------------------------------------
    # More scale-free frequency parameterization
    # ------------------------------------------------------------
    use_scale_free_geometry: bool = True
    scale_free_reference_image_size: int = 224
    scale_free_reference_fov_deg: float = 15.0

    contrast_sf_min_frac_nyq: Optional[float] = None
    contrast_sf_max_frac_nyq: Optional[float] = None
    contrast_logsf_min_frac_nyq: Optional[float] = None
    contrast_anchor_half_width_frac_nyq: Optional[float] = None

    # ------------------------------------------------------------
    # Contrast filtering behavior
    # ------------------------------------------------------------
    contrast_restrict_to_band: bool = True
    contrast_demean_for_fft: bool = True
    contrast_keep_dc: bool = True

    contrast_band_range_mode: str = "endpoints"  # "endpoints" or "trim"
    contrast_anchor_half_width_bins: int = 2

    # Keep exactly as original
    contrast_trim_low_pct: float = 99.0
    contrast_trim_high_pct: float = 1.0

    contrast_use_soft_mask: bool = True
    contrast_soft_tau_log10: float = 0.10

    contrast_keep_min: float = 1e-4
    contrast_keep_max: float = 1.0
    contrast_eps: float = 1e-12

    # ------------------------------------------------------------
    # Barten CSF usage
    # ------------------------------------------------------------
    contrast_use_barten_csf: bool = True
    contrast_csf_ngrid: int = 400

    contrast_csf_mode: str = "relative_nyquist"  # "absolute_cpd" | "relative_nyquist"
    contrast_csf_reference_max_cpd: float = 30.0

    # ------------------------------------------------------------
    # Visual acuity filtering
    # ------------------------------------------------------------
    blur_mode: str = "gaussian"  # "gaussian" | "freq"
    acuity_apply_with_contrast: bool = True
    acuity_cpd_at_20_20: float = 30.0
    acuity_soft_transition_cpd: float = 0.5
    acuity_use_scale_free_cutoff: bool = True


# ============================================================
# Main Transformer
# ============================================================
class DVDTransformer:
    """
    DVD Data Transformer (main API)

    Accepts:
      - [C,H,W] or [B,C,H,W] float in [0,1], C in {1,3,4}
      - or a list of such tensors

    Returns same container type/shape as input.
    """

    def __init__(self, config: DVDConfig = DVDConfig()):
        self.cfg = config

        # Cache radial indexing per (H,W,fov_deg,device,dtype)
        self._ridx_cache: Dict[Tuple[int, int, float, torch.device, torch.dtype], Dict[str, torch.Tensor]] = {}

        # Cache CSF gain
        self._csf_cache: Dict[Tuple, torch.Tensor] = {}

        # Small constant cache
        self._gray_lab_cache: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}

        self._init_scale_free_defaults()

    # ============================================================
    # Scale-free helpers
    # ============================================================
    def _init_scale_free_defaults(self) -> None:
        """
        Convert legacy CPD defaults into ratio defaults once, so that
        scale-free mode preserves behavior similar to the old DVD setup.
        """
        cfg = self.cfg

        ref_W = float(cfg.scale_free_reference_image_size)
        ref_fov = float(cfg.scale_free_reference_fov_deg)
        ref_nyq = 0.5 / (ref_fov / ref_W)

        if cfg.contrast_sf_min_frac_nyq is None:
            cfg.contrast_sf_min_frac_nyq = max(0.0, float(cfg.contrast_sf_min_cpd) / ref_nyq)

        if cfg.contrast_sf_max_frac_nyq is None:
            cfg.contrast_sf_max_frac_nyq = min(0.98, float(cfg.contrast_sf_max_cpd) / ref_nyq)

        if cfg.contrast_logsf_min_frac_nyq is None:
            cfg.contrast_logsf_min_frac_nyq = max(1e-4, float(cfg.contrast_logsf_min_cpd) / ref_nyq)

        if cfg.contrast_anchor_half_width_frac_nyq is None and cfg.contrast_anchor_half_width_cpd is not None:
            cfg.contrast_anchor_half_width_frac_nyq = max(
                1e-4,
                float(cfg.contrast_anchor_half_width_cpd) / ref_nyq,
            )

    def _resolve_band_params(self, W: int) -> Tuple[float, float, float]:
        """
        Resolve (sf_min, sf_max, logsf_min) for the current grid.
        Returns values in cycles/deg.
        """
        cfg = self.cfg
        nyq = self._nyquist_cpd(W=W, fov_deg=float(cfg.contrast_fov_deg))

        if not cfg.use_scale_free_geometry:
            sf_min = float(cfg.contrast_sf_min_cpd)
            sf_max = min(float(cfg.contrast_sf_max_cpd), 0.98 * nyq)
            logsf_min = float(cfg.contrast_logsf_min_cpd)
            return sf_min, sf_max, logsf_min

        sf_min = float(cfg.contrast_sf_min_frac_nyq) * nyq
        sf_max = min(float(cfg.contrast_sf_max_frac_nyq) * nyq, 0.98 * nyq)
        logsf_min = max(float(cfg.contrast_logsf_min_frac_nyq) * nyq, 1e-6)

        sf_min = max(0.0, min(sf_min, sf_max - 1e-6))
        logsf_min = min(logsf_min, max(sf_max * 0.999, 1e-6))

        return sf_min, sf_max, logsf_min

    def _resolve_anchor_halfwidth_bins(self, sf: torch.Tensor, W: int) -> int:
        """
        Resolve local readout / endpoint anchor width.
        Prefer fraction-of-Nyquist when scale-free mode is enabled.
        """
        cfg = self.cfg

        if cfg.use_scale_free_geometry and (cfg.contrast_anchor_half_width_frac_nyq is not None):
            nyq = self._nyquist_cpd(W=W, fov_deg=float(cfg.contrast_fov_deg))
            half_width_cpd = float(cfg.contrast_anchor_half_width_frac_nyq) * nyq
            return self._halfwidth_bins_from_cpd(
                sf=sf,
                half_width_cpd=half_width_cpd,
                fallback_bins=int(cfg.contrast_anchor_half_width_bins),
            )

        return self._halfwidth_bins_from_cpd(
            sf=sf,
            half_width_cpd=cfg.contrast_anchor_half_width_cpd,
            fallback_bins=int(cfg.contrast_anchor_half_width_bins),
        )

    def _resolve_acuity_cutoff_for_grid(self, acuity_cpd: float, W: int) -> float:
        """
        Convert developmental acuity into the current grid's cutoff.
        In scale-free mode, use relative fraction of adult acuity and map that
        fraction onto current Nyquist.
        """
        cfg = self.cfg
        nyq = self._nyquist_cpd(W=W, fov_deg=float(cfg.contrast_fov_deg))

        if (not cfg.use_scale_free_geometry) or (not cfg.acuity_use_scale_free_cutoff):
            return min(float(acuity_cpd), 0.98 * nyq)

        adult = max(float(cfg.acuity_cpd_at_20_20), 1e-6)
        frac = float(acuity_cpd) / adult
        frac = max(0.0, min(1.0, frac))
        return frac * (0.98 * nyq)

    def _sf_to_csf_eval_axis(self, sf: torch.Tensor, W: int) -> torch.Tensor:
        """
        Convert current sf axis into the axis used for CSF evaluation.
        """
        cfg = self.cfg

        if cfg.contrast_csf_mode == "absolute_cpd":
            return sf

        if cfg.contrast_csf_mode == "relative_nyquist":
            nyq = self._nyquist_cpd(W=W, fov_deg=float(cfg.contrast_fov_deg))
            rel = sf / max(nyq, 1e-6)
            ref_max = float(cfg.contrast_csf_reference_max_cpd)
            return rel * ref_max

        raise ValueError(f"Unknown contrast_csf_mode={cfg.contrast_csf_mode}")

    # ============================================================
    # Development curves
    # ============================================================
    def get_early_visual_acuity(self, age_months: float) -> float:
        """Returns visual acuity as Gaussian blur sigma for a given age (months)."""
        a, b, c, d = (
            18.035945052640425,
            0.7933899743217134,
            1.6012490927401029,
            0.027054604551482078,
        )
        return float(a * np.exp(-b * age_months) + c * np.exp(-d * age_months))

    def get_early_visual_acuity_cpd(self, age_months: float, image_size: int) -> float:
        """
        Frequency-domain acuity cutoff in cycles/degree (cpd), derived from the same age→sigma curve.
        Uses `image_size` so it stays consistent if you resize inputs.
        """
        cfg = self.cfg
        sigma = float(self.get_early_visual_acuity(age_months)) * (float(image_size) / 224.0)

        denom = 600.0 * (sigma / (4.0 * float(image_size) / 100.0))
        if denom <= 0:
            return float(cfg.acuity_cpd_at_20_20)

        dec = (20.0 / denom) if denom >= 20.0 else 1.0
        dec = float(max(0.0, min(1.0, dec)))

        cpd = float(cfg.acuity_cpd_at_20_20) * dec
        return max(0.0, cpd)

    def get_color_sensitivity(self, age_months: float) -> float:
        """Chromatic sensitivity factor in [0,1] (higher = more color preserved)."""
        color_params = {
            "AverageRGBDevelop": {
                "a": 0.008604133954779169,
                "b": 4.380740053287391e-05,
                "alpha": 0.8807610802743646,
            },
            "min_sensitivity_threshold_ages": (21.2 + 21.1 + 18.8) / 3,
        }

        def T(age: float, a: float, b: float, alpha: float) -> float:
            return a * age ** (-alpha) + b * age ** alpha

        def average_color_sensitivity(age_m: float) -> float:
            age_years = age_m / 12.0
            param = color_params["AverageRGBDevelop"]
            a, b, alpha = param["a"], param["b"], param["alpha"]
            min_age = float(color_params["min_sensitivity_threshold_ages"])
            return (T(min_age, a, b, alpha) / T(age_years, a, b, alpha)) if age_years != 0 else 0.0

        return float(average_color_sensitivity(age_months))

    def get_contrast_sensitivity_development(
        self,
        age_months: float,
        age50: float = 4.8 * 12,
        n: float = 2.1633375920569247,
    ) -> float:
        """
        Returns normalized contrast sensitivity development in [0,1].
        """
        y_max = (300 ** n) / (300 ** n + age50 ** n)
        val = (age_months ** n) / (age_months ** n + age50 ** n) / y_max
        return float(np.clip(val, 0.0, 1.0))

    # ============================================================
    # Public call
    # ============================================================
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

            acuity_cpd_for_contrast: Optional[float] = None

            # --- Visual acuity ---
            if cfg.apply_blur:
                age_use = self._sample_age(age_m)

                if cfg.blur_mode == "gaussian":
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

                    if cfg.apply_contrast and cfg.acuity_apply_with_contrast:
                        acuity_cpd_for_contrast = acuity_cutoff
                    else:
                        img = self._apply_visual_acuity_freq_cutoff(img, cutoff_cpd=acuity_cutoff)

            # --- Contrast sensitivity ---
            if cfg.apply_contrast:
                age_use = self._sample_age(age_m)
                keep_frac = float(self.get_contrast_sensitivity_development(age_use))
                keep_frac = float(max(cfg.contrast_keep_min, min(cfg.contrast_keep_max, keep_frac)))

                img = self._apply_contrast_barten_csf_cutoff(
                    img,
                    keep_frac=keep_frac,
                    acuity_cpd=acuity_cpd_for_contrast,
                )

            # --- Color development ---
            if cfg.apply_color:
                age_use = self._sample_age(age_m)
                chroma = float(self.get_color_sensitivity(age_use))
                img = self._apply_color_development(img, chroma)

            if squeeze:
                img = img.squeeze(0)
            return img

        if isinstance(image, list):
            return [_process_one_tensor(x, float(age_months)) for x in image]
        return _process_one_tensor(image, float(age_months))

    # ============================================================
    # Input / resize helpers
    # ============================================================
    @staticmethod
    def _standardize_input(x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
        """
        Return (img, squeeze) where img is [B,C,H,W] and C in {1,3,4}.
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze = True
        elif x.dim() == 4:
            squeeze = False
        else:
            raise ValueError(f"Expected [C,H,W] or [B,C,H,W], got shape={tuple(x.shape)}")

        C = int(x.shape[1])
        if C not in (1, 3, 4):
            raise ValueError(f"Expected C in {{1,3,4}}, got C={C} for shape={tuple(x.shape)}")

        return x, squeeze

    def _maybe_resize(self, img: torch.Tensor) -> torch.Tensor:
        """
        If cfg.resize_input is True, force img to [B,C,image_size,image_size] for any C.
        """
        cfg = self.cfg
        if not cfg.resize_input:
            return img

        _, _, H, W = img.shape
        tgt = int(cfg.image_size)
        if H == tgt and W == tgt:
            return img

        interpolation = getattr(
            TF.InterpolationMode,
            cfg.resize_mode.upper(),
            TF.InterpolationMode.BILINEAR,
        )

        img = TF.resize(
            img,
            [tgt, tgt],
            interpolation=interpolation,
            antialias=cfg.resize_antialias,
        )
        return img.clamp(0.0, 1.0)

    # ============================================================
    # Color helpers (C=1 no-op; C=3 apply; C=4 apply to RGB only)
    # ============================================================
    def _get_gray_lab_constant(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device, dtype)
        if key not in self._gray_lab_cache:
            self._gray_lab_cache[key] = torch.tensor(
                [50.0, 0.0, 0.0], device=device, dtype=dtype
            ).view(1, 3, 1, 1)
        return self._gray_lab_cache[key]

    def _apply_color_development(self, img: torch.Tensor, chromatic_sensitivity: float) -> torch.Tensor:
        cfg = self.cfg
        B, C, H, W = img.shape
        if C < 3:
            return img

        orig_dtype = img.dtype
        rgb = img[:, :3].to(torch.float32)

        if cfg.apply_threshold_color:
            rgb = self.apply_chromatic_threshold_sensitivity(rgb, chromatic_sensitivity)
        else:
            rgb = self.interpolate_color_grayscale(rgb, chromatic_sensitivity)

        rgb = rgb.to(orig_dtype)

        if C == 3:
            return rgb.clamp(0.0, 1.0)

        out = img.clone()
        out[:, :3] = rgb
        return out.clamp(0.0, 1.0)

    @staticmethod
    def interpolate_color_grayscale(rgb: torch.Tensor, chromatic_sensitivity: float) -> torch.Tensor:
        """Blend RGB with grayscale according to chromatic_sensitivity. Expects rgb: (B,3,H,W)."""
        grayscale = TF.rgb_to_grayscale(rgb, num_output_channels=3)
        a = float(max(0.0, min(1.0, chromatic_sensitivity)))
        return (a * rgb + (1.0 - a) * grayscale).clamp(0.0, 1.0)

    def apply_chromatic_threshold_sensitivity(self, rgb: torch.Tensor, chromatic_sensitivity: float) -> torch.Tensor:
        """
        Chromatic thresholding: ΔE below a threshold goes grayscale.
        rgb: (B,3,H,W) in [0,1]; threshold ~ 128*(1-chromatic_sensitivity).
        """
        lab = kc.rgb_to_lab(rgb)
        gray_lab = self._get_gray_lab_constant(rgb.device, lab.dtype)
        delta_e = torch.norm(lab - gray_lab, dim=1, keepdim=True)

        thr = 128.0 * (1.0 - float(max(0.0, min(1.0, chromatic_sensitivity))))
        color_mask = (delta_e > thr).to(rgb.dtype)

        gray_rgb = TF.rgb_to_grayscale(rgb, num_output_channels=3)
        return (color_mask * rgb + (1.0 - color_mask) * gray_rgb).clamp(0.0, 1.0)

    # ============================================================
    # Visual acuity helper: frequency-domain low-pass (ALL channels)
    # ============================================================
    def _apply_visual_acuity_freq_cutoff(self, img: torch.Tensor, cutoff_cpd: float) -> torch.Tensor:
        """
        Frequency-domain visual-acuity low-pass for any C:
          keep(sf) = 1 for sf <= cutoff_cpd, else 0 (optionally softened).
        """
        cfg = self.cfg
        assert img.dim() == 4, "Expected img [B,C,H,W]"

        B, C, H, W = img.shape
        device, dtype = img.device, img.dtype

        ridx = self._get_radial_indexer_torch(H, W, device, dtype, fov_deg=float(cfg.contrast_fov_deg))
        bin_idx = ridx["bin_idx"]          # [H,W]
        sf = ridx["sf_centers"]            # [nbins]
        b_flat = bin_idx.reshape(-1)

        cutoff_cpd = float(max(0.0, cutoff_cpd))
        tau = float(cfg.acuity_soft_transition_cpd)

        if tau > 0.0:
            acuity_bins = self._sigmoid((cutoff_cpd - sf) / tau).to(dtype)
        else:
            acuity_bins = (sf <= cutoff_cpd).to(dtype)

        mask2d = acuity_bins[b_flat].reshape(1, 1, H, W).expand(B, 1, H, W)

        if cfg.contrast_keep_dc:
            cy, cx = H // 2, W // 2
            mask2d[:, :, cy, cx] = 1.0

        if cfg.contrast_demean_for_fft:
            mu = img.mean(dim=(-2, -1), keepdim=True)
            x = img - mu
        else:
            mu = None
            x = img

        X = self._fft2_shift(x)
        Xf = X * mask2d.to(X.dtype)
        xr = self._ifft2_ishift(Xf).real

        if mu is not None:
            xr = xr + mu

        return xr.clamp(0.0, 1.0)

    # ============================================================
    # Contrast helper
    # ============================================================
    def _apply_contrast_barten_csf_cutoff(
        self,
        img: torch.Tensor,
        keep_frac: float,
        acuity_cpd: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Apply a perceptual-band log-amplitude cutoff with a frequency-dependent threshold
        shaped by adult CSF gain (optional). Optionally merges visual-acuity low-pass
        mask into the same FFT pass.

        img: [B,C,H,W] in [0,1], C in {1,3,4}
        keep_frac: interpreted as developmental progress.
        """
        cfg = self.cfg
        assert img.dim() == 4, "Expected img [B,C,H,W]"

        B, C, H, W = img.shape
        device, dtype = img.device, img.dtype
        eps = float(cfg.contrast_eps)

        sf_min, sf_max, logsf_min = self._resolve_band_params(W=W)

        if sf_min >= sf_max and cfg.verbose:
            print("[DVD][Warn] contrast band invalid after resolution; consider checking scale-free ratios.")

        # ---- Radial indexer
        ridx = self._get_radial_indexer_torch(H, W, device, dtype, fov_deg=float(cfg.contrast_fov_deg))
        bin_idx = ridx["bin_idx"]               # [H,W]
        sf = ridx["sf_centers"]                 # [nbins]
        counts = ridx["counts"].clamp_min(eps)  # [nbins]
        nbins = int(ridx["nbins"].item())
        b_flat = bin_idx.reshape(-1)

        # ------------------------------------------------------------
        # 1) Reference spectrum from luminance if RGB else channel 0
        # ------------------------------------------------------------
        if C >= 3:
            y_img = 0.2989 * img[:, 0] + 0.5870 * img[:, 1] + 0.1140 * img[:, 2]
        else:
            y_img = img[:, 0]

        if cfg.contrast_demean_for_fft:
            y_ctr = y_img - y_img.mean(dim=(-2, -1), keepdim=True)
        else:
            y_ctr = y_img

        Y = self._fft2_shift(y_ctr)
        amp = torch.abs(Y).to(dtype)

        amp_flat = amp.reshape(B, -1)
        scatter_index = b_flat.unsqueeze(0).expand(B, -1)

        sums = torch.zeros((B, nbins), device=device, dtype=dtype)
        sums.scatter_add_(dim=1, index=scatter_index, src=amp_flat)
        amp_mean = sums / counts.unsqueeze(0)

        valid = amp_mean > eps
        z = torch.full_like(amp_mean, float("nan"))
        z[valid] = torch.log10(amp_mean[valid])

        # ------------------------------------------------------------
        # 2) Optional adult CSF gain G(sf)
        # ------------------------------------------------------------
        if cfg.contrast_use_barten_csf:
            sf_for_csf = self._sf_to_csf_eval_axis(sf, W=W)
            G_1d = self._get_barten_gain_torch(
                sf_for_csf,
                H=H,
                W=W,
                fov_deg=float(cfg.contrast_fov_deg),
                device=device,
                dtype=dtype,
            )
        else:
            G_1d = torch.ones_like(sf, dtype=dtype)

        logG = torch.log10(G_1d.clamp_min(eps)).unsqueeze(0).expand(B, -1)

        # ------------------------------------------------------------
        # 3) Compute (z_max, z_min) within band
        # ------------------------------------------------------------
        band_ok_1d = (sf >= sf_min) & (sf <= sf_max)
        band_ok = band_ok_1d.unsqueeze(0).expand(B, -1)
        band_valid = valid & band_ok
        if not torch.any(band_valid):
            band_valid = valid

        if cfg.contrast_band_range_mode == "trim":
            q_lo = float(cfg.contrast_trim_low_pct) / 100.0
            q_hi = float(cfg.contrast_trim_high_pct) / 100.0

            z_max = torch.empty((B,), device=device, dtype=dtype)
            z_min = torch.empty((B,), device=device, dtype=dtype)

            for b in range(B):
                zb = z[b].masked_select(band_valid[b])
                zb = zb[torch.isfinite(zb)]
                if zb.numel() == 0:
                    z_max[b] = 0.0
                    z_min[b] = 0.0
                else:
                    z_max[b] = torch.quantile(zb, q_lo)
                    z_min[b] = torch.quantile(zb, q_hi)

        else:
            half_bins = self._resolve_anchor_halfwidth_bins(sf=sf, W=W)

            idx_lo = int(torch.argmin(torch.abs(sf - sf_min)).item())
            idx_hi = int(torch.argmin(torch.abs(sf - sf_max)).item())

            lo0, lo1 = max(0, idx_lo - half_bins), min(nbins, idx_lo + half_bins + 1)
            hi0, hi1 = max(0, idx_hi - half_bins), min(nbins, idx_hi + half_bins + 1)

            z_max = torch.empty((B,), device=device, dtype=dtype)
            z_min = torch.empty((B,), device=device, dtype=dtype)

            for b in range(B):
                z_lo = z[b, lo0:lo1]
                z_hi = z[b, hi0:hi1]

                z_lo = z_lo[torch.isfinite(z_lo)]
                z_hi = z_hi[torch.isfinite(z_hi)]

                if z_lo.numel() == 0 or z_hi.numel() == 0:
                    zb = z[b].masked_select(band_valid[b])
                    zb = zb[torch.isfinite(zb)]
                    med = zb.median() if zb.numel() > 0 else torch.tensor(0.0, device=device, dtype=dtype)
                    z_lo_med = med
                    z_hi_med = med
                else:
                    z_lo_med = z_lo.median()
                    z_hi_med = z_hi.median()

                z_max[b] = torch.maximum(z_lo_med, z_hi_med)
                z_min[b] = torch.minimum(z_lo_med, z_hi_med)

        # ------------------------------------------------------------
        # 4) Map developmental progress -> scalar z_thr
        # ------------------------------------------------------------
        p = float(max(cfg.contrast_keep_min, min(cfg.contrast_keep_max, keep_frac)))

        if cfg.contrast_progress_mode == "linear_amplitude":
            p_eff = self._remap_progress(
                p=p,
                gamma=float(cfg.contrast_to_amplitude_gamma),
                remap=str(cfg.contrast_progress_remap),
                log_min=float(cfg.contrast_progress_logspan_start),
            )
            z_thr = z_min + (1.0 - p_eff) * (z_max - z_min)

        elif cfg.contrast_progress_mode == "logsf_to_amplitude":
            sf_prog = self._map_progress_to_logsf(
                p=p,
                sf_min=sf_min,
                sf_max=sf_max,
                gamma=float(cfg.contrast_to_amplitude_gamma),
                min_logsf=float(logsf_min),
                remap=str(cfg.contrast_progress_remap),
                log_min=float(cfg.contrast_progress_logspan_start),
            )

            idx_prog = int(torch.argmin(torch.abs(sf - sf_prog)).item())
            half_bins_prog = self._resolve_anchor_halfwidth_bins(sf=sf, W=W)
            p0 = max(0, idx_prog - half_bins_prog)
            p1 = min(nbins, idx_prog + half_bins_prog + 1)

            z_thr = torch.empty((B,), device=device, dtype=dtype)
            readout_source = (z + logG) if cfg.contrast_use_csf_for_threshold_readout else z

            for b in range(B):
                zb = readout_source[b, p0:p1]
                vb = valid[b, p0:p1]
                zb = zb[vb & torch.isfinite(zb)]

                if zb.numel() == 0:
                    p_eff = self._remap_progress(
                        p=p,
                        gamma=float(cfg.contrast_to_amplitude_gamma),
                        remap=str(cfg.contrast_progress_remap),
                        log_min=float(cfg.contrast_progress_logspan_start),
                    )
                    z_thr[b] = z_min[b] + (1.0 - p_eff) * (z_max[b] - z_min[b])
                else:
                    z_thr[b] = zb.median()

        else:
            raise ValueError(
                f"Unknown contrast_progress_mode={cfg.contrast_progress_mode}. "
                f"Expected 'linear_amplitude' or 'logsf_to_amplitude'."
            )

        # ------------------------------------------------------------
        # 5) Build radial mask bins
        # ------------------------------------------------------------
        score = z + logG

        eligible = valid
        if cfg.contrast_restrict_to_band:
            eligible = eligible & band_ok

        if cfg.contrast_use_soft_mask:
            tau = float(max(cfg.contrast_soft_tau_log10, 1e-6))
            centered = (score - z_thr.unsqueeze(1)) / tau
            mask_bins = torch.zeros_like(score, dtype=dtype)
            mask_bins[eligible] = self._sigmoid(centered[eligible])
        else:
            mask_bins = torch.zeros_like(score, dtype=dtype)
            thr_full = z_thr.unsqueeze(1).expand_as(score)
            mask_bins[eligible] = (score[eligible] > thr_full[eligible]).to(dtype)

        # ------------------------------------------------------------
        # 5b) Merge visual acuity low-pass (optional)
        # ------------------------------------------------------------
        if acuity_cpd is not None:
            cutoff = float(max(0.0, acuity_cpd))
            tau_a = float(cfg.acuity_soft_transition_cpd)

            if tau_a > 0.0:
                acuity_bins = self._sigmoid((cutoff - sf) / tau_a).to(dtype)
            else:
                acuity_bins = (sf <= cutoff).to(dtype)

            mask_bins = mask_bins * acuity_bins.unsqueeze(0)

        # ------------------------------------------------------------
        # 6) Expand radial mask to 2D FFT mask and apply to ALL channels
        # ------------------------------------------------------------
        mask2d = mask_bins[:, b_flat].reshape(B, 1, H, W)

        if cfg.contrast_keep_dc:
            cy, cx = H // 2, W // 2
            mask2d[:, :, cy, cx] = 1.0

        if cfg.contrast_demean_for_fft:
            mu = img.mean(dim=(-2, -1), keepdim=True)
            x_ctr = img - mu
        else:
            mu = None
            x_ctr = img

        X = self._fft2_shift(x_ctr)
        Xf = X * mask2d.to(X.dtype)
        x_rec = self._ifft2_ishift(Xf).real

        if mu is not None:
            x_rec = x_rec + mu

        return x_rec.clamp(0.0, 1.0)

    # ============================================================
    # Cached helpers: radial indexer + CSF gain
    # ============================================================
    def _get_barten_gain_torch(
        self,
        sf: torch.Tensor,
        H: int,
        W: int,
        fov_deg: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Return G(sf) in [0,1] for the given sf axis, cached."""
        cfg = self.cfg

        key = (
            int(H),
            int(W),
            float(fov_deg),
            int(sf.numel()),
            float(sf[0].item()) if sf.numel() > 0 else 0.0,
            float(sf[-1].item()) if sf.numel() > 0 else 0.0,
            int(cfg.contrast_csf_ngrid),
            str(cfg.contrast_csf_mode),
            float(cfg.contrast_csf_reference_max_cpd),
            device,
            dtype,
        )
        if key in self._csf_cache:
            return self._csf_cache[key]

        sf_cpu = sf.detach().float().cpu().numpy()
        u_min = float(np.min(sf_cpu)) if sf_cpu.size > 0 else 0.0
        u_max = float(np.max(sf_cpu)) if sf_cpu.size > 0 else 1.0

        G = csf_relative_gain(
            sf_cpu,
            u_min=u_min,
            u_max=u_max,
            ngrid=int(cfg.contrast_csf_ngrid),
        ).astype(np.float32)

        G_t = torch.from_numpy(G).to(device=device, dtype=dtype)
        self._csf_cache[key] = G_t
        return G_t

    def _get_radial_indexer_torch(
        self,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype,
        fov_deg: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Build and cache:
          - bin_idx:    [H,W] long, radial-bin index per shifted-FFT bin
          - sf_centers: [nbins] float, spatial frequencies (cycles/deg)
          - counts:     [nbins] float, number of pixels per bin
        """
        cfg = self.cfg
        fov = float(cfg.contrast_fov_deg if fov_deg is None else fov_deg)
        key = (int(H), int(W), float(fov), device, dtype)
        if key in self._ridx_cache:
            return self._ridx_cache[key]

        nbins = min(H, W) // 2

        dx = fov / float(W)
        dy = fov / float(H)

        fy = torch.fft.fftfreq(H, d=dy, device=device, dtype=dtype)
        fx = torch.fft.fftfreq(W, d=dx, device=device, dtype=dtype)
        fy = torch.fft.fftshift(fy)
        fx = torch.fft.fftshift(fx)

        FY, FX = torch.meshgrid(fy, fx, indexing="ij")
        rr = torch.sqrt(FX * FX + FY * FY)

        r_max = float(rr.max().item())
        edges = torch.linspace(0.0, r_max + 1e-12, nbins + 1, device=device, dtype=dtype)

        bin_idx = torch.bucketize(rr.reshape(-1), edges, right=False) - 1
        bin_idx = bin_idx.clamp_(0, nbins - 1).reshape(H, W).to(torch.long)

        sf_centers = 0.5 * (edges[:-1] + edges[1:])
        counts = torch.bincount(bin_idx.reshape(-1), minlength=nbins).to(dtype)

        out = {
            "bin_idx": bin_idx,
            "sf_centers": sf_centers,
            "counts": counts,
            "nbins": torch.tensor(nbins, device=device),
        }
        self._ridx_cache[key] = out
        return out

    # ============================================================
    # Small math helpers
    # ============================================================
    @staticmethod
    def _nyquist_cpd(W: int, fov_deg: float) -> float:
        return 0.5 / (float(fov_deg) / float(W))

    @staticmethod
    def _halfwidth_bins_from_cpd(
        sf: torch.Tensor,
        half_width_cpd: Optional[float],
        fallback_bins: int,
    ) -> int:
        """
        Convert +/- half_width_cpd into approx +/- N bins based on sf sampling density.
        If half_width_cpd is None, fall back to fixed bins.
        """
        if half_width_cpd is None:
            return max(1, int(fallback_bins))

        sfv = sf[torch.isfinite(sf)]
        if sfv.numel() < 3:
            return max(1, int(fallback_bins))

        ds = sfv[1:] - sfv[:-1]
        ds = ds[torch.isfinite(ds) & (ds > 0)]
        if ds.numel() == 0:
            return max(1, int(fallback_bins))

        ds_med = float(ds.median().item())
        return max(1, int(round(float(half_width_cpd) / ds_med)))

    @staticmethod
    def _remap_progress(
        p: float,
        gamma: float = 1.0,
        remap: str = "log",
        log_min: float = 0.01,
    ) -> float:
        """
        Remap developmental progress p in [0,1] to p_eff in [0,1] or [m,1]
        depending on the chosen rule.
        """
        p = float(max(0.0, min(1.0, p)))

        if remap == "none":
            return p

        if remap == "power":
            return p ** float(gamma)

        if remap == "log":
            m = float(max(1e-6, min(log_min, 1.0)))
            return 10.0 ** (math.log10(m) * (1.0 - p))

        raise ValueError(f"Unknown remap={remap}")

    @staticmethod
    def _map_progress_to_logsf(
        p: float,
        sf_min: float,
        sf_max: float,
        gamma: float = 1.0,
        min_logsf: float = 0.1,
        remap: str = "log",
        log_min: float = 0.01,
    ) -> float:
        """
        Map developmental progress p in [0,1] onto a spatial-frequency location
        using interpolation in log-SF space.
        """
        p_eff = DVDTransformer._remap_progress(
            p=p,
            gamma=gamma,
            remap=remap,
            log_min=log_min,
        )

        sf_lo = float(sf_min) if float(sf_min) > 0.0 else float(min_logsf)
        sf_hi = max(float(sf_max), sf_lo + 1e-6)

        log_lo = math.log10(sf_lo)
        log_hi = math.log10(sf_hi)

        log_sf = log_lo + p_eff * (log_hi - log_lo)
        return float(10.0 ** log_sf)

    # ============================================================
    # FFT helpers
    # ============================================================
    @staticmethod
    def _fft2_shift(x: torch.Tensor) -> torch.Tensor:
        """Centered FFT2: fftshift(fft2(x))."""
        return torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1)), dim=(-2, -1))

    @staticmethod
    def _ifft2_ishift(X: torch.Tensor) -> torch.Tensor:
        """Inverse centered FFT2: ifft2(ifftshift(X))."""
        return torch.fft.ifft2(torch.fft.ifftshift(X, dim=(-2, -1)), dim=(-2, -1))

    @staticmethod
    def _sigmoid(x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x)


# ============================================================
# Utilities
# ============================================================
def generate_age_months_curve(
    total_epochs: int,
    len_train_loader: int,
    months_per_epoch: float,
    shuffle: bool = False,
    seed: Optional[int] = None,
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