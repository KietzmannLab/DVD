# dvd/dvd/helpers/dvd_utils.py
# -*- coding: utf-8 -*-
"""
Internal helpers for the Scale-Free Developmental Visual Diet (DVD).

This file contains:
  - developmental curves
  - scale-free parameter resolution
  - input / resize helpers
  - color development helpers
  - visual acuity filtering
  - contrast sensitivity filtering
  - cached spectral helpers
  - math / FFT utilities
"""

import math
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import torchvision.transforms.functional as TF
import kornia.color as kc
from dvd_scale_free.dvd_scale_free.helpers.csf_barten_1999 import csf_relative_gain


class DVDTransformerBase:
    """
    Internal base class that holds all helper logic.
    DVDTransformer in development.py inherits from this class.
    """

    def __init__(self, config):
        self.cfg = config

        self._ridx_cache: Dict[Tuple[int, int, float, torch.device, torch.dtype], Dict[str, torch.Tensor]] = {}
        self._csf_cache: Dict[Tuple, torch.Tensor] = {}
        self._gray_lab_cache: Dict[Tuple[torch.device, torch.dtype], torch.Tensor] = {}

        self._init_scale_free_defaults()

    # ============================================================
    # Development curves
    # ============================================================
    def get_early_visual_acuity(self, age_months: float) -> float:
        a, b, c, d = (
            18.035945052640425,
            0.7933899743217134,
            1.6012490927401029,
            0.027054604551482078,
        )
        return float(a * np.exp(-b * age_months) + c * np.exp(-d * age_months))

    def get_early_visual_acuity_cpd(self, age_months: float, image_size: int) -> float:
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

    def get_cs_sensitivity_development(
        self,
        age_months: float,
        age50: float = 4.8 * 12,
        n: float = 2.1633375920569247,
    ) -> float:
        y_max = (300 ** n) / (300 ** n + age50 ** n)
        val = (age_months ** n) / (age_months ** n + age50 ** n) / y_max
        return float(np.clip(val, 0.0, 1.0))

    # ============================================================
    # Scale-free helpers
    # ============================================================
    def _init_scale_free_defaults(self) -> None:
        cfg = self.cfg

        ref_W = float(cfg.scale_free_reference_image_size)
        ref_fov = float(cfg.scale_free_reference_fov_deg)
        ref_nyq = 0.5 / (ref_fov / ref_W)

        if cfg.cs_sf_min_frac_nyq is None:
            cfg.cs_sf_min_frac_nyq = max(0.0, float(cfg.cs_sf_min_cpd) / ref_nyq)

        if cfg.cs_sf_max_frac_nyq is None:
            cfg.cs_sf_max_frac_nyq = min(0.98, float(cfg.cs_sf_max_cpd) / ref_nyq)

        if cfg.cs_logsf_min_frac_nyq is None:
            cfg.cs_logsf_min_frac_nyq = max(1e-4, float(cfg.cs_logsf_min_cpd) / ref_nyq)

        if cfg.cs_anchor_half_width_frac_nyq is None and cfg.cs_anchor_half_width_cpd is not None:
            cfg.cs_anchor_half_width_frac_nyq = max(
                1e-4,
                float(cfg.cs_anchor_half_width_cpd) / ref_nyq,
            )

    def _resolve_band_params(self, W: int) -> Tuple[float, float, float]:
        cfg = self.cfg
        nyq = self._nyquist_cpd(W=W, fov_deg=float(cfg.cs_fov_deg))

        if not cfg.use_scale_free_geometry:
            sf_min = float(cfg.cs_sf_min_cpd)
            sf_max = min(float(cfg.cs_sf_max_cpd), 0.98 * nyq)
            logsf_min = float(cfg.cs_logsf_min_cpd)
            return sf_min, sf_max, logsf_min

        sf_min = float(cfg.cs_sf_min_frac_nyq) * nyq
        sf_max = min(float(cfg.cs_sf_max_frac_nyq) * nyq, 0.98 * nyq)
        logsf_min = max(float(cfg.cs_logsf_min_frac_nyq) * nyq, 1e-6)

        sf_min = max(0.0, min(sf_min, sf_max - 1e-6))
        logsf_min = min(logsf_min, max(sf_max * 0.999, 1e-6))

        return sf_min, sf_max, logsf_min

    def _resolve_anchor_halfwidth_bins(self, sf: torch.Tensor, W: int) -> int:
        cfg = self.cfg

        if cfg.use_scale_free_geometry and (cfg.cs_anchor_half_width_frac_nyq is not None):
            nyq = self._nyquist_cpd(W=W, fov_deg=float(cfg.cs_fov_deg))
            half_width_cpd = float(cfg.cs_anchor_half_width_frac_nyq) * nyq
            return self._halfwidth_bins_from_cpd(
                sf=sf,
                half_width_cpd=half_width_cpd,
                fallback_bins=int(cfg.cs_anchor_half_width_bins),
            )

        return self._halfwidth_bins_from_cpd(
            sf=sf,
            half_width_cpd=cfg.cs_anchor_half_width_cpd,
            fallback_bins=int(cfg.cs_anchor_half_width_bins),
        )

    def _resolve_acuity_cutoff_for_grid(self, acuity_cpd: float, W: int) -> float:
        cfg = self.cfg
        nyq = self._nyquist_cpd(W=W, fov_deg=float(cfg.cs_fov_deg))

        if (not cfg.use_scale_free_geometry) or (not cfg.acuity_use_scale_free_cutoff):
            return min(float(acuity_cpd), 0.98 * nyq)

        adult = max(float(cfg.acuity_cpd_at_20_20), 1e-6)
        frac = float(acuity_cpd) / adult
        frac = max(0.0, min(1.0, frac))
        return frac * (0.98 * nyq)

    def _sf_to_csf_eval_axis(self, sf: torch.Tensor, W: int) -> torch.Tensor:
        cfg = self.cfg

        if str(getattr(cfg, "cs_csf_mode", "relative_nyquist")) == "absolute_cpd":
            return sf

        if str(getattr(cfg, "cs_csf_mode", "relative_nyquist")) == "relative_nyquist":
            nyq = self._nyquist_cpd(W=W, fov_deg=float(cfg.cs_fov_deg))
            rel = sf / max(nyq, 1e-6)
            ref_max = float(getattr(cfg, "cs_csf_reference_max_cpd", 30.0))
            return rel * ref_max

        mode = str(getattr(cfg, "cs_csf_mode", "relative_nyquist"))
        raise ValueError(f"Unknown cs_csf_mode={mode}")

    # ============================================================
    # Input / resize helpers
    # ============================================================
    @staticmethod
    def _standardize_input(x: torch.Tensor):
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
    # Color helpers
    # ============================================================
    def _get_gray_lab_constant(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device, dtype)
        if key not in self._gray_lab_cache:
            self._gray_lab_cache[key] = torch.tensor(
                [50.0, 0.0, 0.0], device=device, dtype=dtype
            ).view(1, 3, 1, 1)
        return self._gray_lab_cache[key]

    def _apply_color_development(
        self,
        img: torch.Tensor,
        chromatic_sensitivity: float,
        return_mode: str = "blend",   # "blend" | "gray_base" | "chroma_keep" | "chroma_remove" | "both"
        clamp: bool = True,
    ):
        """
        Color development with optional visualization-friendly decomposition.

        For RGB input, define:
            gray   = grayscale(rgb)
            chroma = rgb - gray

        Developmental color output is:
            blend = gray + a * chroma
        where a = chromatic_sensitivity in [0, 1].

        Return modes:
        - "blend":
            Standard developmental color output.

        - "gray_base":
            Grayscale base only.

        - "chroma_keep":
            Visualization-friendly image showing the chromatic information
            preserved by development. Implemented as:
                gray + a * chroma
            which is identical to blend, but kept as an explicit mode for clarity.

        - "chroma_remove":
            Visualization-friendly image showing the chromatic information
            removed by development. Implemented as:
                gray + (1 - a) * chroma
            This gives a natural-image-like visualization of the removed color content,
            instead of returning the signed residual directly.

        - "both":
            Returns:
                (gray_base, blend)

        Notes:
        - In smooth interpolation mode (apply_threshold_color=False), we have:
            rgb   = gray + chroma
            blend = gray + a * chroma

        - The exact removed signed residual is:
            removed_residual = (1 - a) * chroma
        but that is not returned directly here, because signed residuals do not
        display naturally as RGB images.

        - If you want exact decomposition analysis, compute it separately with:
            removed_residual = rgb - blend
        and avoid treating it as a normal image.
        """
        cfg = self.cfg
        B, C, H, W = img.shape

        if C < 3:
            return img

        orig_dtype = img.dtype
        rgb = img[:, :3].to(torch.float32)

        a = float(max(0.0, min(1.0, chromatic_sensitivity)))

        def _pack_rgb_result(rgb_result: torch.Tensor):
            rgb_result = rgb_result.to(orig_dtype)

            if C == 3:
                return rgb_result.clamp(0.0, 1.0) if clamp else rgb_result

            out = img.clone()
            out[:, :3] = rgb_result
            return out.clamp(0.0, 1.0) if clamp else out

        # ------------------------------------------------------------
        # Threshold mode: only standard blend is well-defined
        # ------------------------------------------------------------
        if cfg.apply_threshold_color:
            if return_mode not in ("blend", "chroma_keep"):
                raise ValueError(
                    'Color decomposition modes other than "blend"/"chroma_keep" '
                    'are only supported when apply_threshold_color=False.'
                )

            rgb_out = self.apply_chromatic_threshold_sensitivity(rgb, a)
            if clamp:
                rgb_out = rgb_out.clamp(0.0, 1.0)

            return _pack_rgb_result(rgb_out)

        # ------------------------------------------------------------
        # Smooth grayscale <-> color interpolation
        # ------------------------------------------------------------
        gray = TF.rgb_to_grayscale(rgb, num_output_channels=3)
        chroma = rgb - gray

        gray_base = gray
        chroma_keep_vis = gray + a * chroma
        chroma_remove_vis = gray + (1.0 - a) * chroma
        blend = chroma_keep_vis

        if clamp:
            gray_base = gray_base.clamp(0.0, 1.0)
            chroma_keep_vis = chroma_keep_vis.clamp(0.0, 1.0)
            chroma_remove_vis = chroma_remove_vis.clamp(0.0, 1.0)
            blend = blend.clamp(0.0, 1.0)

        if return_mode == "blend":
            return _pack_rgb_result(blend)

        if return_mode == "gray_base":
            return _pack_rgb_result(gray_base)

        if return_mode == "chroma_keep":
            return _pack_rgb_result(chroma_keep_vis)

        if return_mode == "chroma_remove":
            return _pack_rgb_result(chroma_remove_vis)

        if return_mode == "both":
            return _pack_rgb_result(gray_base), _pack_rgb_result(blend)

        raise ValueError(f"Unknown return_mode={return_mode}")

    @staticmethod
    def interpolate_color_grayscale(rgb: torch.Tensor, chromatic_sensitivity: float) -> torch.Tensor:
        grayscale = TF.rgb_to_grayscale(rgb, num_output_channels=3)
        a = float(max(0.0, min(1.0, chromatic_sensitivity)))
        return (a * rgb + (1.0 - a) * grayscale).clamp(0.0, 1.0)

    def apply_chromatic_threshold_sensitivity(self, rgb: torch.Tensor, chromatic_sensitivity: float) -> torch.Tensor:
        lab = kc.rgb_to_lab(rgb)
        gray_lab = self._get_gray_lab_constant(rgb.device, lab.dtype)
        delta_e = torch.norm(lab - gray_lab, dim=1, keepdim=True)

        thr = 128.0 * (1.0 - float(max(0.0, min(1.0, chromatic_sensitivity))))
        color_mask = (delta_e > thr).to(rgb.dtype)

        gray_rgb = TF.rgb_to_grayscale(rgb, num_output_channels=3)
        return (color_mask * rgb + (1.0 - color_mask) * gray_rgb).clamp(0.0, 1.0)

    # ============================================================
    # FFT / acuity / contrast helpers
    # ============================================================
    
    # ============================================================
    # Visual acuity helper: frequency-domain low-pass (ALL channels)
    # ============================================================
    def _apply_visual_acuity_freq_cutoff(
        self,
        img: torch.Tensor,
        cutoff_cpd: float,
        return_mode: str = "keep",   # "keep" | "remove" | "both"
        clamp: bool = True,
    ):
        """
        Frequency-domain visual-acuity decomposition for any C.

        Returns:
        - "keep":   low-pass / kept component
        - "remove": complementary removed component
        - "both":   (kept, removed)

        If clamp=False, then kept + removed == original (up to numerical precision).
        """
        cfg = self.cfg
        assert img.dim() == 4, "Expected img [B,C,H,W]"

        B, C, H, W = img.shape
        device, dtype = img.device, img.dtype

        ridx = self._get_radial_indexer_torch(H, W, device, dtype, fov_deg=float(cfg.cs_fov_deg))
        bin_idx = ridx["bin_idx"]
        sf = ridx["sf_centers"]
        b_flat = bin_idx.reshape(-1)

        cutoff_cpd = float(max(0.0, cutoff_cpd))
        tau = float(cfg.acuity_soft_transition_cpd)

        if tau > 0.0:
            keep_bins = self._sigmoid((cutoff_cpd - sf) / tau).to(dtype)
        else:
            keep_bins = (sf <= cutoff_cpd).to(dtype)

        keep_mask2d = keep_bins[b_flat].reshape(1, 1, H, W).expand(B, 1, H, W)

        if cfg.cs_keep_dc:
            cy, cx = H // 2, W // 2
            keep_mask2d[:, :, cy, cx] = 1.0

        remove_mask2d = 1.0 - keep_mask2d

        if cfg.cs_demean_for_fft:
            mu = img.mean(dim=(-2, -1), keepdim=True)
            x = img - mu
        else:
            mu = None
            x = img

        X = self._fft2_shift(x)

        X_keep = X * keep_mask2d.to(X.dtype)
        X_remove = X * remove_mask2d.to(X.dtype)

        x_keep = self._ifft2_ishift(X_keep).real
        x_remove = self._ifft2_ishift(X_remove).real

        if mu is not None:
            x_keep = x_keep + mu
            # removed branch gets no DC/mean part because it is complementary to kept

        if clamp:
            x_keep = x_keep.clamp(0.0, 1.0)
            x_remove = x_remove.clamp(-1.0, 1.0)

        if return_mode == "keep":
            return x_keep
        if return_mode == "remove":
            return x_remove
        if return_mode == "both":
            return x_keep, x_remove

        raise ValueError(f"Unknown return_mode={return_mode}")

    # ============================================================
    # Contrast helper
    # ============================================================
    def _apply_cs_sensitivity_cutoff(
        self,
        img: torch.Tensor,
        keep_frac: float,
        acuity_cpd: Optional[float] = None,
        return_mode: str = "keep",   # "keep" | "remove" | "both"
        clamp: bool = True,
    ):
        """
        Contrast-sensitivity decomposition.

        Returns:
        - "keep":   component preserved by the CS filter
        - "remove": complementary removed component
        - "both":   (kept, removed)

        If clamp=False, then kept + removed == original (up to numerical precision).
        """
        cfg = self.cfg
        assert img.dim() == 4, "Expected img [B,C,H,W]"

        B, C, H, W = img.shape
        device, dtype = img.device, img.dtype
        eps = float(cfg.cs_eps)

        sf_min, sf_max, logsf_min = self._resolve_band_params(W=W)

        ridx = self._get_radial_indexer_torch(H, W, device, dtype, fov_deg=float(cfg.cs_fov_deg))
        bin_idx = ridx["bin_idx"]
        sf = ridx["sf_centers"]
        counts = ridx["counts"].clamp_min(eps)
        nbins = int(ridx["nbins"].item())
        b_flat = bin_idx.reshape(-1)

        # reference luminance spectrum
        if C >= 3:
            y_img = 0.2989 * img[:, 0] + 0.5870 * img[:, 1] + 0.1140 * img[:, 2]
        else:
            y_img = img[:, 0]

        if cfg.cs_demean_for_fft:
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

        use_barten = getattr(cfg, "cs_use_barten_csf", False)
        if use_barten:
            sf_for_csf = self._sf_to_csf_eval_axis(sf, W=W)
            G_1d = self._get_barten_gain_torch(
                sf_for_csf, H=H, W=W, fov_deg=float(cfg.cs_fov_deg), device=device, dtype=dtype
            )
        else:
            G_1d = torch.ones_like(sf, dtype=dtype)

        logG = torch.log10(G_1d.clamp_min(eps)).unsqueeze(0).expand(B, -1)

        band_ok_1d = (sf >= sf_min) & (sf <= sf_max)
        band_ok = band_ok_1d.unsqueeze(0).expand(B, -1)
        band_valid = valid & band_ok
        if not torch.any(band_valid):
            band_valid = valid

        if cfg.cs_band_range_mode == "trim":
            q_lo = float(cfg.cs_trim_low_pct) / 100.0
            q_hi = float(cfg.cs_trim_high_pct) / 100.0

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

        p = float(max(cfg.cs_keep_min, min(cfg.cs_keep_max, keep_frac)))

        if cfg.cs_progress_mode == "linear_amplitude":
            p_eff = self._remap_progress(
                p=p,
                gamma=float(cfg.cs_to_amplitude_gamma),
                remap=str(cfg.cs_progress_remap),
                log_min=float(cfg.cs_logspan_start),
            )
            z_thr = z_min + (1.0 - p_eff) * (z_max - z_min)

        elif cfg.cs_progress_mode == "logsf_to_amplitude":
            sf_prog = self._map_progress_to_logsf(
                p=p,
                sf_min=sf_min,
                sf_max=sf_max,
                gamma=float(cfg.cs_to_amplitude_gamma),
                min_logsf=float(logsf_min),
                remap=str(cfg.cs_progress_remap),
                log_min=float(cfg.cs_logspan_start),
            )

            idx_prog = int(torch.argmin(torch.abs(sf - sf_prog)).item())
            half_bins_prog = self._resolve_anchor_halfwidth_bins(sf=sf, W=W)
            p0 = max(0, idx_prog - half_bins_prog)
            p1 = min(nbins, idx_prog + half_bins_prog + 1)

            z_thr = torch.empty((B,), device=device, dtype=dtype)
            readout_source = (z + logG) if getattr(cfg, "cs_use_csf_for_threshold_readout", False) else z

            for b in range(B):
                zb = readout_source[b, p0:p1]
                vb = valid[b, p0:p1]
                zb = zb[vb & torch.isfinite(zb)]

                if zb.numel() == 0:
                    p_eff = self._remap_progress(
                        p=p,
                        gamma=float(cfg.cs_to_amplitude_gamma),
                        remap=str(cfg.cs_progress_remap),
                        log_min=float(cfg.cs_logspan_start),
                    )
                    z_thr[b] = z_min[b] + (1.0 - p_eff) * (z_max[b] - z_min[b])
                else:
                    z_thr[b] = zb.median()
        else:
            raise ValueError(f"Unknown cs_progress_mode={cfg.cs_progress_mode}")

        score = z + logG

        eligible = valid
        if cfg.cs_restrict_to_band:
            eligible = eligible & band_ok

        if cfg.cs_use_soft_mask:
            tau = float(max(cfg.cs_soft_tau_log10, 1e-6))
            centered = (score - z_thr.unsqueeze(1)) / tau
            keep_bins = torch.zeros_like(score, dtype=dtype)
            keep_bins[eligible] = self._sigmoid(centered[eligible])
        else:
            keep_bins = torch.zeros_like(score, dtype=dtype)
            thr_full = z_thr.unsqueeze(1).expand_as(score)
            keep_bins[eligible] = (score[eligible] > thr_full[eligible]).to(dtype)

        if acuity_cpd is not None:
            cutoff = float(max(0.0, acuity_cpd))
            tau_a = float(cfg.acuity_soft_transition_cpd)

            if tau_a > 0.0:
                acuity_bins = self._sigmoid((cutoff - sf) / tau_a).to(dtype)
            else:
                acuity_bins = (sf <= cutoff).to(dtype)

            keep_bins = keep_bins * acuity_bins.unsqueeze(0)

        keep_mask2d = keep_bins[:, b_flat].reshape(B, 1, H, W)
        if cfg.cs_keep_dc:
            cy, cx = H // 2, W // 2
            keep_mask2d[:, :, cy, cx] = 1.0

        remove_mask2d = 1.0 - keep_mask2d

        if cfg.cs_demean_for_fft:
            mu = img.mean(dim=(-2, -1), keepdim=True)
            x_ctr = img - mu
        else:
            mu = None
            x_ctr = img

        X = self._fft2_shift(x_ctr)

        X_keep = X * keep_mask2d.to(X.dtype)
        X_remove = X * remove_mask2d.to(X.dtype)

        x_keep = self._ifft2_ishift(X_keep).real
        x_remove = self._ifft2_ishift(X_remove).real

        if mu is not None:
            x_keep = x_keep + mu
            # remove stays zero-mean complement

        if clamp:
            x_keep = x_keep.clamp(0.0, 1.0)
            x_remove = x_remove.clamp(-1.0, 1.0)

        if return_mode == "keep":
            return x_keep
        if return_mode == "remove":
            return x_remove
        if return_mode == "both":
            return x_keep, x_remove

        raise ValueError(f"Unknown return_mode={return_mode}")

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
            int(getattr(cfg, "cs_csf_ngrid", 400)), 
            str(getattr(cfg, "cs_csf_mode", "relative_nyquist")),
            float(getattr(cfg, "cs_csf_reference_max_cpd", 30.0)), 
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
            ngrid=int(getattr(cfg, "cs_csf_ngrid", 400)),
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
        fov = float(cfg.cs_fov_deg if fov_deg is None else fov_deg)
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
        p_eff = DVDTransformerBase._remap_progress(
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