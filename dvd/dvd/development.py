"""development.py
======================================
    * `DevelopmentVisualDietTransformer` – apply age-adaptive developmental visual diet (DVD) augmentations
    * `generate_age_months_curve` – helper that maps (epoch, batch) → age (months)

"""
from __future__ import annotations

import random
import numpy as np
import torch
from typing import  List, Sequence, Union

# Third-party vision libraries -------------------------------------------------
import kornia.color as kc
import kornia.filters as KF

__all__ = [
    "DVDTransformer",
    "generate_age_months_curve",
]

# ---------------------------------------------------------------------------
# Helper class ----------------------------------------------------------------
# ---------------------------------------------------------------------------
class DVDTransformer:
    """Apply physiologically-motivated augmentations mimicking early vision.

    All public methods mirror the original interface – only formatting,
    type-hinting and commentary were improved for readability.
    """

    # ---------------------------------------------------------------------
    # developmental visual diet (DVD) across 3 aspects
    # ---------------------------------------------------------------------
    @staticmethod
    def get_early_visual_acuity(age_months: float) -> float:
        """fit of infant visual acuity maturation over ages (months).

        The parameters stem from *Banks & Salapatek, 1981* and are left
        untouched to guarantee identical numerical behaviour.
        """
        a, b, c, d = (
            18.035945052640425,
            0.7933899743217134,
            1.6012490927401029,
            0.027054604551482078,
        )
        return a * np.exp(-b * age_months) + c * np.exp(-d * age_months)

    @staticmethod
    def get_contrast_sensitivity_development(
        age_months: float,
        *,
        age50: float = 4.8 * 12,
        n: float = 2.1633375920569247,
    ) -> float:
        """ fit of contrast sensitivity maturation over ages (months).

        Args:
            age_months: Age in months.
            age50: Age at which 50 % of maximum sensitivity is reached.
            n: Steepness parameter.
        """
        y_max = (300 ** n) / (300**n + age50**n)  # 25 years ⇒ asymptote
        return (age_months**n) / (age_months**n + age50**n) / y_max

    @staticmethod
    def get_chromatic_sensitivity(age_months: float) -> float:
        """Normalised chromatic sensitivity curve *[0-1]* based on age in months."""
        params = {
            "a": 0.008604133954779169,
            "b": 4.380740053287391e-05,
            "alpha": 0.8807610802743646,
            "min_age": 20.37 / 12,  # mean of three threshold ages (months → years)
        }

        def t(x: float) -> float:
            return params["a"] * x ** (-params["alpha"]) + params["b"] * x ** params["alpha"]

        age_years = age_months / 12
        return t(params["min_age"]) / t(age_years) if age_years > 0 else 0.0
    
    # ------------------------------------------------------------------
    # Augmentation pipeline -------------------------------------------
    # ------------------------------------------------------------------
    def apply_fft_transformations(
        self,
        image: Union[torch.Tensor, Sequence[torch.Tensor]],
        age_months: float,
        *,
        apply_blur: bool = True,
        apply_color: bool = True,
        apply_contrast: bool = True,
        contrast_threshold: float = 0.2,
        apply_threshold_color: bool = False,
        image_size: int = 224,
        fully_random: bool = False,
        age_months_curve: Sequence[float] | None = None,
        verbose: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Apply blur, colour and contrast degradations as a differentiable op.

        All parameters preserve the original defaults; only keyword-only style
        was enforced to make calls explicit.
        """

        def _process(im: torch.Tensor, current_age: float) -> torch.Tensor:
            if verbose:
                print("Original mean/std:", im.mean().item(), im.std().item())

            # -----------------------------
            # 1 Visual acuity (Gaussian blur here, but also can be implemented in frequency domain)
            # -----------------------------
            if apply_blur:
                chosen_age = (
                    current_age
                    if not fully_random
                    else random.choice(age_months_curve or [current_age])
                )
                sigma = self.get_early_visual_acuity(chosen_age) * (image_size / 224)
                if sigma > 0:
                    ksize = int(8 * sigma) | 1  # ensure odd
                    im = KF.gaussian_blur2d(
                        im, (ksize, ksize), (sigma, sigma), border_type="reflect"
                    )

            # -----------------------------
            # 2 Chromatic sensitivity
            # -----------------------------
            if apply_color:
                chosen_age = (
                    current_age
                    if not fully_random
                    else random.choice(age_months_curve or [current_age])
                )
                chroma = self.get_chromatic_sensitivity(chosen_age)
                im = (
                    self.apply_chromatic_threshold_sensitivity(im, chroma)
                    if apply_threshold_color
                    else self.interpolate_color_grayscale(im, chroma)
                )

            # -----------------------------
            # 3 Contrast sensitivity (FFT domain)
            # -----------------------------
            if apply_contrast:
                chosen_age = current_age if not fully_random else random.choice(age_months_curve or [current_age])

                sensitivity = self.get_contrast_sensitivity_development(chosen_age) #+ 1e-10  # could add 1e-10 just avoid division of 0

                # Per-channel 2-D FFT ------------------------------------------------------------------
                fft_channels = [torch.fft.fft2(im[:, ch, :, :]) for ch in range(3)]
                power_spectra = [torch.abs(c) ** 2 for c in fft_channels]
                max_power = torch.stack([p.max() for p in power_spectra]).max()
                thr = max_power * (1 - sensitivity) * 0.001 * contrast_threshold

                kept = [c * (p >= thr) for c, p in zip(fft_channels, power_spectra)]
                spatial = [torch.fft.ifft2(c).real for c in kept]
                im = torch.stack(spatial, dim=1).clamp(0, 1)

            return im

        # Support lists for convenience -------------------------------------
        if isinstance(image, (list, tuple)):
            return [_process(img, age_months) for img in image]
        return _process(image, age_months)

    # ------------------------------------------------------------------
    # static helpers ---------------------------------------------
    # ------------------------------------------------------------------
    @staticmethod
    def interpolate_color_grayscale(image: torch.Tensor, chromatic_sensitivity: float) -> torch.Tensor:
        """Linearly blend original with grayscale according to sensitivity."""
        gray = torch.nn.functional.rgb_to_grayscale(image, num_output_channels=3)
        return chromatic_sensitivity * image + (1 - chromatic_sensitivity) * gray

    @staticmethod
    def apply_chromatic_threshold_sensitivity(
        images: torch.Tensor, chromatic_sensitivity: float
    ) -> torch.Tensor:
        """Desaturate pixels whose ΔE to neutral grey falls below a threshold.

        Args:
            images: *Tensor[B, 3, H, W]* in **[0, 1]**.
            chromatic_sensitivity: ∈[0, 1]; low values ⇒ stronger desaturation.
        """
        lab = kc.rgb_to_lab(images)
        gray_lab = torch.tensor([50.0, 0.0, 0.0], device=images.device).view(1, 3, 1, 1)
        delta_e = torch.norm(lab - gray_lab, dim=1, keepdim=True)

        mask = (delta_e > (128 * (1 - chromatic_sensitivity))).float()
        gray = torch.nn.functional.rgb_to_grayscale(images, num_output_channels=3)
        return mask * images + (1 - mask) * gray


# ---------------------------------------------------------------------------
# Utility 
# ---------------------------------------------------------------------------

def generate_age_months_curve(
    total_epochs: int,
    len_train_loader: int,
    months_per_epoch: float,
    *,
    shuffle: bool = False,
    seed: int | None = None,
    mid_phase: bool = False,
) -> List[float]:
    """Create a deterministic mapping from *batch* to *age (months)*.

    The helper is left implementation-identical while clarifying edge cases
    (shuffle, mid-phase) and adding type hints.
    """
    curve: List[float] = []
    for epoch in range(total_epochs):
        for batch_idx in range(len_train_loader):
            age = epoch * months_per_epoch + batch_idx * months_per_epoch / len_train_loader
            curve.append(age)

    if mid_phase:
        half = len(curve) // 2
        first_half = sorted(curve[::2], reverse=True)
        second_half = curve[1::2]
        curve = first_half + second_half  # type: ignore[list-item]

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(curve)

    return curve