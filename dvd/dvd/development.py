# dvd/dvd/development.py
"""
from dvd.dvd.development import DVDTransformer  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Developmental Visual Diet Simulator includes three-stage
pre-processing pipeline:

1. **Visual acuity**   → Gaussian blur or also in frequency domain
2. **Contrast sensitivity** → frequency-domain thresholding
3. **Chromatic sensitivity** → grayscale interpolation or ΔE thresholding
##########################
"""

import torch
import random
import numpy as np
from dataclasses import dataclass
from typing import List, Sequence, Union, Optional
import torchvision.transforms.functional as F
import kornia.color as kc
import kornia.filters as KF

@dataclass
class DVDConfig:
    """Configuration for DVDTransformer. Defaults preserve prior behavior."""
    apply_blur: int = 1
    apply_color: int = 1
    apply_contrast: int = 1
    contrast_threshold: float = 0.2
    apply_threshold_color: bool = False
    image_size: int = 224
    fully_random: bool = False
    age_months_curve: Optional[Sequence[float]] = None
    verbose: bool = False


class DVDTransformer:
    """
    DVD Data Transformer (main API)

    Use:
        cfg = DVDConfig()
        dvdt = DVDTransformer(cfg)
        out = dvdt(image_tensor, months=age_in_months)
    """

    def __init__(self, config: DVDConfig = DVDConfig()):
        self.cfg = config

    # ---------- Development curves ----------
    def get_early_visual_acuity(self, age_months: float) -> float:
        """Returns visual acuity as Gaussian blur sigma for a given age (months)."""
        a, b, c, d = (
            18.035945052640425,
            0.7933899743217134,
            1.6012490927401029,
            0.027054604551482078,
        )
        return a * np.exp(-b * age_months) + c * np.exp(-d * age_months)

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
            min_age = color_params["min_sensitivity_threshold_ages"]
            return (T(min_age, a, b, alpha) / T(age_years, a, b, alpha)) if age_years != 0 else 0.0

        return average_color_sensitivity(age_months)

    def get_contrast_sensitivity_development(
        self,
        age_months: float,
        age50: float = 4.8 * 12, 
        n: float = 2.1633375920569247,
    ) -> float:
        """Returns normalized contrast sensitivity development in [0,1]."""
        y_max = (300 ** n) / (300 ** n + age50 ** n)  # at ~25 years (300 months)
        return (age_months ** n) / (age_months ** n + age50 ** n) / y_max  # Range [0, 1]

    # ---------- Public call ----------
    def __call__(
        self,
        image: Union[torch.Tensor, List[torch.Tensor]],
        months: int,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Applies age-dependent blur, color desaturation, and frequency-domain contrast filtering.

        Args:
            image: Tensor [B,3,H,W] in [0,1] or list of such tensors.
            months: Age in months guiding all transformations.
        """
        return self._apply_transformations(image, months)

    # ---------- Core transform (minimally changed from original) ----------
    def _apply_transformations(
        self,
        image: Union[torch.Tensor, List[torch.Tensor]],
        age_months: int,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:

        cfg = self.cfg

        def process_image(img: torch.Tensor, age_m: float) -> torch.Tensor:
            if cfg.verbose:
                print(f"Original mean: {img.mean()} and {img.std}")

            # --- Visual acuity (Gaussian blur) ---
            if cfg.apply_blur:
                age_m = random.choice(cfg.age_months_curve) if cfg.fully_random and cfg.age_months_curve else age_m
                blur_sigma = self.get_early_visual_acuity(age_m) * (cfg.image_size / 224)

                if blur_sigma > 0:
                    k = int(8 * blur_sigma)
                    kernel_size = k + (1 if k % 2 == 0 else 0)
                    img = KF.gaussian_blur2d(
                        img,
                        (kernel_size, kernel_size),
                        (blur_sigma, blur_sigma),
                        border_type="reflect",
                    )

            # --- Color development (RGB↔grayscale mix or threshold) ---
            if cfg.apply_color:
                age_m = random.choice(cfg.age_months_curve) if cfg.fully_random and cfg.age_months_curve else age_m
                chromatic_sensitivity = self.get_color_sensitivity(age_m)

                if cfg.apply_threshold_color:
                    img = self.apply_chromatic_threshold_sensitivity(img, chromatic_sensitivity)
                else:
                    img = self.interpolate_color_grayscale(img, chromatic_sensitivity)

            # --- Contrast sensitivity (frequency-domain filtering) ---
            if cfg.apply_contrast:
                age_m = random.choice(cfg.age_months_curve) if cfg.fully_random and cfg.age_months_curve else age_m
                contrast_sensitivity = self.get_contrast_sensitivity_development(age_m) + 1e-10

                # Simple hard threshold in linear power domain
                fft_channels = [torch.fft.fft2(img[:, i, :, :]) for i in range(3)]
                power_spectra = [torch.abs(fc) ** 2 for fc in fft_channels]
                max_power = max(ps.max() for ps in power_spectra)
                threshold = 0.001 * max_power * cfg.contrast_threshold * (1 - contrast_sensitivity) 

                fft_filtered = [
                    fc * (ps >= threshold) for fc, ps in zip(fft_channels, power_spectra)
                ]
                filtered_channels = [torch.fft.ifft2(fc).real for fc in fft_filtered]
                img = torch.stack(filtered_channels, dim=1).clip(0, 1)

            return img

        # Handle list vs. tensor input
        if isinstance(image, list):
            return [process_image(img, age_months) for img in image]
        return process_image(image, age_months)

    # ---------- Helpers ----------
    @staticmethod
    def interpolate_color_grayscale(image: torch.Tensor, chromatic_sensitivity: float) -> torch.Tensor:
        """Blend RGB image with its grayscale version according to chromatic_sensitivity."""
        grayscale_image = F.rgb_to_grayscale(image, num_output_channels=3)
        return chromatic_sensitivity * image + (1 - chromatic_sensitivity) * grayscale_image

    @staticmethod
    def apply_chromatic_threshold_sensitivity(
        images: torch.Tensor, chromatic_sensitivity: float
    ) -> torch.Tensor:
        """
        Chromatic thresholding: ΔE below a threshold goes grayscale.
        images: (B,3,H,W) in [0,1]; chromatic_sensitivity*128 = ΔE threshold.
        """
        lab_images = kc.rgb_to_lab(images)  # (B, 3, H, W)
        gray_lab = torch.tensor([50.0, 0.0, 0.0], device=images.device).view(1, 3, 1, 1)
        delta_e = torch.norm(lab_images - gray_lab, dim=1, keepdim=True)  # (B, 1, H, W)
        color_mask = (delta_e > (128 * (1 - chromatic_sensitivity))).float()
        grayscale_images = F.rgb_to_grayscale(images, num_output_channels=3)
        return color_mask * images + (1 - color_mask) * grayscale_images


##########################
# Utilities
##########################

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

    return curve