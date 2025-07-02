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


# import os
# import re
# import cv2
# import json
# import h5py
# import time
# import math
# import subprocess
# import wandb
# import argparse
# import random
# import numpy as np
# import scipy.stats as stats
# from collections import Counter
# import matplotlib.pyplot as plt
# from PIL import Image, ImageFilter, ImageChops

# import os
# import torch
# import torchvision
# import timm
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.models as models
# import torchvision.transforms as transforms
# from collections import defaultdict
# from torchvision import transforms
# import torchvision.transforms.functional as F
# import torch.optim.lr_scheduler as lr_scheduler
# from torch.optim.lr_scheduler import _LRScheduler
# from torchvision.datasets import ImageFolder

# import torch
# import torch.nn as nn
# import kornia.augmentation as K
# import kornia.filters as KF
# import kornia.color as kc


# ##########################
# ## Visual acuity & Color & Contrast Sensitivity Development Strategy
# ##########################
# class EarlyVisualDevelopmentTransformer:

#     def __init__(self):
#         pass

#     def get_early_visual_acuity(self, age_months):
#         a, b, c, d = 18.035945052640425, 0.7933899743217134, 1.6012490927401029, 0.027054604551482078
#         return a * np.exp(-b * age_months) + c * np.exp(-d * age_months)

#     def get_chromatic_sensitivity(self, age_months): 
       
#         color_params = { "AverageRGBDevelop": {  "a": 0.008604133954779169, "b": 4.380740053287391e-05, "alpha": 0.8807610802743646 },
#             'min_sensitivity_threshold_ages': (21.2 + 21.1 + 18.8) / 3, }

#         def T(age, a, b, alpha):
#             return a * age ** (-alpha) + b * age ** alpha
#         def custom_average_color_mix(age_months):
#             age_years = age_months / 12  # Convert months to years
#             param = color_params["AverageRGBDevelop"]
#             a, b, alpha = param["a"], param["b"], param["alpha"]
#             min_age = color_params["min_sensitivity_threshold_ages"]
#             return (T(min_age, a, b, alpha) / T(age_years, a, b, alpha)) if age_years != 0 else 0

#         return custom_average_color_mix(age_months)

#     def get_contrast_sensitivity_development(self, age_months, age50=4.8*12, n=2.1633375920569247):
#         y_max = (300**n) /(300 ** n + age50 ** n) # when age_months = 300 is max | 25 years old
#         return (age_months ** n) / (age_months ** n + age50 ** n) / y_max  # Range in [0,1]


#     def apply_fft_transformations(self, image, age_months, apply_blur=1, apply_color=1, apply_contrast=1, contrast_threshold =0.2, apply_threshold_color = False, image_size= 224, fully_random=False, age_months_curve= None, verbose=False):
#         """Applies a contrast sensitivity filter to the image based on the age of the subject.

#         Args:
#             image (PIL.Image or Tensor): The input image to be processed.
#             age_months (int): The age in months which determines the visual filter settings.
#             verbose (bool): If True, prints additional debug information.
        
#         Returns:
#             PIL.Image: The filtered image.
#         """
#         def process_image(image, age_months):
#             if verbose:
#                 print(f"Original mean: {image.mean()} and {image.std}")

#             # Visual acuity
#             if apply_blur:
#                 # just for fully random control
#                 age_months =  age_months if  fully_random else random.choice(age_months_curve) # random_age

#                 # Compute blur_sigma based on the age in months
#                 blur_sigma = self.get_early_visual_acuity(age_months) * ( image_size / 224 ) # Sigma scaling with image size 
#                 if blur_sigma > 0:
#                     kernel_size = int(8 * blur_sigma) + (1 if int(8 * blur_sigma) % 2 == 0 else 0)
#                     image = KF.gaussian_blur2d(
#                         image, (kernel_size, kernel_size),
#                         (blur_sigma, blur_sigma),
#                         border_type="reflect"
#                     )

#             # Color
#             if apply_color:
#                 # just for fully random control
#                 age_months =  age_months if not fully_random else random.choice(age_months_curve) # random_age

#                 chromatic_sensitivity = self.get_chromatic_sensitivity(age_months)
#                 if apply_threshold_color:
#                     image = self.apply_chromatic_threshold_sensitivity(image, chromatic_sensitivity)
#                 else:
#                     image = self.interpolate_color_grayscale(image, chromatic_sensitivity)

#             # Contrast
#             if apply_contrast:
#                 # just for fully random control
#                 age_months =  age_months if not fully_random else random.choice(age_months_curve) # random_age

#                 # Compute contrast sensitivity with a small offset to avoid division by zero
#                 contrast_sensitivity = self.get_contrast_sensitivity_development(age_months) + 1e-10
#                 # Process each channel (R, G, B) in the frequency domain
#                 fft_channels = [torch.fft.fft2(image[:, i, :, :]) for i in range(3)]  # List of [Batch_size, H, W] tensors
#                 # Compute the power spectrum for each channel
#                 power_spectra = [torch.abs(fft_channel) ** 2 for fft_channel in fft_channels]
#                 # Set a dynamic threshold based on age and contrast sensitivity
#                 max_power = max([power_spectra[i].max() for i in range(3)]) # RGB  #* max of all channels
#                 threshold = max_power* (1 - contrast_sensitivity) * 0.001 * contrast_threshold # contrast_threshold higher lower,  to control the speed of contrast drop speed, since it filters out the high frequency more
#                 # Apply thresholding to suppress low-power frequencies
#                 fft_filtered = [fft_channel * (power_spectrum >= threshold)
#                                 for fft_channel, power_spectrum in zip(fft_channels, power_spectra)]

#                 # Perform inverse FFT to obtain the filtered image in the spatial domain
#                 filtered_channels = [torch.fft.ifft2(fft_channel).real for fft_channel in fft_filtered]
#                 # Stack the filtered channels back into a single tensor
#                 image = torch.stack(filtered_channels, dim=1)  # Shape: [batch, 3, H, W]
#                 # Ensure pixel values are within the valid range [0, 1]
#                 image = image.clip(0, 1)

#             return image

#         # If the input is a list, process each image individually
#         if isinstance(image, list):
#             return [process_image(img, age_months) for img in image]
#         else:
#             return process_image(image, age_months)
            
#     @staticmethod
#     def interpolate_color_grayscale(image, chromatic_sensitivity):
#         # Convert the image to grayscale
#         grayscale_image = F.rgb_to_grayscale(image, num_output_channels=3)
        
#         # Blend the original image and the grayscale image based on the chromatic_sensitivity
#         blended_image = chromatic_sensitivity * image + (1 - chromatic_sensitivity) * grayscale_image
#         return blended_image

#     @staticmethod
#     def apply_chromatic_threshold_sensitivity(images: torch.Tensor, chromatic_sensitivity: float) -> torch.Tensor:
#         """
#         Apply chromatic thresholding to a batch of RGB images.

#         Parameters:
#         - images: Tensor of shape (B, 3, H, W), with values in [0, 1].
#         - chromatic_sensitivity: float, chromatic_sensitivity*128 = the ΔE threshold below which colors are converted to grayscale.

#         Returns:
#         - Tensor of shape (B, 3, H, W) with thresholded images.
#         """
#         # Convert RGB images to LAB color space
#         lab_images = kc.rgb_to_lab(images)  # Shape: (B, 3, H, W)

#         # Define the LAB value for neutral gray (L=50, a=0, b=0)
#         gray_lab = torch.tensor([50.0, 0.0, 0.0], device=images.device).view(1, 3, 1, 1)

#         # Compute ΔE (Euclidean distance in LAB space) between each pixel and gray
#         delta_e = torch.norm(lab_images - gray_lab, dim=1, keepdim=True)  # Shape: (B, 1, H, W)

#         # Create a mask where ΔE is greater than the threshold
#         color_mask = (delta_e > (128*(1-chromatic_sensitivity))).float()  # Shape: (B, 1, H, W)

#         # Convert original images to grayscale with 3 channels
#         grayscale_images = F.rgb_to_grayscale(images, num_output_channels=3)  # Shape: (B, 3, H, W)

#         # Blend images based on the mask
#         output_images = color_mask * images + (1 - color_mask) * grayscale_images

#         return output_images

# #* Determine the development time order: normal, random, mid-phase ...
# def generate_age_months_curve(total_epochs, len_train_loader, months_per_epoch, shuffle=False, seed=None, mid_phase=False):
#     """
#     Generate the sequence of age_months based on the epochs and the number of batches.
    
#     Args:
#     - total_epochs (int): Total number of epochs.
#     - len_train_loader (int): Number of batches in the training loader.
#     - months_per_epoch (float): Months per epoch.
#     - shuffle (bool): If True, shuffle the age_months curve.
#     - seed (int, optional): Seed for reproducibility when shuffling.
#     - mid_phase (bool): If True, the ages first go down in an alternating manner, then go up in an alternating manner.

#     Returns:
#     - age_months_curve (list): The generated (and possibly shuffled or mid-phased) age months curve.
#     """
#     age_months_curve = []
    
#     for epoch in range(0, total_epochs):
#         for batch_id in range(len_train_loader):
#             age_month = (epoch - 0) * months_per_epoch + batch_id * months_per_epoch / len_train_loader
#             age_months_curve.append(age_month)

#     if mid_phase:
#         half = len(age_months_curve) // 2
        
#         # Interleave the descending and ascending curves: first half decrease, second half increase
#         mid_phase_age_months_curve = [None] * len(age_months_curve)
#         mid_phase_age_months_curve[:half] = sorted(age_months_curve[::2], reverse=True)
#         mid_phase_age_months_curve[half:] = age_months_curve[1::2]
#         assert len(mid_phase_age_months_curve) ==  len(age_months_curve)
#         return mid_phase_age_months_curve

#     if shuffle:
#         if seed is not None:
#             random.seed(seed)
#         random.shuffle(age_months_curve)
#         return age_months_curve
    
#     return age_months_curve