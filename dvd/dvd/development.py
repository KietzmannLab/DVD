"""
development.py
==============

Utility components to simulate the **Developmental Visual Diet (DVD)** of the
human vision developmetn and apply it as a pre-processing pipeline to image tensors during neural‑network training.

----------
- `DVDTransformer` – differentiable augmentation pipeline that imposes the DVD
  in three dimensions (acuity, chromatic sensitivity, contrast sensitivity).
- `generate_age_months_curve` – helper that maps *(epoch, batch) → age/months*,
  enabling curriculum‑style training schedules.

"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Union

import numpy as np
import torch
import torchvision
import kornia.color as kc
import kornia.filters as KF

__all__ = ["DVDTransformer", "generate_age_months_curve"]

# --------------------------------------------------------------------------- #
# Parameter curves fitted from decades of developmental psychophysics data (see more in our DVD paper)                 
# --------------------------------------------------------------------------- #

_VISUAL_ACUITY_PARAMS = (18.06, 0.79, 1.60, 0.027)

_CONTRAST_SENSITIVITY_PARAMS = dict(age50=4.8 * 12, n=2.1633375920569247)

_CHROMA_PARAMS = dict(a=0.008604133954779169,
                      b=4.380740053287391e-05,
                      alpha=0.8807610802743646,
                      min_sensitivity_threshold_age = (21.2 + 21.1 + 18.8) / 3)

# --------------------------------------------------------------------------- #
# Main DVD transformer as a image pre-processing pipeline                                                          
# --------------------------------------------------------------------------- #
class DVDTransformer:
    """Differentiable image augmentations in DVD pipeline that emulate early visual development.The pipeline follows three visual asepcts:
        1. **Visual acuity** – Snellen-visual-acuity equivalent
        2. **Contrast sensitivity** – Frequency‑domain thresholding of the power spectrum.
        3. **Chromatic sensitivity** – Age‑dependent color fidelity  
    """

    # --------------------------------------------------------------------- #
    # Developmental curves                                                  #
    # --------------------------------------------------------------------- #
    @staticmethod
    def get_early_visual_acuity(age_months: float) -> float:
        """Fitted development trajectory of visual acuity maturation (Snellen-equivalent) via a dual-exponential function.
        Based on decades of infant and child psychophysics data (see paper for details).
        """
        a, b, c, d = _VISUAL_ACUITY_PARAMS
        return a * np.exp(-b * age_months) + c * np.exp(-d * age_months)

    @staticmethod
    def get_contrast_sensitivity_development(age_months: float, *, age50: float = _CONTRAST_SENSITIVITY_PARAMS["age50"], n: float = _CONTRAST_SENSITIVITY_PARAMS["n"],) -> float:
        """ Fitted development trajector of Contrast sensitivity(0–1), using a smooth, monotonic logistic fucntion.
            Grounded in childhood CSF psychophysics (see paper for details).
        Args:
            age_months: Age in months.
            age50: Age at which 50 % of maximum sensitivity is reached.
            n: fitted Steepness parameter.
        """
        y_max = (300**n) / (300**n + age50**n)
        return (age_months**n) / (age_months**n + age50**n) / y_max

    @staticmethod
    def get_chromatic_sensitivity(age_months: float) -> float:
        """Fitted development trajector of chromatic sensitivity(0–1), using a smooth, monotonic logistic fucntion,
        based via decades of psychophysical data from infancy to adulthood  (see paper for details).
        """
        param = _CHROMA_PARAMS

        def T(age, a, b, alpha):
                return a * age ** (-alpha) + b * age ** alpha
        
        def color_fidelity(age_months,param):
            age_years = age_months / 12  # Convert months to years
            a, b, alpha, min_age  = param["a"], param["b"], param["alpha"], param["min_sensitivity_threshold_age"]
            return (T(min_age, a, b, alpha) / T(age_years, a, b, alpha)) if age_years != 0 else 0

        return color_fidelity(age_months,param)
    

    # ------------------------------------------------------------------ #
    # Augmentation pipeline                                              #
    # ------------------------------------------------------------------ #
    def apply_fft_transformations(
        self,
        image: Union[torch.Tensor, Sequence[torch.Tensor]],
        age_months: float,
        *,
        # Toggles
        apply_blur: bool = True,
        apply_color: bool = True,
        apply_contrast: bool = True,
        # Hyper‑parameters
        contrast_amplitude_beta: float = 0.1,
        contrast_amplitude_lambda: float = 150,
        apply_threshold_color: bool = False,
        # others
        image_size: int = 224,
        fully_random: bool = False,
        age_months_curve: Sequence[float] | None = None,
        verbose: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Simulate vision at *age_months* on one or a list of image tensors.

        Parameters
        ----------
        image
            A `torch.Tensor` *[B, 3, H, W]* in **[0, 1]** or an iterable of such tensors.
        age_months
            Reference age in months; ignored if `fully_random=True` as a control model./
        apply_blur, apply_color, apply_contrast
            Enable or disable the three DVD dimensions individually.
        contrast_amplitude_beta, contrast_amplitude_lambda
            Factors controlling the contrast‑sensitivity threshold.
        apply_threshold_color
            If *True*, performs ΔE thresholding instead of linear blending for
            chromatic sensitivity.
        image_size
            Input spatial resolution. Used to scale the blur kernel for visual acuity simulation.
        fully_random
            When *True*, `age_months` becomes the upper bound of a uniform sample drawn every call. If `age_months_curve` is supplied the
            sample is taken from that list instead.
        age_months_curve
            Optional curriculum schedule to sample from when `fully_random=True`.
        verbose
            Print mean/std diagnostics of every processed image.

        Returns
        -------
        torch.Tensor or list[torch.Tensor]
            The augmented image(s) in the same shape as input.
        """

        def _process(im: torch.Tensor, ref_age: float) -> torch.Tensor:
            if verbose:
                print("Mean/std pre:", im.mean().item(), im.std().item())

            # ---------------------------------------------------------- #
            # 1. Visual acuity (Implement as Gaussian blur)                      
            # ---------------------------------------------------------- #
            if apply_blur:
                chosen_age = _random_or_fixed_age(
                    ref_age, fully_random=fully_random, curve=age_months_curve
                )
                sigma = (
                    self.get_early_visual_acuity(chosen_age) * (image_size / 224)
                )
                if sigma > 0:
                    ksize = int(8 * sigma) | 1  # ensure odd
                    im = KF.gaussian_blur2d(
                        im,
                        kernel_size=(ksize, ksize),
                        sigma=(sigma, sigma),
                        border_type="reflect",
                    )

            # ---------------------------------------------------------- #
            # 2. Chromatic sensitivity                                   #
            # ---------------------------------------------------------- #
            if apply_color:
                chosen_age = _random_or_fixed_age(
                    ref_age, fully_random=fully_random, curve=age_months_curve
                )
                chroma = self.get_chromatic_sensitivity(chosen_age)
                if apply_threshold_color:
                    im = self.apply_chromatic_threshold_sensitivity(im, chroma)
                else:
                    im = self.interpolate_color_grayscale(im, chroma)

            # ---------------------------------------------------------- #
            # 3. Contrast sensitivity (FFT)                              #
            # ---------------------------------------------------------- #
            if apply_contrast:
                chosen_age = _random_or_fixed_age(
                    ref_age, fully_random=fully_random, curve=age_months_curve
                )
                sens = self.get_contrast_sensitivity_development(chosen_age)

                # Channel‑wise FFT ↔ spatial‑domain pipeline
                fft_ch = [torch.fft.fft2(im[:, c, :, :]) for c in range(3)]
                power = [torch.abs(c) ** 2 for c in fft_ch]
                max_power = torch.stack([p.max() for p in power]).max()

                thr = (
                    max_power
                    * (1 - sens)
                    * 0.001
                    / max(math.floor(age_months / contrast_amplitude_lambda) * 2, 1)
                    * contrast_amplitude_beta
                )

                kept = [c * (p >= thr) for c, p in zip(fft_ch, power)]
                spatial = [torch.fft.ifft2(k).real for k in kept]
                im = torch.stack(spatial, dim=1).clamp(0, 1)

            return im

        # Support sequence inputs for convenience
        if isinstance(image, (list, tuple)):
            return [_process(img, age_months) for img in image]
        return _process(image, age_months)

    # ------------------------------------------------------------------ #
    # helper methods                                              #
    # ------------------------------------------------------------------ #
    @staticmethod
    def interpolate_color_grayscale(
        image: torch.Tensor, chromatic_sensitivity: float
    ) -> torch.Tensor:
        """Blend `image` with its grayscale version.

        A value of 1 → original colour; 0 → full grayscale.
        """
        gray = _grayscale(image)
        return chromatic_sensitivity * image + (1 - chromatic_sensitivity) * gray

    @staticmethod
    def apply_chromatic_threshold_sensitivity(
        images: torch.Tensor, chromatic_sensitivity: float
    ) -> torch.Tensor:
        """Desaturate pixels whose ΔE to mid‑gray ≤ threshold."""
        lab = kc.rgb_to_lab(images)
        neutral = torch.tensor([50.0, 0.0, 0.0], device=images.device).view(1, 3, 1, 1)
        delta_e = torch.norm(lab - neutral, dim=1, keepdim=True)

        mask = (delta_e > (128 * (1 - chromatic_sensitivity))).float()
        gray = _grayscale(images)
        return mask * images + (1 - mask) * gray


# --------------------------------------------------------------------------- #
# Curriculum helper                                                           #
# --------------------------------------------------------------------------- #

def generate_age_months_curve(
    total_epochs: int,
    len_train_loader: int,
    months_per_epoch: float,
    *,
    shuffle: bool = False,
    seed: int | None = None,
    mid_phase: bool = False,
) -> List[float]:
    """Map each batch index to an **age in months** for curriculum learning.

    Parameters
    ----------
    total_epochs
        Number of training epochs.
    len_train_loader
        Number of batches per epoch.
    months_per_epoch
        Increment added to *age* after every completed epoch.
    shuffle
        Shuffle the resulting curve (deterministically if *seed* is given).
    mid_phase
        If *True*, reverse every second element of the first half to introduce
        an 'interleaved' curriculum.

    Returns
    -------
    list[float]
        Flattened list of length `total_epochs × len_train_loader`.
    """
    curve: List[float] = [
        epoch * months_per_epoch + batch * months_per_epoch / len_train_loader
        for epoch in range(total_epochs)
        for batch in range(len_train_loader)
    ]

    if mid_phase:
        half = len(curve) // 2
        first_half = sorted(curve[::2], reverse=True)
        second_half = curve[1::2]
        curve = first_half + second_half  

    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(curve)

    return curve

# --------------------------------------------------------------------------- #
# Other helper utilities                                                            #
# --------------------------------------------------------------------------- #
def _grayscale(image: torch.Tensor) -> torch.Tensor:
    """Return a 3‑channel gray version of *image* ∈[0, 1]."""
    return torchvision.transforms.functional.rgb_to_grayscale(image, num_output_channels=3)

def _random_or_fixed_age(
    current_age: float,
    *,
    fully_random: bool,
    curve: Sequence[float] | None,
) -> float:
    """Either return *current_age* or sample uniformly from *curve*."""
    if fully_random and curve:
        return random.choice(curve)
    return current_age

if __name__ == "__main__":
    from pathlib import Path
    from typing import List

    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import torch

    # Local import or replace with the actual class definition
    # from dvd.dvd.development import DVDTransformer  # Replace or comment out if defined locally

    # Configuration (manually assigned)
    AGES = [1, 4, 16, 64, 256][:]  # in months
    IMG_SIZE = 224              # Resize target size

    # Set your input images and output file here
    SOURCE_DIR = '/share/klab/lzejin/lzejin/codebase/P001_DVD/data/example_stimuli/'
    IMAGE_PATHS = [
        SOURCE_DIR + "faces/Anthony_Hopkins_5977_3460.jpeg",
        SOURCE_DIR + "nature_scenes/window_table_flower.jpeg",
        SOURCE_DIR + "nature_scenes/msca_img2.jpg",
        SOURCE_DIR + "nature_scenes/indoor_living_room.jpeg",
        SOURCE_DIR + "nature_scenes/1164-early-morning-light-bear-canyon-arizona.jpg",
    ]
    RESULT_DIR = '/share/klab/lzejin/lzejin/codebase/P001_DVD/results/demo_outputs/'
    OUTPUT_PATH = Path(RESULT_DIR + "dvd_demo_output.pdf")

    def load_as_tensor(fp: str | Path) -> torch.Tensor:
        """Load an RGB image as a 4D torch tensor [1, 3, H, W] in [0, 1]."""
        img = Image.open(fp).convert("RGB")
        img.thumbnail((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        arr = np.asarray(img).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(arr).float().unsqueeze(0)


    def grid_demo(image_paths: List[str], out_file: Path) -> None:
        dvdt = DVDTransformer()
        tensors = [load_as_tensor(p) for p in image_paths]

        rows, cols = len(tensors), len(AGES)
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

        for r, img_t in enumerate(tensors):
            for c, age in enumerate(AGES):
                out = dvdt.apply_fft_transformations(img_t.clone(), age_months=age) # , apply_color = False)
                # out = img_t.clone()
                vis = (out.squeeze(0).permute(1, 2, 0).cpu().numpy()).clip(0, 1)
                axes[r, c].imshow(vis)
                axes[r, c].axis("off")
                if r == 0:
                    axes[r, c].set_title(f"{age} mo", fontsize=12)

        fig.tight_layout()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=200)
        print(f"✔ Grid saved to {out_file.resolve()}")


    # Run it
    grid_demo(IMAGE_PATHS, OUTPUT_PATH)
