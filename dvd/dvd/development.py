from __future__ import annotations

"""from dvd.dvd.development import DVDTransformer  
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Developmental Visual Diet Simulator includes three-stage
pre-processing pipeline:

1. **Visual acuity**   → isotropic Gaussian blur
2. **Contrast sensitivity** → frequency-domain thresholding
3. **Chromatic sensitivity** → grayscale interpolation or ΔE thresholding

"""

from dataclasses import dataclass
import math
import random
from typing import Iterable, List, Sequence, Tuple, Union

import numpy as np
import torch
import torchvision.transforms.functional as F
import kornia.color as kc
import kornia.filters as kf

try:
    import pytorch_lightning as pl
except ModuleNotFoundError:  # keep import‑side optional for non‑Lightning users
    pl = None  # type: ignore

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    "DVDConfig",
    "DVDTransformer",  # backwards-compat alias: DVDTransformer
    "AgeCurve",
]

# ---------------------------------------------------------------------------
# Parameter curves (fitted from developmental psychophysics literature)
# ---------------------------------------------------------------------------
class _DevCurves:
    """Static helpers that turn *age ∈ months* into perceptual parameters."""

    _VISUAL_ACUITY = (18.06, 0.79, 1.60, 0.027)  # dual-exponential fit
    _CONTRAST = dict(age50=4.8 * 12, n=2.1633375920569247)  # logistic
    _CHROMA = dict(
        a=0.008604133954779169,
        b=4.380740053287391e-05,
        alpha=0.8807610802743646,
        min_age=(21.2 + 21.1 + 18.8) / 3,  # psychophysics threshold
    )

    # ----------------------------- Visual acuity ----------------------------
    @staticmethod
    def visual_acuity(months: float) -> float:
        a, b, c, d = _DevCurves._VISUAL_ACUITY
        return a * math.exp(-b * months) + c * math.exp(-d * months)

    # ------------------------- Contrast sensitivity -------------------------
    @staticmethod
    def contrast(months: float, /) -> float:
        age50, n = _DevCurves._CONTRAST.values()
        y_max = (300**n) / (300**n + age50**n)
        return (months**n) / (months**n + age50**n) / y_max

    # ------------------------- Chromatic sensitivity ------------------------
    @staticmethod
    def chroma(months: float, /) -> float:
        p = _DevCurves._CHROMA
        a, b, alpha, min_age = p["a"], p["b"], p["alpha"], p["min_age"]
        yrs = months / 12.0

        def _t(age_yrs: float) -> float:
            return a * age_yrs ** (-alpha) + b * age_yrs ** alpha

        return 0.0 if yrs == 0 else _t(min_age) / _t(yrs)


# ---------------------------------------------------------------------------
# DVD Configuration
# ---------------------------------------------------------------------------
@dataclass
class DVDConfig:
    """Hyper-parameters controlling each perceptual dimension."""

    # enable/disable stages
    blur: bool = True
    contrast: bool = True
    color: bool = True

    # image geometry
    image_size: int = 224

    # contrast-sensitivity (Default: value-based hyperparameters)
    beta: float = 0.1
    lam: float = 150

    # contrast-sensitivity (percentile-based hyperparameters)
    # If your input images are not normalized to [0, 1], consider switching to percentile-based thresholding,
    # which adapts to the image’s actual intensity distribution.
    by_percentile: bool = False
    gamma: float = 8.0  # slope of logistic
    mid_q: float = 0.99  # inflection

    # colour
    threshold_color: bool = False  # use ΔE threshold vs. interpolation


# ---------------------------------------------------------------------------
# Developmental Visual Diet (DVD) simulator 
# ---------------------------------------------------------------------------
class DVDTransformer:
    """Drop-in replacement for *DVDTransformer* with cleaner internals."""

    def __init__(self, cfg: DVDConfig | None = None):
        self.cfg = cfg or DVDConfig()

    # ------------------------- API ----------------------
    def __call__(
        self,
        image: Union[torch.Tensor, Sequence[torch.Tensor]],
        months: float,
        *,
        randomise: bool = False,
        curriculum: Sequence[float] | None = None,
        verbose: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Simulate visual perception of a child *months* old.

        Args:
            image: A *[B, 3, H, W]* tensor in **[0, 1]** or an iterable thereof.
            months: Reference age in months.
            randomise: Uniformly sample ages ≤ *months* each call.  If *curriculum*
                        is provided, sample from it instead.
            curriculum: Optional list of ages to draw from when *randomise*.
            verbose: Print mean/std diagnostics.
        """
        if isinstance(image, (list, tuple)):
            return [self._process(img, months, randomise, curriculum, verbose) for img in image]
        return self._process(image, months, randomise, curriculum, verbose)

    # ------------------------------ Internals ------------------------------
    def _process(
        self,
        img: torch.Tensor,
        months: float,
        randomise: bool,
        curriculum: Sequence[float] | None,
        verbose: bool,
    ) -> torch.Tensor:
        if verbose:
            print("Mean/std pre:", img.mean().item(), img.std().item())

        # 1) Visual acuity (Gaussian blur)
        if self.cfg.blur:
            age = _sample_age(months, randomise, curriculum)
            sigma = _DevCurves.visual_acuity(age) * (self.cfg.image_size / 224)
            if sigma > 0:
                ksize = int(8 * sigma) | 1  # ensure odd kernel
                img = kf.gaussian_blur2d(
                    img,
                    kernel_size=(ksize, ksize),
                    sigma=(sigma, sigma),
                    border_type="reflect",
                )

        # 2) Chromatic sensitivity
        if self.cfg.color:
            age = _sample_age(months, randomise, curriculum)
            chroma = _DevCurves.chroma(age)
            if self.cfg.threshold_color:
                img = _apply_color_threshold(img, chroma)
            else:
                img = _blend_grayscale(img, chroma)

        # 3) Contrast sensitivity (FFT)
        if self.cfg.contrast:
            age = _sample_age(months, randomise, curriculum)
            sens = _DevCurves.contrast(age)
            img = _contrast_fft(
                img,
                sens,
                beta=self.cfg.beta,
                lam=self.cfg.lam,
                by_percentile=self.cfg.by_percentile,
                gamma=self.cfg.gamma,
                mid_q=self.cfg.mid_q,
            )

        if verbose:
            print("Mean/std post:", img.mean().item(), img.std().item())
        return img.clamp(0, 1)


# ---------------------------------------------------------------------------
# epoch-to-age curve genrator 
# ---------------------------------------------------------------------------
class AgeCurve:
    """Generate curriculum ages for training."""

    @staticmethod
    def generate(
        epochs: int,
        batches_per_epoch: int,
        months_per_epoch: float,
        *,
        shuffle: bool = False,
        seed: int | None = None,
        mid_phase: bool = False,
    ) -> List[float]:
        """Return length *epochs × batches_per_epoch* age curve."""
        curve = [
            ep * months_per_epoch + b * months_per_epoch / batches_per_epoch
            for ep in range(epochs)
            for b in range(batches_per_epoch)
        ]

        if mid_phase:
            first = sorted(curve[::2], reverse=True)
            second = curve[1::2]
            curve = first + second

        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(curve)
        return curve


# ---------------------------------------------------------------------------
# ------------------------------ Helper funcs ------------------------------ #
# ---------------------------------------------------------------------------

def _sample_age(current: float, randomise: bool, curriculum: Sequence[float] | None) -> float:
    """Either return *current* or randomly sample from *curriculum*."""
    if randomise and curriculum:
        return random.choice(curriculum)
    return current


def _blend_grayscale(img: torch.Tensor, chroma: float) -> torch.Tensor:
    """Linear interpolate between colour and grayscale."""
    gray = F.rgb_to_grayscale(img, num_output_channels=3)
    return chroma * img + (1 - chroma) * gray


def _apply_color_threshold(img: torch.Tensor, chroma: float) -> torch.Tensor:
    """Desaturate pixels whose ΔE is below threshold."""
    lab = kc.rgb_to_lab(img)
    neutral = torch.tensor([50.0, 0.0, 0.0], device=img.device).view(1, 3, 1, 1)
    delta_e = torch.norm(lab - neutral, dim=1, keepdim=True)

    mask = (delta_e > (128 * (1 - chroma))).float()
    gray = F.rgb_to_grayscale(img, num_output_channels=3)
    return mask * img + (1 - mask) * gray


def _contrast_fft(
    img: torch.Tensor,
    sens: float,
    *,
    beta: float,
    lam: float,
    by_percentile: bool,
    gamma: float,
    mid_q: float,
) -> torch.Tensor:
    """Low-pass filter in the frequency domain based on *sens* (0-1)."""
    fft = torch.fft.fft2(img)  # shape: [B, 3, H, W]
    power = torch.abs(fft) ** 2

    if by_percentile:
        q = 1 - 1 / (1 + math.exp(-gamma * (sens - mid_q)))
        q = float(max(0.01, min(q, 0.99)))

        # --- new: quantile per sample, avoids >16 M limit ---
        B = power.size(0)
        flat = power.reshape(B, -1)         # [B, N]
        thr = torch.quantile(flat, q, dim=1, keepdim=True)  # [B,1] # dim=1 keeps each sample “independent”
        mask = flat >= thr                  # [B,N]
        mask = mask.view_as(power)          # reshape back
    else:
        max_p = power.max()
        thr = (
            max_p
            * (1 - sens)
            * 0.001
            / max(math.floor(sens * 300 / lam) * 2, 1)
            * beta
        )
        mask = power >= thr

    fft_filtered = fft * mask
    return torch.fft.ifft2(fft_filtered).real

# ---------------------------------------------------------------------------
# Lightning integration — PyTorch‑Lightning Callback
# ---------------------------------------------------------------------------

class DVDCallback(pl.Callback if pl else object):
    """Inject age‑dependent degradations **before** the forward pass.

    The callback is completely stateless w.r.t. Lightning – every process holds
    its *own* `AgeCurve`, so DDP training is deterministic.  The curriculum is
    based on **global optimiser steps** (`trainer.global_step`) which makes
    resuming from checkpoints seamless.

    Parameters
    ----------
    cfg
        :class:`DVDConfig` instance that selects which degradations are active
        and their hyper‑parameters.
    total_epochs
        Total number of *training* epochs Lightning will run.
    iters_per_epoch
        Number of optimisation steps (batches) per epoch.  If you use dynamic
        re‑sizing (e.g. distributed `BatchSampler`) compute this **after** the
        dataloader is built: ``iters = len(train_loader)``.
    months_per_epoch
        Either a *float* (uniform growth) **or** a list/tuple with one entry per
        epoch to allow custom pacing.
    seed
        RNG seed for shuffled curricula.
    time_order
        Behaviour of the curriculum: ``"linear"`` (monotone), ``"mid_phase"``
        (shuffle 2nd half), ``"random"`` (shuffle whole curve) or
        ``"fully_random"`` (sample ≤ current age every call).
    """

    def __init__(
        self,
        *,
        cfg: DVDConfig,
        total_epochs: int,
        iters_per_epoch: int,
        months_per_epoch: float | Sequence[float] = 0.25,
        seed: int | None = 0,
        time_order: str = "normal",
    ) -> None:
        if pl is None:
            raise RuntimeError("pytorch_lightning not installed – callback unusable")

        self.transformer = DVDTransformer(cfg)
        self.curve = AgeCurve.generate(
            epochs=total_epochs,
            batches_per_epoch=iters_per_epoch,
            months_per_epoch=months_per_epoch if isinstance(months_per_epoch, float) else months_per_epoch[0] if months_per_epoch else 0.25,
            shuffle=(time_order == "random"),
            mid_phase=(time_order == "mid_phase"),
            seed=seed,
        )
        self.fully_random = time_order == "fully_random"

    # ------------------------------------------------------------------
    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Tuple[Any, ...] | List[Any],
        batch_idx: int,
    ) -> Tuple[Any, ...] | List[Any] | None:  # noqa: D401
        """Apply DVD before the model sees the image batch."""
        global_step = trainer.global_step  # counts optimiser steps *across* epochs

        # Guard against over‑stepping (can happen with grad‑accum)
        if global_step >= len(self.curve):
            return batch

        age_mo = self.curve[global_step]

        # We expect batch → (images, *rest)
        imgs, *rest = batch
        if not isinstance(imgs, torch.Tensor):
            raise TypeError(
                "DVDVisionCallback expects first element of batch to be a Tensor, "
                f"got {type(imgs).__name__} instead."
            )

        imgs = self.transformer(
            imgs,
            age_mo,
            randomise=self.fully_random,
        )
        return (imgs, *rest)  # Lightning will forward the modified batch

# ---------------------------------------------------------------------------
# ---------------------------- Usage Example ------------------------------- #
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """Demonstration script that recreates Fig.‑1 from the original DVD paper.

    It tiles several input images across columns that correspond to different
    developmental ages, producing a PDF grid for easy visual inspection.
    """
    from pathlib import Path
    from typing import List

    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    import torch

    # Local import (use relative import if the file is renamed/moved)
    from dvd.dvd.development import DVDTransformer, DVDConfig  # noqa: E402

    # ----------------------- Configuration ----------------------------------
    AGES: List[int] = [1, 4, 16, 64, 256]  # in months
    IMG_SIZE: int = 224                   # resize target size (pixels)

    # Input images & output file (change to suit your filesystem)
    SOURCE_DIR = Path("./assets/example_stimuli/")
    IMAGE_PATHS = [
        SOURCE_DIR / "example_1.jpeg",
        SOURCE_DIR / "example_2.jpeg",
    ]
    RESULT_DIR = Path("./results/dvd_demo_output/")
    OUTPUT_PATH = RESULT_DIR / "dvd_demo_output_debug_precentile.pdf"

    # -------------------------- Helpers -------------------------------------
    def load_as_tensor(fp: str | Path) -> torch.Tensor:
        """Load an RGB image as a 4‑D torch tensor *[1, 3, H, W]* in [0, 1]."""
        img = Image.open(fp).convert("RGB")
        img.thumbnail((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        arr = np.asarray(img).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(arr).float().unsqueeze(0)

    def grid_demo(image_paths: List[Path], out_file: Path, apply_contrast_by_percentile) -> None:
        """Create a PDF grid showing age‑progressive simulations."""
        dvdt = DVDTransformer(DVDConfig(by_percentile=apply_contrast_by_percentile))
        tensors = [load_as_tensor(p) for p in image_paths]

        rows, cols = len(tensors), len(AGES)
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

        for r, img_t in enumerate(tensors):
            for c, age in enumerate(AGES):
                out = dvdt(img_t.clone(), months=age)  # simulate perception
                vis = (out.squeeze(0).permute(1, 2, 0).cpu().numpy()).clip(0, 1)
                axes[r, c].imshow(vis)
                axes[r, c].axis("off")
                if r == 0:
                    axes[r, c].set_title(f"{age} mo", fontsize=12)

        fig.tight_layout()
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=200)
        print(f"Grid saved to {out_file.resolve()}")

    # --------------------------- Run it -------------------------------------
    grid_demo(IMAGE_PATHS, OUTPUT_PATH, apply_contrast_by_percentile=True)


