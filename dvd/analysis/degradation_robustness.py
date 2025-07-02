# -*- coding: utf-8 -*-
"""
Degradation Robustness Evaluation Utilities
===========================================
This module provides a *model‑agnostic* evaluator for measuring image‑classification
accuracy under common signal‑degradation settings (Gaussian blur and the ImageNet‑C
style corruption benchmark).

Key design goals
----------------
* **No internal model loading** – a *ready‑to‑run* `torch.nn.Module` is passed in by the caller.
* **Pluggable data loaders** – the user supplies plain `torch.utils.data.DataLoader`
  objects (or factory callables) for every degradation they wish to test.
* **Minimal dependencies** – only the PyTorch stack, `numpy`, and `pandas` are required.
* **Structured results** – everything is returned as Python dictionaries and can be
  exported to `.csv`, `.npz`, or a Pandas `DataFrame` with one‑line helpers.

Werid: Spatter,2, 3
Bad: Contrast Brightness Saturate

"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import logging
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast

import dvd.utils
from dvd.models.eval import validate, validate_in_limited

# from dig import get_corruption_names

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

################################################################################
# Helper functions                                                               #
################################################################################


def _accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Compute top‑1 accuracy in **percent** on a batch (no gradient graph)."""
    correct = (pred == target).sum().item()
    return 100.0 * correct / target.numel()


################################################################################
# Configuration dataclass                                                       #
################################################################################

@dataclass
class RobustnessConfig:
    """All *hyper‑parameters* for a robustness run in a single place."""

    eval_sigmas: Sequence[int] = (1, 2, 3, 4, 8, 12, 16, 20, 24, 32) #  (8, 32) #  
    # distortion_names: Sequence[str] = (
    #     # Noise
    #     "Gaussian Noise",
    #     "Shot Noise",
    #     "Impulse Noise",
    #     "Speckle Noise",

    #     # Weather
    #     "Frost",
    #     "Glass Blur",
        
    #     "Spatter",
    #     "Snow",

    #     "JPEG Compression",
    #     # Vision asepects

    #     "Contrast",
    #     "Brightness",
    #     "Saturate",

    #     "Gaussian Blur",
    #     "Defocus Blur",
    #     "Motion Blur",
    #     "Zoom Blur",
    #     "Fog",
    #     "Pixelate",
    #     "Elastic Transform",

    #     "Grid Mask" ,
    #     "Random Erasing",
    #     "Lens Distortion",  
    #     "Color Quantization",  
    #     "Chromatic", 
    #     "Aberration", 
    #     "Vignetting",
    # )
    
    # distortion_names = get_corruption_names()[:]
    # distortion_names = ["row_dropout", "tps_warp", "pixel_dropout", "line_jitter",
    #                     "clipping_artifacts", "reduce_bit_depth", "random_mosaic", "wave_distort",
    #                     "haze", "puddle_reflection", "sandstorm", "mist",
    #                     "dust_storm", "drizzle", "smog", "blizzard"]
    # distortion_names = ['frost', 'drizzle', 'blizzard', 'pixel_dropout', 'spatter']
    # distortion_names = ['speckle_noise', 'motion_blur', 'zoom_blur', 'glass_blur', 'gaussian_blur', 'defocus_blur', 'pixelate', 'jpeg_compression',]
    distortion_names = [ 
                        "gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise",
                         "gaussian_blur", "defocus_blur",  "motion_blur", "zoom_blur",
                        'rain','glass_blur', 'drizzle', 'blizzard', 
                        "pixelate", "jpeg_compression",'pixel_dropout','wave_distort',]  # 'pixelate', 'wave_distort', 
                 
    # distortion_names = ["color_quantization", "spatter", "posterize"]
    # distortion_names = [
    #                 "gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise",
    #                 "gaussian_blur", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    #                 "snow", "frost", "fog", "spatter",
    #                 "brightness", "contrast", "saturate",
    #                 "elastic_transform", "pixelate", "jpeg_compression",

    #                 "grid_mask", "random_erasing", "lens_distortion",
    #                 "vignetting", "chromatic_aberration", "color_quantization",
    #                 "solarize", "posterize", "rain", "hue", "invert"
    #             ]
    severities: Sequence[int] = (1, 2, 3, 4, 5)
    
    

    #* miniecoset
    # candidate_16ids =  [
    #     10,  # zebra
    #     30,  # butterfly
    #     107, # giraffe
    #     15,  # rhino
    #     36,  # mosquito
    #     8,   # tiger
    #     29,  # wasp
    #     4,   # elephant

    #     56,  # aloe
    #     90,  # lighthouse
    #     59,  # cherry
    #     100,  # keyboard
    #     48,  # fern
    #     94, # windmill
    #     83,  # bus
    #     49,  # cactus
    # ]
    # candidate_16ids = [
    #     82, 10, 90, 83, 56, 69, 100, 94,
    #     15, 32, 59, 30, 46, 66, 4, 99,
    #     68, 80, 88, 91, 48, 60, 65, 86,
    #     89, 41, 61, 62, 64, 92, 16, 25
    # ][:16]
    #* ecoset
    # candidate_16ids = [
    #     336,  # zebra
    #     198,  # butterfly
    #     487, # giraffe
    #     263,  # rhino
    #     300,  # mosquito
    #     42,   # tiger
    #     179,  # wasp
    #     60,   # elephant

    #     397,  # aloe
    #     169,  # lighthouse
    #     536,  # cherry
    #     297,  # keyboard
    #     87,  # fern
    #     280,  # windmill
    #     28,  # bus
    #     90,  # cactus
    # ]

    # old 16
    # candidate_16names = [
    #     'zebra', 'butterfly', 'giraffe','rhino','mosquito','tiger','wasp','elephant',
    #     'aloe','lighthouse','cherry','keyboard','fern','windmill','bus','cactus',
    # ]
    # object_candidate_16ids = [
    #             316, 487, 27, 493, 261, 336, 138, 160,
    #             169, 234, 276, 42, 54, 198, 226, 280,]

                # 426, 495, 126, 130, 263, 314, 446, 453,
                # 463, 523, 526, 533, 534, 230, 297, 386,
                # ][:16]
    # top_32_names = [
    #             "fireworks", "graffiti", "bus", "odometer", "jaguar", "zebra", "moose", "drawers",
    #             "lighthouse", "hedgehog", "flea", "tiger", "moon", "butterfly", "stagecoach", "windmill",
    #             "stegosaurus", "chess", "cave", "motorcycle", "rhino", "crib", "parachute", "gramophone",
    #             "silverfish", "chameleon", "geyser", "helicopter", "manatee", "treadmill", "typewriter", "mushroom"
    #         ][:16]

    # ecoset | high in sigma 8
    # object_candidate_16ids = [416, 54, 561, 463, 374, 526, 316, 552, 550, 260, 276, 437, 470, 
    #         143, 422, 48, 194, 332, 120, 219, 281, 478, 107, 191, 300, 505, 
    #         180, 184, 339, 394, 480, 483][:16]

    # 16 in frank tong paper
    # object_candidate_16ids = [54, 526, 61, 339, 292, 112, 26, 51,    134, 459, 3, 23, 16, 3, 97, 184]
    # candidate_16names = ['bear', 'bison', 'elephant', 'hamster', 'hare', 'lion', 'bird', 'cat', 'airplane', 'couch', 'car', 'ship', 'boat', 'car', 'lamp', 'teapot']

    # object_candidate_16ids = [54, 526, 61, 339, 292, 112, 26, 51,    134, 459, 3, 23, 16, 3, 97, 184]
    # object_candidate_16ids = [53, 525, 60, 338, 291, 111, 25, 50,    133, 458, 2, 22, 15,2, 96, 183]
    # candidate_16names = ['bear', 'bison', 'elephant', 'hamster', 'hare', 'lion', 'bird', 'cat', 
    # 'airplane', 'couch', 'car', 'ship', 'boat', 'car', 'lamp', 'teapot']

    # ok ish
    # object_candidate_16ids = [53, 525, 60, 338, 291, 111, 
    #                           171, 50, 
    #                           474, #134, 
    #                           458, 
    #                            2, 22, 15, 227, 463, 
    #                           183]

#     object_candidate_16ids = [
#     53, 525, 60, 338, 291, 111, 
#     170, 50, 
#    131, 
#     458, 
#     2, 22, 15, 226, 462, 
#     183
# ]
#     candidate_16names = ['bear', 'bison', 'elephant', 'hamster', 'hare', 'lion',
#                           'dolphin',  'cat',  ##
#                           'airplane',  # 131 # "antenna", #  473,   # # can change
#                            "couch", 
#                             'car', 'ship', 'boat', 'stagecoach', 'highlighter', # 'lamp',  # can change
#                             'teapot'
#                             ]

    # high choice
#     object_candidate_16ids = [
#     463, 526, 552, 300, 495, 55, 474, 470, 295, 281,
#     472, 422, 128, 160, 227, 238, 339, 437, 318, 550,
#     410, 194, 333, 448, 483, 171, 332, 319, 467, 48,
#     6, 40, 45, 75, 80, 107, 172, 180, 191, 200,
#     260, 271, 292, 311, 316, 480, 512
# ][:16]
    candidate_16names=[
        # high
        'panda','silverfish', 'spider', 'mosquito', # 'skunk', 'worm', 
        'missile', 'candelabra', 'chess', 'pan','hammer', 'drawers', # 'hourglass',
        # 16
        "bird", 'hare', 'hamster', 'cat',
        "couch", 'teapot'
     ]
    object_candidate_16ids = [
        # high
       474, 463, 295, 238, 128, 300,
       339, 311, 495, 281, 422, 160,
        # 16
       25, 291,  338, 50,  
       458, 183, 
    ]


    extra = ['highlighter', 'bison', 'hay', 'scallion', 'bracelet']
    extra_id = [463,526,]
    
    face_candidate_16ids=[
        68, 102, 105, 106, 11, 111, 87, 94, 63, 1 #! 1 just temporally involved since there are only 9 here
    ]
    

################################################################################
# Main evaluator class                                                          #
################################################################################

class DegradationRobustnessEvaluator:
    """A thin, **model‑agnostic** wrapper that orchestrates robustness evaluation.

    Parameters
    ----------
    test_loader
        Clean validation set.
    blur_loader_fn
        Callable that takes *sigma* (int) and returns a `DataLoader` with Gaussian‑
        blurred images.
    distortion_loader_fn
        Callable that takes *(method_name, severity)* and returns a `DataLoader`
        with that corruption.
    candidate_ids
        Optional subset of class indices (length 16 by convention) to compute
        accuracy on; if *None* the metric is standard top‑1.
    device
        Explicit CUDA / CPU choice; defaults to the best available option.

    All other knobs live in `RobustnessConfig`.
    """

    def __init__(
        self,
        *,
        test_loader: DataLoader,
        blur_loader_fn: Callable[[int], DataLoader],
        distortion_loader_fn: Callable[[str, int], DataLoader],
        config: RobustnessConfig | None = None,
        candidate_ids: Sequence[int] | None = None,
        device: torch.device | None = None,
        args = None,
    ) -> None:
        self.test_loader = test_loader
        self.blur_loader_fn = blur_loader_fn
        self.distortion_loader_fn = distortion_loader_fn
        self.config = config or RobustnessConfig()
        self.object_candidate_ids = candidate_ids or RobustnessConfig().object_candidate_16ids
        self.face_candidate_ids = candidate_ids or RobustnessConfig().face_candidate_16ids
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def evaluate_clean(self, test_loader, model: nn.Module, criterion, epoch,  image_size = 256, eval_subset_classes=True, test_type = ['all','face','obejct'][0], **kwargs) -> Dict[str, float]:
        """Return a *single* number – clean accuracy – wrapped in a dict."""
        
        print(f"testing for {test_type}")

        if test_type == 'face':
            top1, top5 = validate_in_limited(test_loader, model, criterion, epoch,  image_size , subset_class_ids=self.face_candidate_ids if eval_subset_classes else None ,)
        elif test_type == 'object':
            top1, top5 = validate_in_limited(test_loader, model, criterion, epoch,  image_size , subset_class_ids=self.object_candidate_ids if eval_subset_classes else None ,)
        else:
            top1, top5 = validate(test_loader, model, criterion, epoch,  image_size )
        
        print(f"Clean accuracy {top1} {top5}")
        return {"clean": top1}

    def evaluate_blur(
        self, model: nn.Module, args, criterion, epoch,  image_size = 256, eval_subset_classes=False, test_type = ['all','face','obejct'][0], *, transform_fn: Callable[[torch.Tensor], torch.Tensor] | None = None, 
    ) -> Dict[int, float]:
        """Evaluate *Gaussian‑blur* robustness across `config.eval_sigmas`."""
        model = model.to(self.device)
        results: Dict[int, float] = {}
        for sigma in self.config.eval_sigmas:
            loader = self.blur_loader_fn(int(sigma), args)
            if test_type == 'face':
                acc, _ = validate_in_limited(loader, model, criterion, epoch,  image_size, subset_class_ids=self.face_candidate_ids if eval_subset_classes else None ,)
            elif test_type == 'object':
                acc, _ = validate_in_limited(loader, model, criterion, epoch,  image_size , subset_class_ids=self.object_candidate_ids if eval_subset_classes else None ,)
            else:
                acc, _ = validate(loader, model, criterion, epoch,  image_size )
        
            results[int(sigma)] = acc
            logger.info("σ=%s → %.2f%%", sigma, acc)
        return results

    def evaluate_distortions(
        self, model: nn.Module, args,  criterion, epoch,  image_size = 256,  *, transform_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> Dict[str, Dict[int, float]]:
        """Full ImageNet‑C style corruption sweep.

        Returns
        -------
        Nested dict – `{method_name: {severity: accuracy, ...}, ...}`
        """
        model = model.to(self.device)
        results: Dict[str, Dict[int, float]] = {}

        for method in self.config.distortion_names:
            method_dict: Dict[int, float] = {}
            for severity in self.config.severities:
                loader = self.distortion_loader_fn(method, int(severity), args)
                # acc, _ = evaluate_loader(
                #     model,
                #     loader,
                #     self.device,
                #     candidate_ids=self.candidate_ids,
                #     transform_fn=transform_fn,
                # )
                acc, _ = validate(loader, model, criterion, epoch,  image_size )
                method_dict[int(severity)] = acc
                logger.info("%s | sev=%d → %.2f%%", method, severity, acc)
            results[method] = method_dict
        return results

    # ------------------------------------------------------------------
    # Convenience I/O helpers                                           
    # ------------------------------------------------------------------

    @staticmethod
    def results_to_dataframe(results_dict, model_name, metric="accuracy"):
        records = []
        for distortion, sev_dict in results_dict.items():
            if isinstance(sev_dict, float):  # clean result
                records.append({
                    "model": model_name,
                    "distortion": distortion,
                    "severity": 0,
                    metric: sev_dict,
                })
            else:
                for severity, acc in sev_dict.items():
                    records.append({
                        "model": model_name,
                        "distortion": distortion,
                        "severity": severity,
                        metric: acc,
                    })
        return pd.DataFrame(records)

    @staticmethod
    def save_npz(path: str | Path, **arrays: Mapping[str, np.ndarray] | np.ndarray) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)  # type: ignore[arg-type]
        logger.info("Saved NPZ ↗ %s", path)
