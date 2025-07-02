# -*- coding: utf-8 -*-
"""
Degradation Robustness Evaluation Utilities
===========================================
This module provides a *model‑agnostic* evaluator for measuring image‑classification
accuracy under common signal‑degradation settings (Gaussian blur and the ImageNet‑C
style corruption benchmark).


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

    # Gaussian blur sigmas
    eval_sigmas: Sequence[int] = (1, 2, 3, 4, 8, 12, 16, 20, 24, 32) 
    
    # Degradation conditions
    distortion_names = [ 
                        "gaussian_noise", "shot_noise", "impulse_noise", "speckle_noise",
                         "gaussian_blur", "defocus_blur",  "motion_blur", "zoom_blur",
                        'rain','icy_window', 'drizzle', 'snow', 
                        "pixelate", "jpeg_compression",'pixel_dropout','wave_distort',]  # 'pixelate', 'wave_distort', 

    severities: Sequence[int] = (1, 2, 3, 4, 5)
    
    
    # 16 objects: 8 animancy, 8 inanimancy
    candidate_16names=[
        "bird", 'hare', 'hamster', 'cat', 'panda','silverfish', 'spider', 'mosquito',
        "couch", 'teapot', 'missile', 'candelabra', 'chess', 'pan','hammer', 'drawers',  
     ]
    object_candidate_16ids = [
       25, 291,  338, 50,   474, 463, 295, 238, 
       458, 183,  339, 311, 495, 281, 422, 160, 
    ]
    
    # 10 faces
    face_candidate_10ids=[
        68, 102, 105, 106, 11, 111, 87, 94, 63, 1 
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
        self.face_candidate_ids = candidate_ids or RobustnessConfig().face_candidate_10ids
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
