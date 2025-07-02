# -*- coding: utf-8 -*-
"""GPU‑aware re‑implementation of *adversarial_robustness.py* using
`adversarial‑robustness‑toolbox` (ART) **plus** integration of three *noise‑based*
Foolbox attacks (``L2AdditiveGaussianNoiseAttack``,
``L2AdditiveUniformNoiseAttack`` and ``SaltAndPepperNoiseAttack``).

The new Noise attacks are executed with **Foolbox**'s ``PyTorchModel`` because they
are not implemented in ART. All other behaviour remains identical.
"""
from __future__ import annotations

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Mapping, Sequence, Tuple
import copy

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ART -------------------------------------------------------------------------
from art.attacks.evasion import (
    FastGradientMethod,
    BasicIterativeMethod,
    ProjectedGradientDescent,
    CarliniL2Method,
    CarliniLInfMethod,
    DeepFool,
    ElasticNet,
    AutoProjectedGradientDescent,
    BoundaryAttack,
    HopSkipJump,
    ZooAttack,
    SimBA,
    SquareAttack,
    UniversalPerturbation,
    TargetedUniversalPerturbation,
    PixelAttack,
    SpatialTransformation,
    AdversarialPatch,
)
from art.estimators.classification import PyTorchClassifier

# Foolbox (ONLY for the three noise attacks) ----------------------------------
import foolbox as fb
from foolbox import PyTorchModel
from foolbox.attacks import (
    L2AdditiveGaussianNoiseAttack,
    L2AdditiveUniformNoiseAttack,
    SaltAndPepperNoiseAttack,
)

# -----------------------------------------------------------------------------
# Logger
# -----------------------------------------------------------------------------
logging.getLogger("art.attacks.evasion").setLevel(logging.WARNING)  # mute ART spam
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

# -----------------------------------------------------------------------------
# 1. Configuration dataclass
# -----------------------------------------------------------------------------
@dataclass
class RobustnessConfig:
    """Hyper‑parameters for corruption & adversarial robustness (ART + FB)."""

    # Classic corruption knobs (kept for parity, unused here)
    eval_sigmas: Sequence[int] = (1, 2, 4, 8, 16, 32)
    distortion_names: Sequence[str] = ("Gaussian Noise", "Snow", "Contrast")
    severities: Sequence[int] = (1, 2, 3, 4, 5)

    # Epsilon / parameter grids – names kept identical to the Foolbox version
    attack_grid: Dict[str, Sequence[float]] = field(
        default_factory=lambda: {
            # White‑box (ART) --------------------------------------------------
            "FGSM":  [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3],
            "FGM":   [1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 0.5, 0.8, 1.0],
            "PGD":   [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3],
            "C&W_L2":    [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3],
            "DeepFool":  [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3],
            "MI_FGSM":   [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3],
            "EAD":       [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3],
            "Auto_PGD":  [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3],
            "C&W_Linf":  [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3],
            "BIM":       [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3],
            "LBFGS":     [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3],
            # Black‑box / decision‑based (ART) -------------------------------
            "BoundaryAttack": [0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
            "HopSkipJump":    [1e-3, 1e-2, 1e-1, 1, 10, 100],
            "ZOO":            [0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
            "SimBA":          [0.05, 0.1, 0.2, 0.4, 0.6, 0.8],
            "Square":         [0.05, 0.1, 0.3, 0.5, 0.7, 0.9],
            "UniversalPert":  [1, 10, 50, 100, 500, 1000],
            "PixelAttack":    [1, 10, 50, 100, 500, 1000],
            "AdversarialPatch": [1e-3, 1e-2, 1e-1, 1, 10, 100],
            # Spatial transforms (ART)
            "SpatialAttack_rotation":    [15, 30, 45, 60, 75],
            "SpatialAttack_translation": [15, 30, 45, 60, 75],
            "SpatialAttack_scaling":     [1.2, 1.5, 1.8, 2.0],
            # --- New: Noise‑based (Foolbox) ----------------------------------
            "L2AdditiveGaussianNoiseAttack": [10, 20, 50, 80, 100, 150, 200],
            "L2AdditiveUniformNoiseAttack":  [10, 20, 50, 80, 100, 150, 200],
            "SaltAndPepperNoiseAttack":      [10, 20, 50, 80, 100, 150, 200][:],
        }
    )

    # Map attack‑name → *factory* returning an attack object. For ART attacks the
    # classifier is passed in; for the 3 FB noise attacks, we return an *instance*
    # directly (they do not depend on ART's classifier).
    attack_objects: Mapping[str, Callable[[PyTorchClassifier], object] | object] = field(
        default_factory=lambda: {
            # White‑box --------------------------------------------------------
            "FGSM":       lambda clf: FastGradientMethod(estimator=clf, norm=np.inf, batch_size=512),
            "FGM":        lambda clf: FastGradientMethod(estimator=clf, norm=2, batch_size=512),
            "PGD":        lambda clf: ProjectedGradientDescent(estimator=clf, norm=np.inf, batch_size=512),
            "C&W_L2":     lambda clf: CarliniL2Method(classifier=clf, confidence=0.0, batch_size=512),
            "DeepFool":   lambda clf: DeepFool(classifier=clf, batch_size=512, max_iter= 3),
            "MI_FGSM":    lambda clf: FastGradientMethod(estimator=clf, norm=np.inf, decay=0.99, batch_size=512),
            "EAD":        lambda clf: ElasticNet(classifier=clf, confidence=0.0, batch_size=512),
            "Auto_PGD":   lambda clf: AutoProjectedGradientDescent(estimator=clf, norm=np.inf, batch_size=512),
            "C&W_Linf":   lambda clf: CarliniLInfMethod(classifier=clf, confidence=0.0, batch_size=512),
            "BIM":        lambda clf: BasicIterativeMethod(estimator=clf, batch_size=512),
            "LBFGS":      lambda clf: BasicIterativeMethod(estimator=clf, batch_size=512, eps_step=1e-2),
            # Black‑box / score‑based ----------------------------------------
            "BoundaryAttack":  lambda clf: BoundaryAttack(estimator=clf, batch_size=512),
            "HopSkipJump":     lambda clf: HopSkipJump(classifier=clf, batch_size=512, max_iter = 5, max_eval = 10,),
            "ZOO":             lambda clf: ZooAttack(classifier=clf, nb_parallel=512, max_iter= 2),
            "SimBA":           lambda clf: SimBA(classifier=clf, batch_size=512, max_iter= 10),
            "Square":          lambda clf: SquareAttack(estimator=clf, norm=np.inf, batch_size=512, max_iter= 5),
            "UniversalPert":   lambda clf: TargetedUniversalPerturbation(classifier=clf, max_iter =3),
            "PixelAttack":     lambda clf: PixelAttack(classifier=clf,max_iter=5),
            "AdversarialPatch": lambda clf: AdversarialPatch(estimator=clf, max_iter=5),
            # Spatial --------------------------------------------------------
            "SpatialAttack_rotation":    lambda clf: SpatialTransformation(estimator=clf),
            "SpatialAttack_translation": lambda clf: SpatialTransformation(estimator=clf),
            "SpatialAttack_scaling":     lambda clf: SpatialTransformation(estimator=clf),
            # --- New: Noise‑based (Foolbox) ----------------------------------
            "L2AdditiveGaussianNoiseAttack": L2AdditiveGaussianNoiseAttack(),
            "L2AdditiveUniformNoiseAttack":  L2AdditiveUniformNoiseAttack(),
            "SaltAndPepperNoiseAttack":      SaltAndPepperNoiseAttack(steps=10),
        }
    )

# -----------------------------------------------------------------------------
# 2. Utility: model loader (unchanged API)
# -----------------------------------------------------------------------------

def load_model(checkpoint: str | Path, constructor: Callable[[], nn.Module],
               device: torch.device | None = None) -> nn.Module:
    """Instantiate *constructor*, load weights from *checkpoint*, return on *device*."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = constructor()
    ckpt = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(ckpt.get("state_dict", ckpt))
    return model.to(device).eval()

# -----------------------------------------------------------------------------
# 3. Evaluator
# -----------------------------------------------------------------------------
class AdversarialRobustnessEvaluator:
    """Evaluate many ART **and** Foolbox attacks in one sweep.

    * ART attacks run exactly as before (via :pyclass:`~art.estimators.classification.PyTorchClassifier`).
    * The three noise‑based Foolbox attacks are executed with
      :pyclass:`~foolbox.PyTorchModel` to ensure correctness and speed.
    """

    #: attacks that must be run with Foolbox
    _NOISE_ATTACKS = {
        "L2AdditiveGaussianNoiseAttack",
        "L2AdditiveUniformNoiseAttack",
        "SaltAndPepperNoiseAttack",
    }

    def __init__(
        self,
        *,
        test_loader: DataLoader,
        config: RobustnessConfig | None = None,
        device: torch.device | None = None,
        model_bounds: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self.loader = test_loader
        self.cfg = config or RobustnessConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bounds = model_bounds

    # ------------------------------------------------------------------
    def evaluate(
        self,
        model: nn.Module,
        *,
        attack_subset: Sequence[str] | None = None,
        model_name: str = "model",
        epoch: str | int = "na",
        log_dir: str | Path = "results/adv_robustness",
    ) -> pd.DataFrame:
        """Run *attack_subset* on *model*; return DataFrame of accuracies."""
        model.to(self.device).eval()

        # Build ART classifier (for ART‑backed attacks) ----------------------
        x_ref, _ = next(iter(self.loader))
        input_shape = tuple(x_ref.shape[1:])
        try:
            nb_classes = model(x_ref[:1].to(self.device)).shape[1]
        except Exception:  # pylint: disable=broad-except
            nb_classes = 1000  # fallback
            LOGGER.warning("Falling back to nb_classes=%d (could not infer from model)", nb_classes)

        classifier = PyTorchClassifier(
            model=model,
            loss=nn.CrossEntropyLoss(),
            optimizer=None,
            input_shape=input_shape,
            nb_classes=nb_classes,
            clip_values=self.bounds,
            device_type="gpu" if self.device.type == "cuda" else "cpu",
        )

        # Build Foolbox model once (for noise attacks) -----------------------
        fmodel = PyTorchModel(model, bounds=self.bounds, preprocessing={})

        # Clean accuracy -----------------------------------------------------
        clean_acc = _accuracy_loader(model, self.loader, self.device)
        LOGGER.info("Clean accuracy %.2f%%", 100 * clean_acc)

        # Prepare bookkeeping ------------------------------------------------
        results: Dict[str, Dict[float, float]] = {}
        attacks_to_run = attack_subset or self.cfg.attack_objects.keys()
        save_dir = Path(log_dir) / model_name
        save_dir.mkdir(parents=True, exist_ok=True)

        # Main attack loop ---------------------------------------------------
        for atk_name in attacks_to_run:
            if atk_name not in self.cfg.attack_objects:
                LOGGER.warning("Unknown/unsupported attack %s – skipping.", atk_name)
                continue

            param_grid = self.cfg.attack_grid.get(atk_name, [None])
            atk_results: Dict[float, float] = {}

            # ----------------------------------------------------------------
            # CASE 1: noise‑based Foolbox attack
            # ----------------------------------------------------------------
            if atk_name in self._NOISE_ATTACKS:
                atk_proto = self.cfg.attack_objects[atk_name]  # already an *instance*
                for param in param_grid:
                    atk_obj = copy.deepcopy(atk_proto)
                    LOGGER.info("Running %s @ %s", atk_name, param)
                    acc = _run_noise_attack(
                        fmodel, atk_obj, atk_name, self.loader, self.device, param
                    )
                    atk_results[param] = acc
                    LOGGER.info("  accuracy %.2f%%", 100 * acc)
                    
                    # claer cache
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()

            # ----------------------------------------------------------------
            # CASE 2: everything else – handled by ART
            # ----------------------------------------------------------------
            else:
                atk_factory = self.cfg.attack_objects[atk_name]  # type: ignore[assignment]
                for param in param_grid:
                    atk_obj_factory = copy.deepcopy(atk_factory)  # type: ignore[operator]
                    LOGGER.info("Running %s @ %s", atk_name, param)
                    acc = run_attack_art(
                        classifier, atk_obj_factory, atk_name, self.loader, param, self.device
                    )
                    atk_results[param] = acc
                    LOGGER.info("  accuracy %.2f%%", 100 * acc)

            # Record clean baseline (ε = 0) once per attack ------------------
            atk_results.setdefault(0.0, clean_acc)
            results[atk_name] = atk_results

            # Save per‑attack CSV -------------------------------------------
            df_atk = pd.DataFrame(
                [
                    {"Model": model_name, "Attack": atk_name, "Epsilon": eps, "Accuracy": acc, "Epoch": epoch}
                    for eps, acc in atk_results.items()
                ]
            )
            atk_csv = save_dir / f"{atk_name}_ep{epoch}.csv"
            if atk_csv.exists():
                df_atk.to_csv(atk_csv, mode="a", index=False, header=False)
            else:
                df_atk.to_csv(atk_csv, index=False)

        # Persist global results ---------------------------------------------
        df = self.results_to_dataframe(results, model_name, epoch)
        csv_path = save_dir / f"adv_robustness_ep{epoch}.csv"
        if csv_path.exists():
            df.to_csv(csv_path, mode="a", index=False, header=False)
            LOGGER.info("Appended results → %s", csv_path)
        else:
            df.to_csv(csv_path, index=False)
            LOGGER.info("Created results → %s", csv_path)

        # right before save_npz loop
        for atk, eps_acc in results.items():
            # stringify keys
            strmapped = {str(eps): arr for eps, arr in eps_acc.items()}
            self.save_npz(save_dir / f"{atk}_ep{epoch}.npz", **strmapped)

        return df

    # ---------------- Convenience helpers ------------------------------------
    @staticmethod
    def results_to_dataframe(results: Dict[str, Dict[float, float]],
                             model_name: str, epoch) -> pd.DataFrame:
        recs = [
            {"Model": model_name, "Attack": atk, "Epsilon": eps, "Accuracy": acc, "Epoch": epoch}
            for atk, grid in results.items() for eps, acc in grid.items()
        ]
        return pd.DataFrame(recs)

    @staticmethod
    def save_npz(path: str | Path, **arrays) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)
        LOGGER.debug("Saved NPZ ↗ %s", path)

# -----------------------------------------------------------------------------
# 4. Internal helpers
# -----------------------------------------------------------------------------
@torch.no_grad()
def _accuracy_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Compute top‑1 accuracy over *loader* (no gradients)."""
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        preds = model(x).argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    return correct / total if total else 0.0


# ---------------- ART attack runner ------------------------------------------

def run_attack_art(
    classifier: PyTorchClassifier,
    atk_obj_factory: Callable[[PyTorchClassifier], object],
    atk_name: str,
    loader: DataLoader,
    param,
    device: torch.device,
) -> float:
    """Run one **ART** attack over the full loader."""
    classifier.model.to(device)
    attack = atk_obj_factory(classifier)

    # Spatial attacks get special parameters --------------------------------
    if "SpatialAttack" in atk_name:
        _configure_spatial_art(attack, atk_name, param)
    else:
        # Common parameter names in ART
        for name in ("eps", "epsilon", "noise_scale", "radius"):
            if hasattr(attack, name) and param is not None:
                attack.set_params(**{name: param})
                break

    total, correct = 0, 0
    for x, y in loader:
        x_dev = x.to(device)
        y_dev = y.to(device)
        x_np = x_dev.detach().cpu().numpy()
        y_np = y_dev.detach().cpu().numpy()

        x_adv_np = attack.generate(x=x_np, y=y_np)
        preds = classifier.predict(x_adv_np).argmax(axis=1)
        correct += int((preds == y_np).sum())
        total += y_np.shape[0]

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return correct / total if total else 0.0


# ---------------- Foolbox noise attack runner --------------------------------

def _run_noise_attack(
    fmodel: PyTorchModel,
    atk_obj,
    atk_name: str,
    loader: DataLoader,
    device: torch.device,
    param,
) -> float:
    """Execute *atk_obj* (Foolbox) once over the entire *loader*."""
    total, correct = 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        # Most FB attacks accept epsilons=[...] even if len==1
        try:
            with torch.cuda.amp.autocast():
                raw, clipped, success = atk_obj(fmodel, x, y, epsilons=[param])  # type: ignore[misc]
        except TypeError:
            with torch.cuda.amp.autocast():
                raw, clipped, success = atk_obj(fmodel, x, y)  # type: ignore[misc]

        success = success.squeeze()
        correct += (~success).sum().item()
        total += success.numel()

        del raw, clipped, success, x, y
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    return correct / total if total else 0.0

# ---------------- Helper for spatial ART attacks ----------------------------

def _configure_spatial_art(attack: SpatialTransformation, atk_name: str, param):
    """Specialise *SpatialTransformation* parameters based on *atk_name*."""
    if not isinstance(attack, SpatialTransformation):
        return
    if "rotation" in atk_name:
        attack.set_params(rotation_max=param)
    elif "translation" in atk_name:
        attack.set_params(translation_max=param)
    elif "scaling" in atk_name:
        attack.set_params(scale_max=param)

# -----------------------------------------------------------------------------
# 5. CLI stub (optional)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ART + Foolbox robustness sweep (CUDA‑aware)")
    parser.add_argument("checkpoint", type=str, help="Path to .pth checkpoint")
    parser.add_argument("--csv-dir", default="results/adv_robustness")
    parser.add_argument("--epoch", default="na")
    args = parser.parse_args()

    # User‑provided dataset/model builders -----------------------------------
    from my_dataset import build_val_loader      # TODO: implement
    from my_models import create_model           # TODO: implement

    loader = build_val_loader(batch_size=256, workers=16, pin_memory=True)
    net = load_model(args.checkpoint, create_model)

    evaluator = AdversarialRobustnessEvaluator(test_loader=loader)
    evaluator.evaluate(net, epoch=args.epoch, log_dir=args.csv_dir)


# # adversarial_robustness_art_cuda.py
# """GPU‑aware re‑implementation of *adversarial_robustness.py* using
# `adversarial‑robustness‑toolbox` (ART).

# Key upgrades (behaviour otherwise identical):

# * **Automatic CUDA acceleration** – `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")` is used everywhere, and ART
#   is told to run on GPU with `device_type="gpu"` when possible.
# * **Safe tensor⇆NumPy conversion** – all data passed to or coming from ART
#   attacks/classifiers are **CPU NumPy arrays**.  Torch tensors are moved
#   to CPU before conversion (`tensor.cpu().numpy()`) to avoid the
#   ``TypeError: can't convert cuda:0 device type tensor to numpy`` that the
#   user encountered.
# * **No functional API changes** – all public classes, functions and
#   dataclass fields keep their original signatures so that importing code
#   keeps working unchanged.
# """
# from __future__ import annotations

# # -----------------------------------------------------------------------------
# # Imports
# # -----------------------------------------------------------------------------
# import logging
# import os
# from dataclasses import dataclass, field
# from pathlib import Path
# from typing import Dict, Mapping, Sequence, Tuple, Callable

# import numpy as np
# import pandas as pd
# import torch
# from torch import nn
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# import copy


# from typing import Callable

# # ART -------------------------------------------------------------------------
# from art.attacks.evasion import (
#     FastGradientMethod,
#     BasicIterativeMethod,
#     ProjectedGradientDescent,
#     CarliniL2Method,
#     CarliniLInfMethod,
#     DeepFool,
#     ElasticNet,
#     AutoProjectedGradientDescent,
#     BoundaryAttack,
#     HopSkipJump,
#     ZooAttack,
#     SimBA,
#     SquareAttack,
#     UniversalPerturbation,
#     TargetedUniversalPerturbation,
#     PixelAttack,
#     SpatialTransformation,
#     AdversarialPatch,
# )
# from art.estimators.classification import PyTorchClassifier

# # If you want to mute *all* evasion‐attack logs:
# logging.getLogger("art.attacks.evasion").setLevel(logging.WARNING) #* the default printed acc is wrong

# # -----------------------------------------------------------------------------
# # Logger
# # -----------------------------------------------------------------------------
# LOGGER = logging.getLogger(__name__)
# LOGGER.setLevel(logging.INFO)

# # -----------------------------------------------------------------------------
# # 1. Configuration dataclass
# # -----------------------------------------------------------------------------
# @dataclass
# class RobustnessConfig:
#     """Hyper‑parameters for corruption & adversarial robustness (ART flavour)."""

#     # Classic corruption knobs (kept for parity, unused here)
#     eval_sigmas: Sequence[int] = (1, 2, 4, 8, 16, 32)
#     distortion_names: Sequence[str] = ("Gaussian Noise", "Snow", "Contrast")
#     severities: Sequence[int] = (1, 2, 3, 4, 5)

#     # Epsilon / parameter grids – names kept identical to the Foolbox version
#     attack_grid: Dict[str, Sequence[float]] = field(
#         default_factory=lambda: {
#             "FGSM":  [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3],
#             "FGM":   [1e-4, 1e-3, 1e-2, 3e-2, 0.1, 0.3, 0.5, 0.8, 1.0],
#             "PGD":   [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3],
#             "C&W_L2":    [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3],
#             "DeepFool":  [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3],
#             "MI_FGSM":   [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3],
#             "EAD":       [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3],
#             "Auto_PGD":  [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3],
#             "C&W_Linf":  [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3],
#             "BIM":       [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3],
#             "LBFGS":     [1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3],
#             # Black‑box / decision‑based
#             "BoundaryAttack": [0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
#             "HopSkipJump":    [1e-3,1e-2,1e-1, 1, 10, 100],
#             "ZOO":            [0.01, 0.02, 0.04, 0.06, 0.08, 0.1],
#             "SimBA":          [0.05, 0.1, 0.2, 0.4, 0.6, 0.8],
#             "Square":         [0.05, 0.1, 0.3, 0.5, 0.7, 0.9],
#             "UniversalPert":  [1, 10, 50, 100, 500, 1000],
#             "PixelAttack":    [1, 10, 50, 100, 500, 1000],
#             "AdversarialPatch": [1e-3,1e-2,1e-1, 1, 10, 100],
#             # Spatial transforms
#             "SpatialAttack_rotation":    [15, 30, 45, 60, 75],
#             "SpatialAttack_translation": [15, 30, 45, 60, 75],
#             "SpatialAttack_scaling":     [1.2, 1.5, 1.8, 2.0],
#         }
#     )

#     # Map attack‑name → factory (*callable* that receives the ART classifier)
#     attack_objects: Mapping[str, Callable[[PyTorchClassifier], object]] = field(
#         default_factory=lambda: {
#             # White‑box --------------------------------------------------------
#             "FGSM":       lambda clf: FastGradientMethod(estimator=clf, norm=np.inf, batch_size=512),
#             "FGM":        lambda clf: FastGradientMethod(estimator=clf, norm=2, batch_size=512),
#             "PGD":        lambda clf: ProjectedGradientDescent(estimator=clf, norm=np.inf, batch_size=512),
#             "C&W_L2":     lambda clf: CarliniL2Method(classifier=clf, confidence=0.0, batch_size=512),
#             "DeepFool":   lambda clf: DeepFool(classifier=clf, batch_size=512),
#             "MI_FGSM":    lambda clf: FastGradientMethod(estimator=clf, norm=np.inf, decay=0.99, batch_size=512),
#             "EAD":        lambda clf: ElasticNet(classifier=clf, confidence=0.0, batch_size=512),
#             "Auto_PGD":   lambda clf: AutoProjectedGradientDescent(estimator=clf, norm=np.inf, batch_size=512),
#             "C&W_Linf":   lambda clf: CarliniLInfMethod(classifier=clf, confidence=0.0, batch_size=512),
#             "BIM":        lambda clf: BasicIterativeMethod(estimator=clf, batch_size=512),
#             # Black‑box / score‑based ----------------------------------------
#             "BoundaryAttack":  lambda clf: BoundaryAttack(estimator=clf, batch_size=512),
#             "HopSkipJump":     lambda clf: HopSkipJump(classifier=clf, batch_size= 512),
#             "ZOO":             lambda clf: ZooAttack(classifier=clf, nb_parallel=512), #* more parallelism in the finite‑difference calls:
#             "SimBA":           lambda clf: SimBA(classifier=clf, batch_size = 512), #* Batch size (but, batch process unavailable in this implementation)
#             "Square":          lambda clf: SquareAttack(estimator=clf, norm=np.inf, batch_size = 512),
#             "UniversalPert":   lambda clf: UniversalPerturbation(classifier=clf),
#             "PixelAttack":     lambda clf: PixelAttack(classifier=clf),
#             "AdversarialPatch": lambda clf: AdversarialPatch(estimator=clf),
#             # Spatial --------------------------------------------------------
#             "SpatialAttack_rotation":    lambda clf: SpatialTransformation(estimator=clf),
#             "SpatialAttack_translation": lambda clf: SpatialTransformation(estimator=clf),
#             "SpatialAttack_scaling":     lambda clf: SpatialTransformation(estimator=clf),
#         }
#     )


# # -----------------------------------------------------------------------------
# # 2. Utility: model loader (unchanged API)
# # -----------------------------------------------------------------------------

# def load_model(checkpoint: str | Path, constructor: Callable[[], nn.Module],
#                device: torch.device | None = None) -> nn.Module:
#     """Instantiate *constructor*, load weights from *checkpoint*, return on *device*."""
#     device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = constructor()
#     ckpt = torch.load(checkpoint, map_location="cpu")
#     model.load_state_dict(ckpt.get("state_dict", ckpt))
#     return model.to(device).eval()


# # -----------------------------------------------------------------------------
# # 3. Evaluator
# # -----------------------------------------------------------------------------
# class AdversarialRobustnessEvaluator:
#     """Evaluate many ART attacks in one sweep (signature compatible with the
#     original Foolbox‑based evaluator)."""

#     def __init__(
#         self,
#         *,
#         test_loader: DataLoader,
#         config: RobustnessConfig | None = None,
#         device: torch.device | None = None,
#         model_bounds: Tuple[float, float] = (0.0, 1.0),
#     ) -> None:
#         self.loader = test_loader
#         self.cfg = config or RobustnessConfig()
#         self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.bounds = model_bounds

#     # ------------------------------------------------------------------
#     def evaluate(
#         self,
#         model: nn.Module,
#         *,
#         attack_subset: Sequence[str] | None = None,
#         model_name: str = "model",
#         epoch: str | int = "na",
#         log_dir: str | Path = "results/adv_robustness",
#     ) -> pd.DataFrame:
#         """Run *attack_subset* on *model*; return DataFrame of accuracies."""
#         model.to(self.device).eval()

#         # Build ART classifier -------------------------------------------------
#         x_ref, _ = next(iter(self.loader))
#         input_shape = tuple(x_ref.shape[1:])  # e.g. (3, 256, 256)
#         nb_classes = 565  # Provided by the user; can also infer from model.
#         try:
#             nb_classes = model(x_ref[:1].to(self.device)).shape[1]
#         except Exception:  # pylint: disable=broad-except
#             LOGGER.warning("Falling back to nb_classes=%d (could not infer from model)", nb_classes)

#         classifier = PyTorchClassifier(
#             model=model,
#             loss=nn.CrossEntropyLoss(),
#             optimizer=None,
#             input_shape=input_shape,
#             nb_classes=nb_classes,
#             clip_values=self.bounds,
#             device_type="gpu" if self.device.type == "cuda" else "cpu",
#         )

#         # Clean accuracy -------------------------------------------------------
#         clean_acc = _accuracy_loader(model, self.loader, self.device)
#         LOGGER.info("Clean accuracy %.2f%%", 100 * clean_acc)

#         # Run attacks ----------------------------------------------------------
#         results: Dict[str, Dict[float, float]] = {}
#         attacks_to_run = attack_subset or self.cfg.attack_objects.keys()
#         save_dir = Path(log_dir) / model_name
#         save_dir.mkdir(parents=True, exist_ok=True) 

#         for atk_name in attacks_to_run:

#             try:
#                 if atk_name not in self.cfg.attack_objects:
#                     LOGGER.warning("Unknown/unsupported attack %s – skipping.", atk_name)
#                     continue

#                 param_grid = self.cfg.attack_grid.get(atk_name, [None])
#                 atk_results: Dict[float, float] = {}
                

#                 for param in param_grid:
#                     atk_obj_factory = copy.deepcopy(self.cfg.attack_objects[atk_name])
#                     LOGGER.info("Running %s @ %s", atk_name, param)
#                     # import pdb;pdb.set_trace()
#                     acc = run_attack_art(
#                         classifier, atk_obj_factory, atk_name, self.loader, param, self.device
#                     )
#                     # import pdb;pdb.set_trace()
#                     atk_results[param if param is not None else 0.0] = acc
#                     LOGGER.info("  accuracy %.2f%% \n", 100 * acc)

#                 # Always store clean accuracy under epsilon=0 for plotting parity
#                 atk_results.setdefault(0.0, clean_acc)
#                 results[atk_name] = atk_results

#                 #* --- NEW: save this attack’s results into its own CSV ---
#                 # Build a small DataFrame for this attack
#                 df_atk = pd.DataFrame([
#                     {"Model": model_name, "Attack": atk_name, "Epsilon": eps, "Accuracy": acc, "Epoch": epoch}
#                     for eps, acc in atk_results.items()
#                 ])
#                 atk_csv = save_dir / f"{atk_name}_ep{epoch}.csv"
#                 if atk_csv.exists():
#                     df_atk.to_csv(atk_csv, mode="a", index=False, header=False)
#                     LOGGER.info("Appended per‐attack results → %s", atk_csv)
#                 else:
#                     df_atk.to_csv(atk_csv, index=False)
#                     LOGGER.info("Created per‐attack results → %s", atk_csv)
            
#             except:
#                 print(f"Skip {atk_name} attack due to error")

#         # Persist results ------------------------------------------------------
#         df = self.results_to_dataframe(results, model_name, epoch)
#         save_dir = Path(log_dir) / model_name
#         save_dir.mkdir(parents=True, exist_ok=True)

#         csv_path = save_dir / f"adv_robustness_ep{epoch}.csv"
#         if csv_path.exists():
#             df.to_csv(csv_path, mode="a", index=False, header=False)
#             LOGGER.info("Appended results → %s", csv_path)
#         else:
#             df.to_csv(csv_path, index=False)
#             LOGGER.info("Created results → %s", csv_path)

#         for atk, eps_acc in results.items():
#             self.save_npz(save_dir / f"{atk}_ep{epoch}.npz", **eps_acc)

#         return df

#     # ---------------- Convenience helpers ------------------------------------
#     @staticmethod
#     def results_to_dataframe(results: Dict[str, Dict[float, float]],
#                              model_name: str, epoch) -> pd.DataFrame:
#         recs = [
#             {"Model": model_name, "Attack": atk, "Epsilon": eps, "Accuracy": acc, "Epoch": epoch}
#             for atk, grid in results.items()
#             for eps, acc in grid.items()
#         ]
#         return pd.DataFrame(recs)

#     @staticmethod
#     def save_npz(path: str | Path, **arrays) -> None:
#         path.parent.mkdir(parents=True, exist_ok=True)
#         np.savez(path, **arrays)
#         LOGGER.debug("Saved NPZ ↗ %s", path)


# # -----------------------------------------------------------------------------
# # 4. Internal helpers
# # -----------------------------------------------------------------------------
# @torch.no_grad()
# def _accuracy_loader(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
#     """Compute top‑1 accuracy over *loader* (no gradients)."""
#     total, correct = 0, 0
#     for x, y in loader:
#         x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
#         preds = model(x).argmax(dim=1)
#         correct += (preds == y).sum().item()
#         total += y.size(0)
#     return correct / total if total else 0.0



# def run_attack_art(
#     classifier: PyTorchClassifier,
#     atk_obj_factory: Callable[[PyTorchClassifier], object],
#     atk_name: str,
#     loader: DataLoader,
#     param,
#     device: torch.device,
# ) -> float:
#     """Run *one* attack (with parameter *param*) over the full loader,
#     automatically using CUDA if available."""
#     # ensure classifier's model is on device
#     classifier.model.to(device)
#     # classifier.device_type = "cuda" if device.type == "cuda" else "cpu"

#     attack = atk_obj_factory(classifier)  # fresh instance per ε/param

#     # Parameterisation --------------------------------------------------------
#     if "SpatialAttack" in atk_name:
#         _configure_spatial_art(attack, atk_name, param)
#     else:
#         # Common naming conventions across ART attacks
#         if hasattr(attack, "eps") and param is not None:
#             attack.set_params(eps=param)
#         elif hasattr(attack, "epsilon") and param is not None:
#             attack.set_params(epsilon=param)
#         elif hasattr(attack, "noise_scale") and param is not None:
#             attack.set_params(noise_scale=param)
#         elif hasattr(attack, "radius") and param is not None:
#             attack.set_params(radius=param)

#     # Main loop ---------------------------------------------------------------
#     total, correct = 0, 0
#     # import pdb;pdb.set_trace()
#     for batch_idx, (x, y) in enumerate(loader):
#         # print(x.shape, y.shape)
#         # if batch_idx >= 2: break
#         # Move to device (GPU or CPU)
#         x_device = x.to(device)
#         y_device = y.to(device)

#         # Detach & move back to CPU before NumPy conversion
#         x_np = x_device.detach().cpu().numpy()
#         y_np = y_device.detach().cpu().numpy()

#         # Generate adversarial examples (runs on CPU via NumPy)
#         x_adv_np = attack.generate(x=x_np, y=y_np)

#         # Evaluate
#         predictions = classifier.predict(x_adv_np)
#         preds_labels = predictions.argmax(axis=1)
#         # accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_np, axis=1)) # / len(y_test)

#         correct += int((preds_labels == y_np).sum())
#         total += y_np.shape[0]

#         # Free GPU VRAM between batches (important for large inputs)
#         if device.type == "cuda":
#             torch.cuda.empty_cache()
#         # import pdb;pdb.set_trace()

#     # import pdb;pdb.set_trace()
#     acc = correct / total if total else 0.0
#     print(f"Accuracy: {acc}\n")
#     return acc


# # # -----------------------------------------------------------------------------
# # # 4. Internal helpers (replace run_attack_art with this version)
# # # -----------------------------------------------------------------------------
# # def run_attack_art(
# #     classifier: PyTorchClassifier,
# #     atk_obj_factory: Callable[[PyTorchClassifier], object],
# #     atk_name: str,
# #     loader: DataLoader,
# #     param,
# #     device: torch.device,
# #     micro_batch_size: int = 16,          # <- new tuning knob
# # ) -> float:
# #     """Run *one* attack over the full loader, with micro‑batching + big GPU eval."""
# #     # Move model
# #     classifier.model.to(device)
# #     attack = atk_obj_factory(classifier)

# #     # ZOO already has nb_parallel in factory; otherwise you could detect here:
# #     # if atk_name == "ZOO": attack.set_params(nb_parallel=32)

# #     # Set epsilon/radius etc.
# #     if "SpatialAttack" in atk_name:
# #         _configure_spatial_art(attack, atk_name, param)
# #     else:
# #         for name in ("eps", "epsilon", "noise_scale", "radius"):
# #             if hasattr(attack, name) and param is not None:
# #                 attack.set_params(**{name: param})
# #                 break

# #     total, correct = 0, 0
# #     buffer_x, buffer_y = [], []

# #     def flush_batch():
# #         nonlocal total, correct, buffer_x, buffer_y
# #         # stack and move to device
# #         xb = torch.cat(buffer_x, dim=0).to(device)
# #         yb = torch.cat(buffer_y, dim=0).to(device)
# #         x_np = xb.detach().cpu().numpy()
# #         y_np = yb.detach().cpu().numpy()
# #         buffer_x, buffer_y = [], []

# #         # generate adversarial examples once for whole micro-batch
# #         x_adv_np = attack.generate(x=x_np, y=y_np)

# #         # move back to GPU and predict in one go
# #         x_adv_t = torch.from_numpy(x_adv_np).to(device)
# #         with torch.no_grad():
# #             logits = classifier.model(x_adv_t)
# #             preds = logits.argmax(dim=1).cpu().numpy()

# #         correct += int((preds == y_np).sum())
# #         total += y_np.shape[0]

# #         if device.type == "cuda":
# #             torch.cuda.empty_cache()

# #     # iterate and accumulate
# #     for x, y in loader:
# #         buffer_x.append(x)
# #         buffer_y.append(y)
# #         if sum(b.size(0) for b in buffer_x) >= micro_batch_size:
# #             flush_batch()

# #     # flush leftover
# #     if buffer_x:
# #         flush_batch()

# #     return correct / total if total else 0.0


# def _configure_spatial_art(attack: SpatialTransformation, atk_name: str, param):
#     """Specialise *SpatialTransformation* parameters based on *atk_name*."""
#     if not isinstance(attack, SpatialTransformation):
#         return
#     if "rotation" in atk_name:
#         attack.set_params(rotation_max=param)
#     elif "translation" in atk_name:
#         attack.set_params(translation_max=param)
#     elif "scaling" in atk_name:
#         attack.set_params(scale_max=param)


# # -----------------------------------------------------------------------------
# # 5. CLI stub (optional)
# # -----------------------------------------------------------------------------
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="ART robustness sweep (CUDA‑aware)")
#     parser.add_argument("checkpoint", type=str, help="Path to .pth checkpoint")
#     parser.add_argument("--csv-dir", default="results/adv_robustness")
#     parser.add_argument("--epoch", default="na")
#     args = parser.parse_args()

#     # User‑provided dataset/model builders -----------------------------------
#     from my_dataset import build_val_loader      # TODO: implement
#     from my_models import create_model           # TODO: implement

#     loader = build_val_loader(batch_size=256, workers=16, pin_memory=True)
#     net = load_model(args.checkpoint, create_model)

#     evaluator = AdversarialRobustnessEvaluator(test_loader=loader)
#     evaluator.evaluate(net, epoch=args.epoch, log_dir=args.csv_dir)

