import os
import re
import random
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import timm
from torchvision import models as torchvision_models

import dvd.simclr.optimizer

# add custom models to torchvision_models
from . import custom_models
torchvision_models.__dict__["custom_cnn"] = custom_models.custom_cnn


# ============================================================
# Model creation
# ============================================================
def create_model(args, logger=None) -> Tuple[nn.Module, str]:
    """
    Create model from torchvision first, otherwise from timm.
    Also replace the final classification layer according to dataset_name.
    Returns:
        model, linear_keyword
    """
    if logger is not None:
        logger.info(f"Creating model '{args.arch}'")

    if args.arch in torchvision_models.__dict__:
        model = torchvision_models.__dict__[args.arch](pretrained=False)
        linear_keyword = infer_linear_keyword(model, args.arch)
    else:
        try:
            model = timm.create_model(args.arch, pretrained=False)
        except Exception as e:
            raise ValueError(
                f"Model architecture '{args.arch}' not found in torchvision or timm."
            ) from e
        linear_keyword = infer_linear_keyword(model, args.arch)

    if args.dataset_name in ["rgbd_texture2shape_miniecoset"]:
        model = change_first_layer_channels(model, channels=4)

    out_dim = get_output_dim(args.dataset_name)

    try:
        model = change_last_layer(model, out_dim, linear_keyword)
    except Exception:
        model = change_vit_num_classes(model, num_classes=out_dim)

    return model, linear_keyword


def infer_linear_keyword(model: nn.Module, arch_name: str) -> str:
    """
    Infer the final classifier attribute name.
    """
    vit_like_names = [
        "VisionTransformer",
        "ViT_B_16_Weights",
        "ViT_B_32_Weights",
        "ViT_L_16_Weights",
        "ViT_L_32_Weights",
        "ViT_H_14_Weights",
        "vit_b_16",
        "vit_b_32",
        "vit_l_16",
        "vit_l_32",
        "vit_h_14",
    ]

    if arch_name in vit_like_names:
        return "heads"

    if "vit" in arch_name or "transformer" in arch_name or "swin_" in arch_name:
        if hasattr(model, "head"):
            return "head"

    if hasattr(model, "classifier"):
        return "classifier"
    if hasattr(model, "head"):
        return "head"
    if hasattr(model, "fc"):
        return "fc"

    raise ValueError(f"Unable to determine final layer for architecture '{arch_name}'.")


def change_last_layer(model: nn.Module, out_dim: int, linear_keyword: str) -> nn.Module:
    """
    Replace the final classification layer.
    """
    if linear_keyword == "fc" and hasattr(model, "fc"):
        model.fc = nn.Linear(model.fc.in_features, out_dim)

    elif linear_keyword == "classifier" and hasattr(model, "classifier"):
        if isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, out_dim)
        elif isinstance(model.classifier, nn.Sequential):
            if not isinstance(model.classifier[-1], nn.Linear):
                raise ValueError("model.classifier[-1] is not nn.Linear.")
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, out_dim)
        else:
            raise ValueError("Unsupported classifier structure.")

    elif linear_keyword == "head" and hasattr(model, "head"):
        if not isinstance(model.head, nn.Linear):
            raise ValueError("model.head is not nn.Linear.")
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, out_dim)

    elif linear_keyword == "heads":
        model = change_vit_num_classes(model, num_classes=out_dim)

    else:
        raise ValueError(
            f"Could not change final layer for linear_keyword='{linear_keyword}'."
        )

    return model


def change_vit_num_classes(model: nn.Module, num_classes: int) -> nn.Module:
    """
    Replace final head of torchvision VisionTransformer.
    """
    if hasattr(model, "heads") and isinstance(model.heads, nn.Sequential):
        if "head" in model.heads._modules:
            orig_head = model.heads._modules["head"]
            if not isinstance(orig_head, nn.Linear):
                raise ValueError("VisionTransformer heads['head'] is not nn.Linear.")
            in_features = orig_head.in_features
            model.heads._modules["head"] = nn.Linear(in_features, num_classes)
            model.num_classes = num_classes
            return model

    raise ValueError(
        f"Could not change the final layer for the provided model. Model structure: {model}"
    )


def get_output_dim(dataset_name: str) -> int:
    """
    Return the number of output classes from dataset_name.
    """
    if dataset_name in ["texture2shape_miniecoset", "rgbd_texture2shape_miniecoset"]:
        return 112
    if dataset_name in ["ecoset_square256", "ecoset_square256_patches"]:
        return 565
    if dataset_name == "imagenet":
        return 1000
    if dataset_name == "facescrub":
        return 118
    raise ValueError(f"dataset_name '{dataset_name}' not supported.")


def change_first_layer_channels(model: nn.Module, channels: int = 4) -> nn.Module:
    """
    Replace first conv layer to accept custom number of channels.
    Only supports models with conv1 like ResNet-style backbones.
    """
    if not hasattr(model, "conv1"):
        raise ValueError("Model does not have attribute 'conv1'.")

    original_conv = model.conv1
    if not isinstance(original_conv, nn.Conv2d):
        raise ValueError("model.conv1 is not nn.Conv2d.")

    original_weights = original_conv.weight.data.clone()

    model.conv1 = nn.Conv2d(
        channels,
        original_conv.out_channels,
        kernel_size=original_conv.kernel_size,
        stride=original_conv.stride,
        padding=original_conv.padding,
        bias=False,
    )

    with torch.no_grad():
        num_copy = min(3, channels)
        model.conv1.weight[:, :num_copy, :, :] = original_weights[:, :num_copy, :, :]
        if channels > 3:
            nn.init.normal_(model.conv1.weight[:, 3:, :, :], mean=0.0, std=0.01)

    return model


# ============================================================
# State dict normalization
# ============================================================
def normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Normalize checkpoint keys to bare-model format.
    Removes common wrappers such as:
      - module.
      - _orig_mod.
      - module._orig_mod.
    """
    normalized = {}

    for key, value in state_dict.items():
        new_key = key
        if new_key.startswith("module._orig_mod."):
            new_key = new_key[len("module._orig_mod."):]
        elif new_key.startswith("_orig_mod."):
            new_key = new_key[len("_orig_mod."):]
        elif new_key.startswith("module."):
            new_key = new_key[len("module."):]
        normalized[new_key] = value

    return normalized


def get_bare_model(model: nn.Module) -> nn.Module:
    """
    Return the underlying model if wrapped by DDP/DataParallel.
    """
    return model.module if hasattr(model, "module") else model


def load_model_state(model: nn.Module, state_dict: Dict[str, Any], strict: bool = True, logger=None):
    """
    Load normalized state_dict into bare model.
    """
    bare_model = get_bare_model(model)
    normalized = normalize_state_dict_keys(state_dict)
    msg = bare_model.load_state_dict(normalized, strict=strict)
    if logger is not None:
        logger.info(
            f"Model state loaded (strict={strict}). "
            f"missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)}"
        )
    return msg


# ============================================================
# Optional pretrained init
# ============================================================
def load_pretrained_weights_if_any(args, model, linear_keyword):
    """
    Optionally load pretrained weights.
    This is for initialization only, not for full training resume.
    """
    if not args.pretrained:
        return

    if not os.path.isfile(args.pretrained):
        print(f"=> no checkpoint found at '{args.pretrained}'")
        return

    print(f"=> loading pretrained checkpoint '{args.pretrained}'")
    checkpoint = torch.load(args.pretrained, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    msg = load_model_state(model, state_dict, strict=False)
    print(
        f"=> loaded '{args.pretrained}' "
        f"(missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)})"
    )


# ============================================================
# Optimizer / AMP / DDP
# ============================================================
def compute_effective_lr(args) -> float:
    """
    Compute effective lr from base lr and local batch size.
    Keeps args.lr unchanged.
    """
    return float(args.lr) * float(args.batch_size_per_gpu) / 512.0


def build_optimizer_and_scaler(args, model):
    """
    Convert model to SyncBN + DDP, then build optimizer and scaler.
    Returns:
        model, optimizer, scaler, effective_lr
    """
    effective_lr = compute_effective_lr(args)

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu],
        output_device=args.gpu,
        broadcast_buffers=False,
    )

    if args.optimizer == "lars":
        optimizer = dvd.simclr.optimizer.LARS(
            model.parameters(),
            effective_lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
        )
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=effective_lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=effective_lr,
        )
    else:
        raise ValueError(f"Unknown optimizer '{args.optimizer}'")

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    return model, optimizer, scaler, effective_lr


# ============================================================
# RNG restore
# ============================================================
def restore_rng_state(checkpoint: Dict[str, Any], logger=None):
    """
    Restore RNG states from checkpoint if available.
    """
    try:
        if checkpoint.get("torch_rng_state", None) is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"])

        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state_all", None) is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])

        if checkpoint.get("numpy_rng_state", None) is not None:
            np.random.set_state(checkpoint["numpy_rng_state"])

        if checkpoint.get("python_rng_state", None) is not None:
            random.setstate(checkpoint["python_rng_state"])

        if logger is not None:
            logger.info("RNG states restored from checkpoint.")
    except Exception as e:
        if logger is not None:
            logger.info(f"Warning: failed to restore RNG states exactly: {e}")


def safe_load_scaler_state(scaler, checkpoint: Dict[str, Any], logger=None):
    """
    Restore GradScaler state if available and valid.
    """
    if scaler is None:
        return

    scaler_state = checkpoint.get("scaler", None)
    if scaler_state is None:
        return

    if isinstance(scaler_state, dict) and len(scaler_state) == 0:
        if logger is not None:
            logger.info("Empty scaler state in checkpoint; skipping scaler restore.")
        return

    try:
        scaler.load_state_dict(scaler_state)
        if logger is not None:
            logger.info("GradScaler state restored.")
    except RuntimeError as e:
        if logger is not None:
            logger.info(f"Failed to restore scaler state, skipping it: {e}")


# ============================================================
# Distributed resume sync
# ============================================================
def sync_resume_state(args):
    """
    Broadcast a few important scalar resume states from rank 0 to all ranks.
    """
    if not dist.is_available() or not dist.is_initialized():
        return

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu is not None else "cpu"
    )

    start_epoch_tensor = torch.tensor([args.start_epoch], dtype=torch.long, device=device)
    best_acc1_tensor = torch.tensor([float(args.best_acc1)], dtype=torch.float32, device=device)

    dist.broadcast(start_epoch_tensor, src=0)
    dist.broadcast(best_acc1_tensor, src=0)

    args.start_epoch = int(start_epoch_tensor.item())
    args.best_acc1 = float(best_acc1_tensor.item())


# ============================================================
# Resume config checks
# ============================================================
def _log_config_mismatch(name: str, current_value: Any, ckpt_value: Any, logger=None):
    msg = f"[Resume config mismatch] {name}: current={current_value} | checkpoint={ckpt_value}"
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)


def check_resume_compatibility(args, checkpoint: Dict[str, Any], logger=None):
    """
    Warn when important resume-time hyperparameters differ from checkpoint.
    """
    keys_to_check = [
        "arch",
        "dataset_name",
        "image_size",
        "optimizer",
        "batch_size_per_gpu",
        "months_per_epoch",
        "contrast_amplitude_beta",
        "contrast_amplitude_lambda",
        "time_order",
        "development_strategy",
        "lr",
        "lr_scheduler",
        "label_smoothing",
    ]

    checkpoint_args = checkpoint.get("args", {})
    if not isinstance(checkpoint_args, dict):
        checkpoint_args = {}

    for key in keys_to_check:
        if key in checkpoint_args and hasattr(args, key):
            current_value = getattr(args, key)
            ckpt_value = checkpoint_args[key]
            if current_value != ckpt_value:
                _log_config_mismatch(key, current_value, ckpt_value, logger)


# ============================================================
# Checkpoint loading helpers
# ============================================================
def _load_checkpoint_file(checkpoint_path: str, args):
    loc = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=loc)
    return checkpoint


def resume_from_checkpoint_path(
    checkpoint_path: str,
    args,
    model,
    optimizer=None,
    scaler=None,
    logger=None,
):
    """
    Resume from a specific checkpoint path.
    Returns:
        checkpoint dict
    """
    if logger is not None:
        logger.info(f"Loading checkpoint from '{checkpoint_path}'")

    checkpoint = _load_checkpoint_file(checkpoint_path, args)

    state_dict = checkpoint.get("state_dict", checkpoint)
    load_model_state(model, state_dict, strict=True, logger=logger)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if logger is not None:
            logger.info("Optimizer state restored.")

    if scaler is not None:
        safe_load_scaler_state(scaler, checkpoint, logger)

    if "epoch" in checkpoint:
        args.start_epoch = int(checkpoint["epoch"]) + 1

    if "best_acc1" in checkpoint:
        args.best_acc1 = float(checkpoint["best_acc1"])

    restore_rng_state(checkpoint, logger)
    check_resume_compatibility(args, checkpoint, logger)

    if logger is not None:
        logger.info(
            f"Resume success: saved epoch={checkpoint.get('epoch', 'N/A')}, "
            f"start_epoch={args.start_epoch}, best_acc1={args.best_acc1}"
        )

    return checkpoint


def find_latest_checkpoint(log_dir: Optional[str], logger=None) -> Optional[str]:
    """
    Find the most suitable checkpoint path under log_dir/weights.
    Priority:
        1. checkpoint_last.pth
        2. latest checkpoint_{epoch}.pth
        3. checkpoint.pth
    """
    if log_dir is None:
        if logger is not None:
            logger.info("log_dir is None; cannot search latest checkpoint.")
        return None

    weights_dir = os.path.join(log_dir, "weights")
    if not os.path.isdir(weights_dir):
        if logger is not None:
            logger.info(f"No weights directory at '{weights_dir}'.")
        return None

    last_ckpt = os.path.join(weights_dir, "checkpoint_last.pth")
    if os.path.isfile(last_ckpt):
        return last_ckpt

    numbered_ckpts = []
    for fname in os.listdir(weights_dir):
        match = re.match(r"checkpoint_(\d+)\.pth$", fname)
        if match:
            numbered_ckpts.append((int(match.group(1)), fname))

    if numbered_ckpts:
        numbered_ckpts.sort(key=lambda x: x[0], reverse=True)
        return os.path.join(weights_dir, numbered_ckpts[0][1])

    plain_ckpt = os.path.join(weights_dir, "checkpoint.pth")
    if os.path.isfile(plain_ckpt):
        return plain_ckpt

    return None


def resume_checkpoint_if_any(args, model, optimizer, scaler, logger, log_dir):
    """
    Resume training if args.resume is specified, otherwise try latest checkpoint in log_dir.
    Returns:
        checkpoint or None
    """
    checkpoint = None

    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = resume_from_checkpoint_path(
                args.resume, args, model, optimizer, scaler, logger
            )
        else:
            if logger is not None:
                logger.info(f"No checkpoint found at '{args.resume}'")
    else:
        latest_ckpt = find_latest_checkpoint(log_dir, logger)
        if latest_ckpt is not None:
            checkpoint = resume_from_checkpoint_path(
                latest_ckpt, args, model, optimizer, scaler, logger
            )
        else:
            if logger is not None:
                logger.info("No previous checkpoint found; starting from scratch.")

    sync_resume_state(args)
    return checkpoint


def load_checkpoint(model, model_path=None, optimizer=None, log_dir=None, args=None, logger=None):
    """
    Load a checkpoint for evaluation or manual use.
    """
    checkpoint_path = None

    if model_path and os.path.isfile(model_path):
        checkpoint_path = model_path
    elif log_dir:
        candidate = os.path.join(log_dir, "weights", "checkpoint_best.pth")
        if os.path.isfile(candidate):
            checkpoint_path = candidate
        else:
            raise ValueError(f"No checkpoint found at '{candidate}'")
    else:
        raise ValueError("No checkpoint path provided.")

    if logger is not None:
        logger.info(f"Loading checkpoint '{checkpoint_path}'")

    loc = (
        f"cuda:{args.gpu}"
        if args is not None and hasattr(args, "gpu") and args.gpu is not None
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(checkpoint_path, map_location=loc)

    state_dict = checkpoint.get("state_dict", checkpoint)
    load_model_state(model, state_dict, strict=True, logger=logger)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if args is not None and "best_acc1" in checkpoint:
        args.best_acc1 = checkpoint["best_acc1"]

    return checkpoint


# ============================================================
# Checkpoint save helpers
# ============================================================
def get_model_state_for_saving(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Always save bare model weights.
    """
    bare_model = get_bare_model(model)
    return bare_model.state_dict()