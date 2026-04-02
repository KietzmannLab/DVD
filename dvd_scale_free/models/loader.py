import os
import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torchvision.models as torchvision_models
import dvd_scale_free.simclr.optimizer

import timm
from torchvision import models as torchvision_models

# add custom models to torchvision_models
from . import custom_models
torchvision_models.__dict__['custom_cnn'] = custom_models.custom_cnn


def create_model(args, logger=None):
    """
    Creates the model (from torchvision if possible, otherwise from timm)
    and adjusts the final layer based on dataset_name.
    """
    if logger:
        logger.info(f"Creating model '{args.arch}'")

    # Try loading from torchvision first
    if args.arch in torchvision_models.__dict__:
        model = torchvision_models.__dict__[args.arch](pretrained=False)

        if args.arch in [
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
        ]:
            linear_keyword = "heads"
        elif "vit" in args.arch or "transformer" in args.arch or "swin_" in args.arch:
            linear_keyword = "head"
        else:
            if hasattr(model, 'classifier'):
                linear_keyword = "classifier"
            elif hasattr(model, 'head'):
                linear_keyword = "head"
            elif hasattr(model, 'fc'):
                linear_keyword = "fc"
            else:
                raise ValueError(f"Unable to determine final layer for {args.arch}.")
    else:
        try:
            model = timm.create_model(args.arch, pretrained=False)
        except Exception as e:
            raise ValueError(
                f"Model architecture '{args.arch}' not found in torchvision or timm."
            ) from e

        if "vit" in args.arch or "transformer" in args.arch or "swin_" in args.arch:
            linear_keyword = "head"
        else:
            if hasattr(model, 'fc'):
                linear_keyword = "fc"
            elif hasattr(model, 'classifier'):
                linear_keyword = "classifier"
            elif hasattr(model, 'head'):
                linear_keyword = "head"
            else:
                raise ValueError(f"Unable to determine final layer for {args.arch}.")

    if args.dataset_name in ['rgbd_texture2shape_miniecoset']:
        model = change_first_layer_channels(args, model, channels=4)

    out_dim = get_output_dim(args.dataset_name)

    try:
        model = change_last_layer(args, model, out_dim, linear_keyword)
    except Exception:
        model = change_vit_num_classes(model, num_classes=out_dim)

    return model, linear_keyword


def change_last_layer(args, model, out_dim, linear_keyword):
    """
    Changes the last layer of the model to match out_dim based on linear_keyword.
    """
    if linear_keyword == "fc" and hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, out_dim)

    elif linear_keyword == "classifier" and hasattr(model, 'classifier'):
        if isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, out_dim)
        elif isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, out_dim)
        else:
            raise ValueError("Unsupported classifier structure.")

    elif linear_keyword == "head" and hasattr(model, 'head'):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, out_dim)

    else:
        raise ValueError(
            f"Could not change final layer for arch '{args.arch}' with linear_keyword='{linear_keyword}'."
        )

    return model


def change_vit_num_classes(model: nn.Module, num_classes: int) -> nn.Module:
    """
    Replace the final classification head of a VisionTransformer model
    with a new linear layer having the specified number of output classes.
    """
    if hasattr(model, "heads") and isinstance(model.heads, nn.Sequential):
        if "head" in model.heads._modules:
            orig_head = model.heads._modules["head"]
            in_features = orig_head.in_features
            model.heads._modules["head"] = nn.Linear(in_features, num_classes)
            model.num_classes = num_classes
            return model

    raise ValueError(
        f"Could not change the final layer for the provided model. "
        f"Expected 'heads' to contain key 'head'. Model structure: {model}"
    )


def remove_prefix(state_dict: dict) -> dict:
    """
    Normalize state_dict keys for DDP or single-GPU use.
    """
    use_multi_gpu = torch.cuda.device_count() > 1

    new_state = {}
    for k, v in state_dict.items():
        if k.startswith("module._orig_mod."):
            if use_multi_gpu:
                k = "module." + k[len("module._orig_mod."):]
            else:
                k = k[len("module._orig_mod."):]
        elif k.startswith("module.") and not use_multi_gpu:
            k = k[len("module."):]
        new_state[k] = v

    return new_state


def load_pretrained_weights_if_any(args, model, linear_keyword):
    if not args.pretrained:
        return
    if not os.path.isfile(args.pretrained):
        print(f"=> no checkpoint found at '{args.pretrained}'")
        return

    print(f"=> loading checkpoint '{args.pretrained}'")
    checkpoint = torch.load(args.pretrained, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = remove_prefix(state_dict)

    msg = model.load_state_dict(state_dict, strict=False)
    print(
        f"=> loaded '{args.pretrained}' "
        f"(missing={len(msg.missing_keys)}, unexpected={len(msg.unexpected_keys)})"
    )


def build_optimizer_and_scaler(args, model):
    """
    Builds the optimizer and AMP GradScaler in V1 style.
    """
    # infer learning rate before changing batch size
    args.lr = args.lr * args.batch_size_per_gpu / 512

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(args.gpu)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[args.gpu],
        output_device=args.gpu,
        broadcast_buffers=False,
    )

    if args.optimizer == "lars":
        optimizer = dvd_scale_free.simclr.optimizer.LARS(
            model.parameters(),
            args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
        )
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(), args.lr, weight_decay=args.weight_decay
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        raise ValueError(f"Unknown optimizer {args.optimizer}")

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
    return model, optimizer, scaler


def restore_rng_state(checkpoint, logger=None):
    """
    Restore RNG states if they exist in the checkpoint.
    """
    try:
        if "torch_rng_state" in checkpoint and checkpoint["torch_rng_state"] is not None:
            torch.set_rng_state(checkpoint["torch_rng_state"])

        if (
            torch.cuda.is_available()
            and "cuda_rng_state_all" in checkpoint
            and checkpoint["cuda_rng_state_all"] is not None
        ):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])

        if "numpy_rng_state" in checkpoint and checkpoint["numpy_rng_state"] is not None:
            np.random.set_state(checkpoint["numpy_rng_state"])

        if "python_rng_state" in checkpoint and checkpoint["python_rng_state"] is not None:
            random.setstate(checkpoint["python_rng_state"])

        if logger is not None:
            logger.info("RNG states restored from checkpoint.")
    except Exception as e:
        if logger is not None:
            logger.info(f"Warning: failed to restore RNG states exactly: {e}")


def sync_resume_state(args):
    """
    Broadcast start_epoch and best_acc1 from rank 0 to all ranks.
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


def resume_checkpoint_if_any(args, model, optimizer, scaler, logger, log_dir):
    """
    Resume from checkpoint if possible.
    Must be called on all ranks, not only rank 0.
    """
    if args.resume:
        if os.path.isfile(args.resume):
            if logger is not None:
                logger.info("Loading checkpoint '{}'".format(args.resume))

            loc = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
            checkpoint = torch.load(args.resume, map_location=loc)

            state_dict = checkpoint.get("state_dict", checkpoint)
            try:
                model.load_state_dict(remove_prefix(state_dict))
            except Exception:
                model.load_state_dict(state_dict)

            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])

            if scaler is not None and checkpoint.get("scaler") is not None:
                scaler.load_state_dict(checkpoint["scaler"])

            args.start_epoch = checkpoint["epoch"] + 1
            if "best_acc1" in checkpoint:
                args.best_acc1 = checkpoint["best_acc1"]

            restore_rng_state(checkpoint, logger)

            if logger is not None:
                logger.info(
                    "Loaded checkpoint '{}' (saved epoch {}, resuming from epoch {}).".format(
                        args.resume, checkpoint["epoch"], args.start_epoch
                    )
                )
        else:
            if logger is not None:
                logger.info("No checkpoint found at '{}'".format(args.resume))
    else:
        resume_latest_checkpoint(args, model, optimizer, scaler, logger, log_dir)

    sync_resume_state(args)


def load_checkpoint(model, model_path=None, optimizer=None, log_dir=None, args=None):
    """
    Load a checkpoint from a specified model path or from a default log directory.
    """
    if model_path and os.path.isfile(model_path):
        print("Loading checkpoint '{}'".format(model_path))
        loc = f"cuda:{args.gpu}" if hasattr(args, 'gpu') else (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        checkpoint = torch.load(model_path, map_location=loc)
        try:
            model.load_state_dict(remove_prefix(checkpoint["state_dict"]))
        except Exception:
            model.load_state_dict(checkpoint["state_dict"])

        if optimizer and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

    elif log_dir:
        default_checkpoint = os.path.join(log_dir, "weights", "checkpoint_best.pth")
        print(
            "No checkpoint found at '{}', loading default checkpoint '{}'".format(
                model_path, default_checkpoint
            )
        )
        if os.path.isfile(default_checkpoint):
            loc = f"cuda:{args.gpu}" if args else (
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
            checkpoint = torch.load(default_checkpoint, map_location=loc)
            try:
                model.load_state_dict(remove_prefix(checkpoint["state_dict"]))
            except Exception:
                model.load_state_dict(checkpoint["state_dict"], strict=True)

            if optimizer and "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if args is not None and 'best_acc1' in checkpoint:
                args.best_acc1 = checkpoint['best_acc1']
                print(f"Loaded best_acc1: {args.best_acc1}")
        else:
            raise ValueError("No checkpoint found at '{}'".format(default_checkpoint))
    else:
        raise ValueError("No checkpoint found at '{}'".format(model_path))


def resume_latest_checkpoint(args, model, optimizer, scaler, logger, log_dir):
    """
    Finds and loads the latest checkpoint from the 'weights' directory in log_dir.
    """
    if log_dir is None:
        if logger is not None:
            logger.info("log_dir is None; starting from scratch.")
        sync_resume_state(args)
        return

    weights_dir = os.path.join(log_dir, "weights")
    if not os.path.exists(weights_dir):
        if logger is not None:
            logger.info(f"No 'weights' directory found at {weights_dir}; starting from scratch.")
        sync_resume_state(args)
        return

    last_checkpoint_path = os.path.join(weights_dir, "checkpoint_last.pth")
    checkpoint_path = None

    if os.path.isfile(last_checkpoint_path):
        checkpoint_path = last_checkpoint_path
        if logger is not None:
            logger.info(f"Found 'checkpoint_last.pth' at '{last_checkpoint_path}', loading that.")
    else:
        all_ckpts = [
            f for f in os.listdir(weights_dir)
            if f.endswith(".pth")
            and not f.startswith("checkpoint_init")
            and not f.startswith("checkpoint_best")
            and f != "checkpoint_last.pth"
        ]

        if not all_ckpts:
            if logger is not None:
                logger.info("No numbered or rolling checkpoints found; starting from scratch.")
            sync_resume_state(args)
            return

        latest_epoch = -1
        latest_ckpt_file = None
        for ckpt_file in all_ckpts:
            m = re.match(r"checkpoint_(\d+)\.pth", ckpt_file)
            if m:
                epoch_num = int(m.group(1))
                if epoch_num > latest_epoch:
                    latest_epoch = epoch_num
                    latest_ckpt_file = ckpt_file

        if latest_ckpt_file is None:
            if "checkpoint.pth" in all_ckpts:
                latest_ckpt_file = "checkpoint.pth"
                if logger is not None:
                    logger.info("No numbered checkpoints found; using 'checkpoint.pth'.")
            else:
                if logger is not None:
                    logger.info("No valid checkpoint file found; starting from scratch.")
                sync_resume_state(args)
                return

        checkpoint_path = os.path.join(weights_dir, latest_ckpt_file)

    if logger is not None:
        logger.info(f"Loading checkpoint from '{checkpoint_path}' ...")

    loc = f"cuda:{args.gpu}" if args.gpu is not None else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=loc)

    state_dict = checkpoint.get("state_dict", checkpoint)
    try:
        model.load_state_dict(remove_prefix(state_dict))
    except Exception:
        model.load_state_dict(state_dict)

    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])

    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])

    args.start_epoch = checkpoint["epoch"] + 1
    if "best_acc1" in checkpoint:
        args.best_acc1 = checkpoint["best_acc1"]

    restore_rng_state(checkpoint, logger)

    if logger is not None:
        logger.info(
            f"Successfully loaded checkpoint '{checkpoint_path}' "
            f"(saved epoch {checkpoint['epoch']}, resuming from epoch {args.start_epoch})."
        )

    sync_resume_state(args)


def get_output_dim(dataset_name):
    """
    Return the number of output classes based on the dataset name.
    """
    if dataset_name in ["texture2shape_miniecoset", "rgbd_texture2shape_miniecoset"]:
        return 112
    elif dataset_name in ["ecoset_square256", "ecoset_square256_patches"]:
        return 565
    elif dataset_name == "imagenet":
        return 1000
    elif dataset_name == "facescrub":
        return 118
    else:
        raise ValueError(f"dataset_name: {dataset_name} not supported")


def change_first_layer_channels(args, model, channels=4):
    original_conv = model.conv1
    original_weights = original_conv.weight.data

    model.conv1 = torch.nn.Conv2d(
        channels, 64, kernel_size=7, stride=2, padding=3, bias=False
    )

    model.conv1.weight.data[:, :3, :, :] = original_weights
    torch.nn.init.normal_(model.conv1.weight.data[:, 3:, :, :], mean=0.0, std=0.01)

    return model