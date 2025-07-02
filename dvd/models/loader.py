import os
import re
import torch
import torch.nn as nn
import torchvision.models as torchvision_models
# import dvd.models.vits
import dvd.simclr.optimizer

import torch.nn as nn
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
        # Load pretrained model from torchvision
        model = torchvision_models.__dict__[args.arch](pretrained=False)
        # import pdb;pdb.set_trace()
        # Decide which layer name to change
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
            # Only for ViT models in torchvsion models
            linear_keyword = "heads"
            pass 
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
        # Otherwise, fallback to timm for e.g., DeiT, Swin, etc.
        # e.g deit_base_patch16_224 and more see 
        try:
            model = timm.create_model(args.arch, pretrained=False)
        except Exception as e:
            raise ValueError(
                f"Model architecture '{args.arch}' not found in torchvision or timm."
            ) from e

        # Decide which layer name to change
        if "vit" in args.arch or "transformer" in args.arch or "swin_" in args.arch:
            linear_keyword = "head"
        else:
            # For non-ViT or transformer models from timm, we can guess:
            if hasattr(model, 'fc'):
                linear_keyword = "fc"
            elif hasattr(model, 'classifier'):
                linear_keyword = "classifier"
            elif hasattr(model, 'head'):
                linear_keyword = "head"
            else:
                raise ValueError(f"Unable to determine final layer for {args.arch}.")

    # Select how many classes we want based on dataset_name
    if args.dataset_name == "texture2shape_miniecoset":
        out_dim = 112
    elif args.dataset_name in ["ecoset_square256", "ecoset_square256_patches"]:
        out_dim = 565
    elif args.dataset_name == "imagenet":
        out_dim = 1000
    elif  args.dataset_name == 'facescrub':
        out_dim = 118
    else:
        raise ValueError(f"dataset_name: {args.dataset_name} not supported")

    # Update the final layer to match our number of classes
    try:
        model = change_last_layer(args, model, out_dim, linear_keyword)
    except:
        # Only for ViT models in torchvsion models
        model = change_vit_num_classes(model, num_classes=out_dim)

    return model, linear_keyword

def change_last_layer(args, model, out_dim, linear_keyword):
    """
    Changes the last layer of the model to match out_dim based on linear_keyword.
    """
    # For ResNet/ResNeXt
    if linear_keyword == "fc" and hasattr(model, 'fc'):
        model.fc = nn.Linear(model.fc.in_features, out_dim)

    # For e.g. AlexNet, VGG, or any model with 'classifier' as a Sequential or Linear
    elif linear_keyword == "classifier" and hasattr(model, 'classifier'):
        # If classifier is a single linear layer
        if isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, out_dim)
        # If classifier is a Sequential (like in VGG/AlexNet)
        elif isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, out_dim)
        else:
            raise ValueError("Unsupported classifier structure.")

    # For ViT / Swin / DeiT typically using 'head'
    elif linear_keyword == "head" and hasattr(model, 'head'):
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, out_dim)

    else:
        # If none of the above matched, raise an error
        raise ValueError(
            f"Could not change final layer for arch '{args.arch}' with linear_keyword='{linear_keyword}'."
        )

    return model


def change_vit_num_classes(model: nn.Module, num_classes: int) -> nn.Module:
    """
    Replace the final classification head of a VisionTransformer model (e.g. vit_b_16)
    with a new linear layer having the specified number of output classes.

    Args:
        model (nn.Module): An instance of VisionTransformer (e.g. output of vit_b_16())
        num_classes (int): The desired number of classes for the new classifier.
        
    Returns:
        nn.Module: The updated model with a new final classification layer.
    """
    # Ensure the model has the attribute 'heads'
    if hasattr(model, "heads") and isinstance(model.heads, nn.Sequential):
        # Check if the Sequential container has "head" as a key
        if "head" in model.heads._modules:
            # Retrieve the original head layer
            orig_head = model.heads._modules["head"]
            in_features = orig_head.in_features
            # Replace it with a new linear layer mapping to num_classes outputs
            model.heads._modules["head"] = nn.Linear(in_features, num_classes)
            # Optionally update the attribute model.num_classes if defined
            model.num_classes = num_classes
            return model

    # If the expected structure is not found, raise an error.
    raise ValueError(
        f"Could not change the final layer for the provided model. "
        f"Expected 'heads' to contain key 'head'. Model structure: {model}"
    )

# def remove_prefix(state_dict):
#     """Strip the DataParallel prefix from state dict keys if it exists."""
#     use_multi_gpu = torch.cuda.device_count() > 1

#     new_state_dict = {}
#     for k, v in state_dict.items():
#         # import pdb; pdb.set_trace()
#         if k.startswith("module._orig_mod."):
#             if use_multi_gpu:
#                 new_key = k[:len("module.")]+k[len("module._orig_mod."):]
#             else:
#                 new_key = k[len("module._orig_mod."):]
#         elif k.startswith("module."):
#             new_key = k[len("module."):]
#         else:
#             new_key = k
#         new_state_dict[new_key] = v
#     return new_state_dict


def remove_prefix(state_dict: dict) -> dict:
    """
    Normalize state_dict keys for DataParallel or single-GPU use.

    - If torch.cuda.device_count() > 1: assume multi-GPU and only strip
      the `. _orig_mod.` infix, keeping the `module.` prefix.
    - Otherwise: strip off `module.` entirely.

    This lets you seamlessly load checkpoints across different GPU setups.
    """
    use_multi_gpu = torch.cuda.device_count() > 1

    new_state = {}
    for k, v in state_dict.items():
        # import pdb;pdb.set_trace()
        if k.startswith("module._orig_mod."):
            if use_multi_gpu:
                # Keep 'module.' but remove '_orig_mod.' infix
                k = "module." + k[len("module._orig_mod.") :]
            else:
                # Remove the whole 'module._orig_mod.' prefix
                k = k[len("module._orig_mod.") :]
        elif k.startswith("module.") and not use_multi_gpu:
            # Remove 'module.' when running on single-GPU/CPU
            k = k[len("module.") :]
        new_state[k] = v

    return new_state
    
def load_pretrained_weights_if_any(args, model, linear_keyword):
    """
    Loads pretrained weights into model if args.pretrained is specified.
    """
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            try:
                model.load_state_dict(remove_prefix(checkpoint["state_dict"]))
            except:
                model.load_state_dict(checkpoint["state_dict"])
            for k in list(state_dict.keys()):
                # retain only base_encoder up to before the embedding layer
                if k.startswith("module.encoder") and not k.startswith(
                    "module.encoder.%s" % linear_keyword
                ):
                    # remove prefix
                    state_dict[k[len("module.encoder.") :]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

            msg = model.load_state_dict(state_dict, strict=False)
            assert set(msg.missing_keys) == {
                "%s.weight" % linear_keyword,
                "%s.bias" % linear_keyword,
            }

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))


def build_optimizer_and_scaler(args, model):
    """
    Builds the optimizer (LARS or AdamW) and FP16 scaler.
    """
    # infer learning rate before changing batch size
    args.lr = args.lr * args.batch_size_per_gpu / 512 # For SimCLR,  256 is default

    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model)

    if args.optimizer == "lars":
        optimizer = dvd.simclr.optimizer.LARS(
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

    scaler = torch.cuda.amp.GradScaler()
    return model, optimizer, scaler




def resume_checkpoint_if_any(args, model, optimizer, scaler, logger, log_dir):
    """
    Resumes from checkpoint if args.resume is provided.
    """
    # Case 1: If user explicitly gave --resume
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("Loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = "cuda:{}".format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            try:
                model.load_state_dict(remove_prefix(checkpoint["state_dict"]))
            except:
                model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scaler.load_state_dict(checkpoint["scaler"])
            logger.info(
                "Loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(args.resume))
    
    else:
        # Case 2: No --resume given, or file not found => search automatically
        resume_latest_checkpoint(args, model, optimizer, scaler, logger, log_dir)


def load_checkpoint(model, model_path=None, optimizer=None, log_dir=None, args=None):
    """
    Load a checkpoint from a specified model path or from a default log directory.
    This version removes any "module." or "module._orig_mod." prefixes from the state dict,
    which can appear if the model was saved using DataParallel.
    """

    # Choose the checkpoint source: model_path takes precedence.
    if model_path and os.path.isfile(model_path):
        print("Loading checkpoint '{}'".format(model_path))
        loc = "cuda:{}".format(args.gpu) if args else ('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(model_path, map_location=loc)
        # import pdb;pdb.set_trace()
        # Remove unwanted prefixes from keys.
        try:
            model.load_state_dict(remove_prefix(checkpoint["state_dict"]))
        except:
            model.load_state_dict(checkpoint["state_dict"])
        if optimizer and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])

    elif log_dir:
        # Load the default checkpoint from log_dir if no model_path was provided.
        default_checkpoint = os.path.join(log_dir, "weights", "checkpoint_best.pth")
        print("No checkpoint found at '{}', loading default checkpoint '{}'".format(model_path, default_checkpoint))
        if os.path.isfile(default_checkpoint):
            loc = "cuda:{}".format(args.gpu) if args else ('cuda' if torch.cuda.is_available() else 'cpu')
            checkpoint = torch.load(default_checkpoint, map_location=loc)
            try:
                model.load_state_dict(remove_prefix(checkpoint["state_dict"]))
            except:
                model.load_state_dict(checkpoint["state_dict"])
            model.load_state_dict(state_dict, strict=True)
            if optimizer and "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            if 'best_acc1' in checkpoint:
                args.best_acc1 = checkpoint['best_acc1']
                print(f"Loaded best_acc1: {args.best_acc1}")
        else:
            raise ValueError("No checkpoint found at '{}'".format(default_checkpoint))
    else:
        raise ValueError("No checkpoint found at '{}'".format(model_path))
    

def resume_latest_checkpoint(args, model, optimizer, scaler, logger, log_dir):
    """
    Finds and loads the latest checkpoint from the 'weights' directory in log_dir.
    
    Priority:
      1. If 'checkpoint_last.pth' exists, load that.
      2. Else, search for numbered checkpoints (e.g., "checkpoint_10.pth").
      3. If no numbered checkpoint is found, fallback to 'checkpoint.pth'.
    
    If no valid checkpoint is found, logs an informational message and starts from scratch.
    """
    
    weights_dir = os.path.join(log_dir, "weights")
    if not os.path.exists(weights_dir):
        logger.info(f"No 'weights' directory found at {weights_dir}; starting from scratch.")
        return

    # First try to find "checkpoint_last.pth"
    last_checkpoint_path = os.path.join(weights_dir, "checkpoint_last.pth")
    if os.path.isfile(last_checkpoint_path):
        logger.info(f"Found 'checkpoint_last.pth' at '{last_checkpoint_path}', loading that.")
        loc = f"cuda:{args.gpu}" if args.gpu is not None else None
        checkpoint = torch.load(last_checkpoint_path, map_location=loc)
        try:
            model.load_state_dict(checkpoint["state_dict"])
        except:
            model.load_state_dict(remove_prefix(checkpoint["state_dict"]))
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        args.start_epoch = checkpoint["epoch"]
        if "best_acc1" in checkpoint:
            args.best_acc1 = checkpoint["best_acc1"]    
        logger.info(f"Successfully loaded 'checkpoint_last.pth' (epoch {checkpoint['epoch']}).")
        return

    # Otherwise, collect all other .pth files (ignoring checkpoint_init and checkpoint_best)
    all_ckpts = [
        f for f in os.listdir(weights_dir)
        if f.endswith(".pth")
        and not f.startswith("checkpoint_init")
        and not f.startswith("checkpoint_best")
        and f != "checkpoint_last.pth"
    ]
    if not all_ckpts:
        logger.info("No numbered or rolling checkpoints found; starting from scratch.")
        return

    # Search for the latest numbered checkpoint (e.g., "checkpoint_10.pth")
    latest_epoch = -1
    latest_ckpt_file = None
    for ckpt_file in all_ckpts:
        m = re.match(r"checkpoint_(\d+)\.pth", ckpt_file)
        if m:
            epoch_num = int(m.group(1))
            if epoch_num > latest_epoch:
                latest_epoch = epoch_num
                latest_ckpt_file = ckpt_file

    # If no numbered checkpoint is found, fallback to 'checkpoint.pth' if it exists.
    if latest_ckpt_file is None:
        if "checkpoint.pth" in all_ckpts:
            latest_ckpt_file = "checkpoint.pth"
            logger.info("No numbered checkpoints found; using 'checkpoint.pth'.")
        else:
            logger.info("No valid checkpoint file found; starting from scratch.")
            return

    # Load the selected checkpoint
    checkpoint_path = os.path.join(weights_dir, latest_ckpt_file)
    logger.info(f"Loading checkpoint from '{checkpoint_path}' ...")
    loc = f"cuda:{args.gpu}" if args.gpu is not None else None
    checkpoint = torch.load(checkpoint_path, map_location=loc)
    args.start_epoch = checkpoint["epoch"]
    try:
        model.load_state_dict(remove_prefix(checkpoint["state_dict"]))
    except:
        model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    logger.info(f"Successfully loaded checkpoint '{checkpoint_path}' (epoch {checkpoint[