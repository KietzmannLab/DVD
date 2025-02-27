import os
import re
import torch
import torch.nn as nn
import torchvision.models as torchvision_models
import evd.models.vits
import evd.simclr.optimizer



def create_model(args, logger=None):
    """
    Creates the model (either ViT or torchvision model) and adjusts the final layer 
    based on dataset_name. 
    """
    if logger:
        logger.info("Creating model '{}'".format(args.arch))
        
    if args.arch.startswith("vit"):
        model = evd.models.vits.__dict__[args.arch]()
        linear_keyword = "head"
    else:
        model = torchvision_models.__dict__[args.arch]()
        linear_keyword = "fc"

    if args.dataset_name == "texture2shape_miniecoset":
        out_dim = 112
        # change the last layer
        model.fc = nn.Linear(model.fc.in_features, out_dim)
    elif args.dataset_name == "ecoset_square256":
        out_dim = 565
        model.fc = nn.Linear(model.fc.in_features, out_dim)
    elif args.dataset_name == "imagenet":
        pass
    else:
        raise ValueError(f"dataset_name: {args.dataset_name} not supported")

    return model, linear_keyword


def load_pretrained_weights_if_any(args, model, linear_keyword):
    """
    Loads pretrained weights into model if args.pretrained is specified.
    """
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            state_dict = checkpoint["state_dict"]
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
        optimizer = evd.simclr.optimizer.LARS(
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
    
    # Case 2: No --resume given, or file not found => search automatically
    resume_latest_checkpoint(args, model, optimizer, scaler, logger, log_dir)


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
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        args.start_epoch = checkpoint["epoch"]
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
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scaler.load_state_dict(checkpoint["scaler"])
    args.start_epoch = checkpoint["epoch"]
    logger.info(f"Successfully loaded checkpoint '{checkpoint_path}' (epoch {checkpoint['epoch']}).")