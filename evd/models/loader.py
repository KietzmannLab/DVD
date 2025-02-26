import os
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
    args.lr = args.lr * args.batch_size_per_gpu / 256

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
    else:
        raise ValueError(f"Unknown optimizer {args.optimizer}")

    scaler = torch.cuda.amp.GradScaler()
    return model, optimizer, scaler


def resume_checkpoint_if_any(args, model, optimizer, scaler, logger):
    """
    Resumes from checkpoint if args.resume is provided.
    """
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