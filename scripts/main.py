import argparse
import os
import sys
import time
import yaml
import wandb
import logging
import random
from logging.config import fileConfig

import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as torchvision_models

import dvd.utils
import dvd.dvd.development
import dvd.models.loader
import dvd.models.eval

from dvd.datasets.dataset_loader import SupervisedLearningDataset
from dvd.dvd.development import DVDTransformer, DVDConfig


torchvision_model_names = sorted(
    name
    for name in torchvision_models.__dict__
    if name.islower()
    and not name.startswith("__")
    and callable(torchvision_models.__dict__[name])
)

model_names = [
    "customCNN",
] + torchvision_model_names


# ============================================================
# Args
# ============================================================
parser = argparse.ArgumentParser(description="Model Training")

parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument("--dataset-name", default="texture2shape_miniecoset", type=str, help="dataset name")
parser.add_argument(
    "-a", "--arch",
    metavar="ARCH",
    default="resnet50",
    help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
)
parser.add_argument(
    "-j", "--workers",
    default=10,
    type=int,
    metavar="N",
    help="number of data loading workers per node (default: 10)",
)
parser.add_argument("--epochs", default=300, type=int, metavar="N", help="number of total epochs to run")
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b", "--batch-size-per-gpu",
    default=512,
    type=int,
    metavar="N",
    help="mini-batch size per gpu",
)

parser.add_argument(
    "--lr", "--learning-rate",
    default=1e-4,
    type=float,
    metavar="LR",
    help="base learning rate",
    dest="lr",
)
parser.add_argument("--lr-scheduler", type=str, default="", help="Learning rate scheduler to use")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd", "--weight-decay",
    default=1e-6,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-6)",
    dest="weight_decay",
)
parser.add_argument("-p", "--log-freq", default=5, type=int, metavar="N", help="log frequency")
parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to checkpoint")
parser.add_argument("--seed", default=0, type=int, help="random seed")
parser.add_argument(
    "--dist_url",
    default="env://",
    type=str,
    help="url used to set up distributed training",
)

parser.add_argument(
    "--optimizer",
    default="adam",
    type=str,
    choices=["lars", "adamw", "adam"],
    help="optimizer used",
)
parser.add_argument("--warmup-epochs", default=0, type=int, metavar="N", help="number of warmup epochs")
parser.add_argument(
    "--save-checkpoint-every-epochs",
    default=5,
    type=int,
    help="save checkpoint every N epochs",
)
parser.add_argument("--world-size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument("-e", "--evaluate", dest="evaluate", action="store_true", help="evaluate on validation set")
parser.add_argument("--image-size", type=int, default=256)

# Development strategy
parser.add_argument(
    "--development_strategy",
    default="adult",
    type=str,
    help="development strategy (adult or dvd)",
)
parser.add_argument(
    "--time-order",
    default="chronological",
    type=str,
    choices=["chronological", "mid_phase", "random", "fully_random"],
    help="time order of the batches",
)
parser.add_argument("--months_per_epoch", default=2, type=float, help="number of months per epoch")
parser.add_argument(
    "--contrast_amplitude_beta",
    default=1e-4,
    type=float,
    help="base amplitude threshold in frequency domain",
)
parser.add_argument(
    "--contrast_amplitude_lambda",
    default=150,
    type=float,
    help="decrease contrast drop speed",
)

# Ablations
parser.add_argument("--apply_blur", type=int, default=1, help="flag to apply blur")
parser.add_argument("--apply_color", type=int, default=1, help="flag to apply color")
parser.add_argument("--apply_threshold_color", type=int, default=0, help="flag to apply threshold color")
parser.add_argument("--apply_contrast", type=int, default=1, help="flag to apply contrast")

# Additional configs
parser.add_argument("--pretrained", default="", type=str, help="path to pretrained checkpoint")
parser.add_argument("--class-weights-json-path", default=None, type=str, help="path to class weights json file")
parser.add_argument("--label-smoothing", default=0.0, type=float, help="label smoothing")
parser.add_argument("--resize_to_224", default=0, type=int, help="resize images to 224x224")
parser.add_argument("--resize_to_256", default=0, type=int, help="resize images to 256x256")
parser.add_argument("--grayscale-aug", default=1, type=int, help="apply grayscale augmentation")
parser.add_argument("--blur-aug", default=1, type=int, help="apply blur augmentation")
parser.add_argument("--additional-aug", default=1, type=int, help="apply additional augmentation")
parser.add_argument("--crop-min", default=1.0, type=float, help="minimum scale for random cropping")
parser.add_argument("--best-acc1", default=0.0, type=float, help="best accuracy achieved so far")


# ============================================================
# Setup
# ============================================================
def setup_logging_and_wandb(args):
    """
    Setup distributed mode, seeds, logging, and WandB.
    Returns:
        logger, wandb_run, log_dir, net_name
    """
    dvd.utils.init_distributed_mode(args)
    dvd.utils.fix_random_seeds(args.seed)

    cudnn.deterministic = True
    cudnn.benchmark = False

    fileConfig("dvd/models/logging/config.ini")
    logger = logging.getLogger()
    logger.disabled = True

    if args.development_strategy == "dvd":
        net_name = (
            f"{args.arch}_mpe{args.months_per_epoch}_alpha{args.contrast_amplitude_beta}"
            f"_dn{args.contrast_amplitude_lambda}_{args.dataset_name}"
            f"{args.image_size}_{args.lr_scheduler}{args.lr}_dev_{args.development_strategy}"
            f"_b{args.apply_blur}c{args.apply_color}cs{args.apply_contrast}"
            f"_T_{args.time_order}_seed_{args.seed}"
        )
    else:
        net_name = (
            f"{args.arch}_{args.dataset_name}_{args.image_size}_"
            f"{args.lr_scheduler}{args.lr}_dev_{args.development_strategy}_seed_{args.seed}"
        )

    wandb_run = None

    if dvd.utils.is_main_process():
        log_dir = f"logs/{net_name}"
        os.makedirs(log_dir, exist_ok=True)

        wandb.init(
            project="final_early_visual_development",
            name=net_name,
            config=vars(args),
            dir=log_dir,
        )
        wandb_run = wandb.run

        file_handler = logging.FileHandler(os.path.join(log_dir, f"train_gpu={args.gpu}.log"))
        logger.disabled = False
        logger.addHandler(file_handler)
    else:
        log_dir = None

    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    return logger, wandb_run, log_dir, net_name


# ============================================================
# Data
# ============================================================
def stack_collate(batch):
    imgs, labels = zip(*batch)
    labels = [lbl if torch.is_tensor(lbl) else torch.tensor(lbl) for lbl in labels]
    return torch.stack(imgs, dim=0), torch.stack(labels, dim=0)


def get_data_loaders(args):
    """
    Build train / val dataloaders with DistributedSampler.
    """
    dataset = SupervisedLearningDataset(args.data, args).get_dataset(args.dataset_name)
    train_dataset, val_dataset, _ = dataset["train"], dataset["val"], dataset["test"]

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)

    is_imagenet = args.dataset_name == "imagenet"
    collate_fn = stack_collate if is_imagenet else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size_per_gpu,
        shuffle=False,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=(args.workers > 0),
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size_per_gpu,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
        persistent_workers=(args.workers > 0),
    )

    return train_loader, val_loader, train_sampler, val_sampler


# ============================================================
# Curriculum
# ============================================================
def build_age_months_curve(args, train_loader, checkpoint, logger=None):
    """
    Build or restore age_months_curve.
    Priority:
        1. resume checkpoint's saved curve
        2. regenerate from current args
    """
    if checkpoint is not None and checkpoint.get("age_months_curve", None) is not None:
        age_months_curve = checkpoint["age_months_curve"]
        if logger is not None:
            logger.info("Using age_months_curve restored from checkpoint.")
        return age_months_curve

    age_months_curve = dvd.dvd.development.generate_age_months_curve(
        args.epochs,
        len(train_loader),
        args.months_per_epoch,
        mid_phase=(args.time_order == "mid_phase"),
        shuffle=(args.time_order == "random"),
        seed=args.seed,
    )

    if logger is not None:
        logger.info("Generated fresh age_months_curve from current arguments.")

    return age_months_curve


def get_age_month_for_step(age_months_curve, epoch, step_in_epoch, steps_per_epoch):
    """
    Fetch curriculum age for one training step.
    """
    global_step = epoch * steps_per_epoch + step_in_epoch
    if global_step < 0 or global_step >= len(age_months_curve):
        raise IndexError(
            f"global_step={global_step} out of range for age_months_curve "
            f"with length={len(age_months_curve)}"
        )
    return age_months_curve[global_step]


# ============================================================
# LR
# ============================================================
def get_current_lr(optimizer):
    return float(optimizer.param_groups[0]["lr"])


def set_constant_lr_if_needed(optimizer, effective_lr):
    for pg in optimizer.param_groups:
        pg["lr"] = effective_lr


# ============================================================
# Train / validate
# ============================================================
def train_one_epoch(
    train_loader,
    model,
    criterion,
    optimizer,
    scaler,
    wandb_run,
    logger,
    epoch,
    age_months_curve,
    effective_lr,
    args,
):
    batch_time = dvd.utils.AverageMeter("Time", ":6.3f")
    data_time = dvd.utils.AverageMeter("Data", ":6.3f")
    learning_rates = dvd.utils.AverageMeter("LR", ":.4e")
    losses = dvd.utils.AverageMeter("Loss", ":.4e")
    top1 = dvd.utils.AverageMeter("Acc@1", ":6.2f")
    top5 = dvd.utils.AverageMeter("Acc@5", ":6.2f")

    progress = dvd.utils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rates, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]",
    )

    model.train()

    end = time.time()
    iters_per_epoch = len(train_loader)

    # For efficiency, instantiate DVDTransformer once per epoch if needed
    dvdt = None
    if args.development_strategy == "dvd":
        dvdt = DVDTransformer(
            DVDConfig(
                apply_blur=args.apply_blur,
                apply_color=args.apply_color,
                apply_contrast=args.apply_contrast,
                contrast_amplitude_beta=args.contrast_amplitude_beta,
                contrast_amplitude_lam=args.contrast_amplitude_lambda,
                apply_threshold_color=args.apply_threshold_color,
                image_size=args.image_size,
                fully_random=(args.time_order == "fully_random"),
                age_months_curve=age_months_curve,
            )
        )

    for i, (images, target) in enumerate(train_loader):
        global_step = epoch * iters_per_epoch + i
        data_time.update(time.time() - end)

        # LR schedule
        if args.lr_scheduler == "cosine":
            lr = dvd.utils.adjust_learning_rate(
                optimizer,
                epoch + i / iters_per_epoch,
                args,
            )
        elif args.lr_scheduler == "":
            set_constant_lr_if_needed(optimizer, effective_lr)
            lr = effective_lr
        else:
            raise NotImplementedError(f"Learning rate scheduler '{args.lr_scheduler}' not implemented.")

        learning_rates.update(lr)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        if args.development_strategy == "dvd":
            age_months = get_age_month_for_step(age_months_curve, epoch, i, iters_per_epoch)
            images = dvdt(images, age_months)
        elif args.development_strategy == "adult":
            pass
        else:
            raise NotImplementedError(
                f"Development strategy '{args.development_strategy}' not implemented."
            )

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = dvd.utils.accuracy(output, target, topk=(1, 5))

        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if dvd.utils.is_main_process() and wandb_run is not None and global_step % args.log_freq == 0:
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/top1": acc1[0].item(),
                    "train/top5": acc5[0].item(),
                    "lr": get_current_lr(optimizer),
                    "epoch": epoch,
                },
                step=global_step,
            )
            logger.info(progress.display(i))

        batch_time.update(time.time() - end)
        end = time.time()

    print(f"Epoch {epoch} time: {batch_time.sum} seconds")


# ============================================================
# Checkpoint payload
# ============================================================
def build_checkpoint_dict(
    args,
    epoch,
    model,
    optimizer,
    scaler,
    best_acc1,
    age_months_curve,
):
    """
    Build checkpoint payload for saving.
    """
    checkpoint_dict = {
        "epoch": epoch,
        "arch": args.arch,
        "state_dict": dvd.models.loader.get_model_state_for_saving(model),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "best_acc1": best_acc1,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
        "age_months_curve": age_months_curve,
        "args": dict(vars(args)),
    }
    return checkpoint_dict


def save_training_checkpoint(args, log_dir, checkpoint_dict, is_best, epoch):
    """
    Save rolling checkpoint, periodic checkpoint, and last checkpoint.
    """
    filename = "checkpoint.pth"
    if (epoch + 1) % args.save_checkpoint_every_epochs == 0:
        filename = f"checkpoint_{epoch}.pth"

    dvd.utils.save_checkpoint(
        checkpoint_dict,
        is_best=is_best,
        filename=os.path.join(log_dir, "weights", filename),
    )
    dvd.utils.save_last_checkpoint(
        checkpoint_dict,
        filename=os.path.join(log_dir, "weights", "checkpoint_last.pth"),
    )


# ============================================================
# Main
# ============================================================
def main():
    args = parser.parse_args()

    logger, wandb_run, log_dir, net_name = setup_logging_and_wandb(args)

    if dvd.utils.is_main_process() and log_dir is not None:
        dvd.utils.save_config(args, os.path.join(log_dir, "config.yaml"))

    model, linear_keyword = dvd.models.loader.create_model(args, logger)
    dvd.models.loader.load_pretrained_weights_if_any(args, model, linear_keyword)

    model, optimizer, scaler, effective_lr = dvd.models.loader.build_optimizer_and_scaler(args, model)

    checkpoint = dvd.models.loader.resume_checkpoint_if_any(
        args, model, optimizer, scaler, logger, log_dir
    )

    if dvd.utils.is_main_process() and log_dir is not None and args.start_epoch == 0:
        dvd.utils.save_initial_checkpoint(
            log_dir, args, model, optimizer, scaler, logger, net_name
        )

    if dvd.utils.is_main_process() and log_dir is not None:
        stats_file = open(os.path.join(log_dir, "stats.txt"), "a", buffering=1)
        logger.info(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_file)

        with open(os.path.join(log_dir, "metadata.txt"), "a") as f:
            yaml.dump(vars(args), f, allow_unicode=True)
            f.write("\n")
            f.write(str(model))

    train_loader, val_loader, train_sampler, val_sampler = get_data_loaders(args)
    logger.info(f"len(train_loader)={len(train_loader)} | len(val_loader)={len(val_loader)}")

    try:
        criterion = dvd.utils.get_loss_function(args)
    except Exception:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda(args.gpu)

    if args.evaluate:
        dvd.models.eval.validate(val_loader, model, criterion, args.start_epoch, args.gpu)
        return

    age_months_curve = build_age_months_curve(args, train_loader, checkpoint, logger)

    logger.info("Main components ready.")
    logger.info(model)
    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Scaler: {scaler}")
    logger.info(f"Effective LR: {effective_lr}")
    logger.info(f"Start epoch: {args.start_epoch}")
    logger.info(f"Best acc1: {args.best_acc1}")

    if args.start_epoch < args.epochs:
        first_resumed_global_step = args.start_epoch * len(train_loader)
        if first_resumed_global_step < len(age_months_curve):
            logger.info(
                f"First resumed age_months = {age_months_curve[first_resumed_global_step]}"
            )

    logger.info("Starting model training.")

    best_acc1 = float(args.best_acc1)

    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        train_one_epoch(
            train_loader=train_loader,
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            wandb_run=wandb_run,
            logger=logger,
            epoch=epoch,
            age_months_curve=age_months_curve,
            effective_lr=effective_lr,
            args=args,
        )

        acc1, _ = dvd.models.eval.validate(
            val_loader,
            model,
            criterion,
            epoch,
            args.gpu,
            wandb_run,
            logger,
        )

        is_best = acc1 > best_acc1
        best_acc1 = max(float(acc1), best_acc1)

        if dvd.utils.is_main_process() and log_dir is not None:
            checkpoint_dict = build_checkpoint_dict(
                args=args,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                best_acc1=best_acc1,
                age_months_curve=age_months_curve,
            )
            save_training_checkpoint(
                args=args,
                log_dir=log_dir,
                checkpoint_dict=checkpoint_dict,
                is_best=is_best,
                epoch=epoch,
            )

    if dvd.utils.is_main_process():
        wandb.finish()


if __name__ == "__main__":
    main()