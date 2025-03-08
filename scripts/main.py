
import argparse
import builtins
import math
import os
import shutil
import sys
import time
from functools import partial

import yaml

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models

import wandb 

import evd.simclr.builder
import evd.simclr.loader
import evd.simclr.optimizer
import evd.utils
import evd.models.vits
import logging
from logging.config import fileConfig

import evd.evd.development
import evd.models.loader
import evd.models.eval
from evd.datasets.dataset_loader import SupervisedLearningDataset

torchvision_model_names = sorted(
    name
    for name in torchvision_models.__dict__
    if name.islower()
    and not name.startswith("__")
    and callable(torchvision_models.__dict__[name])
)

model_names = [
    "vit_small",
    "vit_base",
    "vit_conv_small",
    "vit_conv_base",
] + torchvision_model_names

parser = argparse.ArgumentParser(description="Model Training")
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument(
    "--dataset-name",
    default="texture2shape_miniecoset", 
    type=str,
    help="dataset name",
)
parser.add_argument(
    "-a",
    "--arch",
    metavar="ARCH",
    default="resnet50",
    choices=model_names,
    help="model architecture: " + " | ".join(model_names) + " (default: resnet50)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=10,
    type=int,
    metavar="N",
    help="number of data loading workers per node (default: 10)",
)
parser.add_argument(
    "--epochs", default=300, type=int, metavar="N", help="number of total epochs to run"
)
parser.add_argument(
    "--start-epoch",
    default=1,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size-per-gpu",
    default=512,
    type=int,
    metavar="N",
    help="mini-batch size per node. Official SimCLR uses a global batch size of 4096 images.",
)

parser.add_argument(
    "--lr",
    "--learning-rate",
    default=1e-4,
    type=float,
    metavar="LR",
    help="initial (base) learning rate",
    dest="lr",
)
parser.add_argument('--lr-scheduler', type=str, default='', help='Learning rate scheduler to use (default None)')

parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--wd",
    "--weight-decay",
    default=1e-6,
    type=float,
    metavar="W",
    help="weight decay (default: 1e-6)",
    dest="weight_decay",
)
parser.add_argument(
    "-p",
    "--log-freq",
    default=5,
    type=int,
    metavar="N",
    help="Log frequency (default: 100)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument("--seed", default=0, type=int, help="Random seed.")
parser.add_argument(
    "--dist_url",
    default="env://",
    type=str,
    help="""url used to set up
                    distributed training; see https://pytorch.org/docs/stable/distributed.html""",
)

parser.add_argument(
    "--optimizer",
    default="adam",
    type=str,
    choices=["lars", "adamw", "adam"],
    help="optimizer used (default: lars)",
)
parser.add_argument(
    "--warmup-epochs", default=0, type=int, metavar="N", help="number of warmup epochs (default None)"
)
parser.add_argument(
    "--save-checkpoint-every-epochs",
    default=5,
    type=int,
    help="Save Frequency (default: 5)",
)
# world_size
parser.add_argument(
    "--world-size",
    default=1,
    type=int,
    help="number of nodes for distributed training",
)
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--image-size', type=int, default=256)

# Setting for development strategy
parser.add_argument('--development-strategy',
                    default='adult', 
                    type=str, 
                    help='development strategy (default: evd)',
                    )
parser.add_argument(
    "--time-order",
    default="normal",
    type=str,
    choices=["normal", "mid_phase", "random"],
    help="time order of the batches",
)
parser.add_argument(
    "--months-per-epoch",
    default=1,
    type=float,
    help="number of months per epoch",
)
parser.add_argument(
    "--contrast_threshold",
    default=0.05,
    type=float,
    help="contrast drop speed factor (default: 0.05)",
)
parser.add_argument(
    "--decrease_contrast_threshold_spd",
    default=100,
    type=float,
    help="decrease contrast drop speed (default: 100)",
)

# Ablations
parser.add_argument('--apply_blur', type=int, default=1, help='Flag to apply blur to images')
parser.add_argument('--apply_color', type=int, default=1, help='Flag to apply color changes')
parser.add_argument('--apply_contrast', type=int, default=1, help='Flag to apply contrast adjustments')

# additional configs:
parser.add_argument('--pretrained', default='', type=str,
                    help='path to start from pretrained checkpoint')

# class_weights_json_path
parser.add_argument(
    "--class-weights-json-path",
    default=None,
    type=str,
    help="path to class weights json file",
)

parser.add_argument(
   "--label-smoothing",
    default=0.0,
    type=float,
    help="Label smoothing to apply in cross-entropy loss (default: 0.0)",
)

def setup_logging_and_wandb(args):
    """
    Sets up logging and initializes WandB run if main process.
    Returns logger, wandb_run (or None), and log_dir.
    """
    # Initialize distributed training
    evd.utils.init_distributed_mode(args)
    evd.utils.fix_random_seeds(args.seed)
    cudnn.benchmark = True

    # Setup logging
    fileConfig("evd/models/logging/config.ini")
    logger = logging.getLogger()
    logger.disabled = True  # Will enable if main process

    net_name = (
        f'{args.arch}_mpe{args.months_per_epoch}_alpha{args.contrast_threshold}'
        f'_dn{args.decrease_contrast_threshold_spd}_{args.dataset_name}'
        f'{args.image_size}_{args.lr_scheduler}{args.lr}_dev_{args.development_strategy}'
        f'_b{args.apply_blur}c{args.apply_color}cs{args.apply_contrast}'
        f'_T_{args.time_order}_seed_{args.seed}'
    )
    wandb_run = None

    if evd.utils.is_main_process():
        log_dir = f'logs/{net_name}'
        os.makedirs(log_dir, exist_ok=True)

        wandb.init(project="final_early_visual_development", name=net_name, config=vars(args), dir=log_dir)
        wandb_run = wandb.run

        FileOutputHandler = logging.FileHandler(
            os.path.join(log_dir, f"train_gpu={args.gpu}.log")
        )
        logger.disabled = False
        logger.addHandler(FileOutputHandler)
    else:
        log_dir = None

    logger.info(
        "\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items()))
    )

    return logger, wandb_run, log_dir, net_name

def get_data_loaders(args):
    """
    Retrieves the training and validation datasets and wraps them into DataLoaders 
    with distributed sampling.
    """
    dataset_name = args.dataset_name
    dataset = SupervisedLearningDataset(args.data) # Load dataset from path (args.data)
    dataset = dataset.get_dataset(dataset_name)
    train_dataset, val_dataset, _ = dataset["train"], dataset["val"], dataset["test"]

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    print(f"train_sampler: {train_sampler}")
    print(f"val_sampler: {val_sampler}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size_per_gpu,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size_per_gpu,
        shuffle=None,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=True,
    )

    return train_loader, val_loader, train_sampler, val_sampler


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    scaler,
    wandb_run,
    logger,
    epoch,
    args,
):
    batch_time = evd.utils.AverageMeter("Time", ":6.3f")
    data_time = evd.utils.AverageMeter("Data", ":6.3f")
    learning_rates = evd.utils.AverageMeter("LR", ":.4e")
    losses = evd.utils.AverageMeter("Loss", ":.4e")
    top1 = evd.utils.AverageMeter("Acc@1", ":6.2f")
    top5 = evd.utils.AverageMeter("Acc@5", ":6.2f")

    # Progress meter display settings
    progress = evd.utils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rates, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    iters_per_epoch = len(train_loader)

    # Generate age months curve to map batches to age months for EVD
    age_months_curve = evd.evd.development.generate_age_months_curve(
        args.epochs,
        len(train_loader),
        args.months_per_epoch,
        mid_phase=(args.time_order == "mid_phase"),
        shuffle=(args.time_order == "random"),
        seed=args.seed,
    )

    for i, (images, target) in enumerate(train_loader):
        # global step
        it = len(train_loader) * epoch + i

        # measure data loading time
        data_time.update(time.time() - end)

        # adjust learning rate
        if args.lr_scheduler == 'cosine':
            lr = evd.utils.adjust_learning_rate(
                optimizer, epoch + i / iters_per_epoch, args
            )
            learning_rates.update(lr)
        elif args.lr_scheduler == '':
            lr = args.lr
        else:
            raise NotImplementedError(
                f"Development strategy {args.lr_scheduler} not implemented"
            )

        # Get age in months (for EVD transformations) | epoch start from 1 so -1
        age_months = age_months_curve[(epoch - 1) * len(train_loader) + i]

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # Experience across visual development
        if args.development_strategy == "evd":
            contrast_control_coeff = max(math.floor(age_months / args.decrease_contrast_threshold_spd) * 2, 1)
            images = evd.evd.development.EarlyVisualDevelopmentTransformer().apply_fft_transformations(
                images,
                age_months,
                apply_blur=args.apply_blur, 
                apply_color=args.apply_color, 
                apply_contrast=args.apply_contrast,
                contrast_threshold=args.contrast_threshold / contrast_control_coeff,
                image_size=args.image_size,
                verbose=False,
            )
        elif args.development_strategy == "adult":
            pass
        else:
            raise NotImplementedError(
                f"Development strategy {args.development_strategy} not implemented"
            )

        # compute output
        with torch.cuda.amp.autocast(True):
            output = model(images)
            loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = evd.utils.accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # Log with wandb if main process
        if evd.utils.is_main_process() and wandb_run is not None and it % args.log_freq == 0:
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/top1": acc1[0].item(),
                    "train/top5": acc5[0].item(),
                    "train/loss": loss.item(),
                    "lr": lr,
                    "epoch": epoch,
                },
                step=it,
            )
            metrics = progress.display(i)
            logger.info(metrics)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    print(f"Epoch {epoch} time: {batch_time.sum} seconds")

def main():
    args = parser.parse_args()

    # 1) Setup logging, distributed, and wandb
    logger, wandb_run, log_dir, net_name = setup_logging_and_wandb(args)
    if evd.utils.is_main_process() and log_dir is not None:
        evd.utils.save_config(args, os.path.join(log_dir, f"config.yaml")) # Saving config setup

    # 2) Create and setup model
    model, linear_keyword = evd.models.loader.create_model(args, logger)
    evd.models.loader.load_pretrained_weights_if_any(args, model, linear_keyword)
    # Optionally compile the model (PyTorch 2.0+) to speed up training
    if hasattr(torch, "compile"):
        model = torch.compile(model)

    # 3) Build optimizer and FP16 scaler
    model, optimizer, scaler = evd.models.loader.build_optimizer_and_scaler(args, model)
    if evd.utils.is_main_process() and log_dir is not None:
        evd.utils.save_initial_checkpoint(log_dir, args, model, optimizer, scaler, logger, net_name) 
            
    # 4) Possibly resume checkpoint
    if evd.utils.is_main_process() and log_dir is not None:
        evd.models.loader.resume_checkpoint_if_any(args, model, optimizer, scaler, logger, log_dir)

    # Prepare to log stats if main process
    if evd.utils.is_main_process() and log_dir is not None:
        stats_file = open(os.path.join(log_dir, "stats.txt"), "a", buffering=1)
        logger.info(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_file)
        with open(os.path.join(log_dir, "metadata.txt"), "a") as f:
            yaml.dump(args, f, allow_unicode=True)
            f.write(str(model))
        
    # 5) Get data loaders
    train_loader, val_loader, train_sampler, val_sampler = get_data_loaders(args)

    # 6) Optionally only evaluate
    if args.evaluate:
        evd.models.eval.validate(val_loader, model, criterion, epoch, args.gpu)
        return

    logger.info("Main components ready.")
    logger.info(model)
    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Scaler: {scaler}")
    logger.info("Starting model training.")

    best_acc1 = 0
    try:
        criterion = evd.utils.get_loss_function(args)
    except:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda(args.gpu)

    for epoch in range(args.start_epoch, args.epochs +1):
        train_sampler.set_epoch(epoch)

        # train for one epoch
        train(
            train_loader,
            model,
            criterion,
            optimizer,
            scaler,
            wandb_run,
            logger,
            epoch,
            args,
        )

        # evaluate on validation set
        acc1, _ = evd.models.eval.validate(val_loader, model, criterion, epoch, args.gpu, wandb_run, logger)
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if evd.utils.is_main_process() and log_dir is not None:
            filename = "checkpoint.pth"
            if (epoch + 1) % args.save_checkpoint_every_epochs == 0:
                filename = f"checkpoint_{epoch}.pth"

            checkpoint_dict = {
                    "epoch": epoch,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                }
            evd.utils.save_checkpoint(
                checkpoint_dict,
                is_best=is_best,
                filename=os.path.join(log_dir, 'weights', filename),
            )
            evd.utils.save_last_checkpoint(
                checkpoint_dict,
                filename=os.path.join(log_dir, 'weights', "checkpoint_last.pth"),
            )

    if evd.utils.is_main_process():
        wandb.finish()  


if __name__ == "__main__":
    main()

