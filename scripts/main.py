import argparse
import math
import os
import sys
import time
import yaml
import wandb
import logging
import random
import numpy as np
from logging.config import fileConfig
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as torchvision_models

import dvd_scale_free.utils
import dvd_scale_free.dvd_scale_free.development
import dvd_scale_free.models.loader
import dvd_scale_free.models.eval
from dvd_scale_free.datasets.dataset_loader import SupervisedLearningDataset
from dvd_scale_free.dvd_scale_free.development import DVDTransformer, DVDConfig


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

parser = argparse.ArgumentParser(description="Model Training")
parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument("--dataset-name", default="texture2shape_miniecoset", type=str, help="dataset name")
parser.add_argument("-a", "--arch", metavar="ARCH", default="resnet50", help="model architecture: " + " | ".join(model_names) + " (default: resnet50)")
parser.add_argument("-j", "--workers", default=10, type=int, metavar="N", help="number of data loading workers per node")
parser.add_argument("--epochs", default=300, type=int, metavar="N", help="number of total epochs to run")
parser.add_argument("--start-epoch", default=0, type=int, metavar="N", help="manual epoch number (useful on restarts)")
parser.add_argument("-b", "--batch-size-per-gpu", default=512, type=int, metavar="N", help="mini-batch size per node")

parser.add_argument("--lr", "--learning-rate", default=1e-4, type=float, metavar="LR", dest="lr")
parser.add_argument('--lr-scheduler', type=str, default='', help='Learning rate scheduler to use')
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument("--wd", "--weight-decay", default=1e-6, type=float, metavar="W", dest="weight_decay")
parser.add_argument("-p", "--log-freq", default=5, type=int, metavar="N", help="Log frequency")
parser.add_argument("--resume", default="", type=str, metavar="PATH", help="path to latest checkpoint")
parser.add_argument("--seed", default=0, type=int, help="Random seed.")
parser.add_argument("--dist_url", default="env://", type=str, help="url used to set up distributed training")

parser.add_argument("--optimizer", default="adam", type=str, choices=["lars", "adamw", "adam"])
parser.add_argument("--warmup-epochs", default=0, type=int, metavar="N", help="number of warmup epochs")
parser.add_argument("--save-checkpoint-every-epochs", default=5, type=int, help="Save Frequency")
parser.add_argument("--world-size", default=1, type=int, help="number of nodes for distributed training")
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--image-size', type=int, default=256)

# Setting for development strategy
parser.add_argument('--development_strategy', default='adult', type=str, help='development strategy (adult | dvd)')
parser.add_argument("--time-order", default="chronological", type=str, choices=["chronological", "mid_phase", "random", "fully_random"])
parser.add_argument("--months_per_epoch", default=2, type=float, help="number of months per epoch")
parser.add_argument("--contrast_amplitude_beta", default=1e-4, type=float)
parser.add_argument("--contrast_amplitude_lambda", default=150, type=float)

# Ablations
parser.add_argument('--apply_blur', type=int, default=1, help='Flag to apply blur to images')
parser.add_argument('--apply_color', type=int, default=1, help='Flag to apply color changes')
parser.add_argument('--apply_threshold_color', type=int, default=0, help='Flag to apply threshold color changes')
parser.add_argument('--apply_contrast', type=int, default=1, help='Flag to apply contrast adjustments')

# ============================================================
# NEW: Scale-Free DVD Core Knobs (Ported from V_a)
# ============================================================
parser.add_argument("--contrast_progress_mode", type=str, default="logsf_to_amplitude", choices=["logsf_to_amplitude", "linear_amplitude"])
parser.add_argument("--contrast_progress_remap", type=str, default="log", choices=["none", "power", "log"])
parser.add_argument("--contrast_progress_logspan_start", type=float, default=5e-3)
parser.add_argument("--contrast_to_amplitude_gamma", type=float, default=1.0)

parser.add_argument("--blur_mode", type=str, default="gaussian", choices=["gaussian", "freq"])
parser.add_argument("--contrast_fov_deg", type=float, default=15.0)
parser.add_argument("--use_scale_free_geometry", type=int, default=1)
parser.add_argument("--acuity_use_scale_free_cutoff", type=int, default=1)

parser.add_argument("--contrast_use_barten_csf", type=int, default=1)
parser.add_argument("--contrast_use_csf_for_threshold_readout", type=int, default=0)
parser.add_argument("--contrast_csf_mode", type=str, default="relative_nyquist", choices=["absolute_cpd", "relative_nyquist"])

parser.add_argument("--contrast_use_soft_mask", type=int, default=1)
parser.add_argument("--contrast_soft_tau_log10", type=float, default=0.10)

parser.add_argument("--contrast_band_range_mode", type=str, default="endpoints", choices=["endpoints", "trim"])
parser.add_argument("--contrast_keep_min", type=float, default=1e-4)
parser.add_argument("--contrast_keep_max", type=float, default=1.0)

parser.add_argument("--contrast_sf_min_frac_nyq", type=float, default=None)
parser.add_argument("--contrast_sf_max_frac_nyq", type=float, default=None)
parser.add_argument("--contrast_logsf_min_frac_nyq", type=float, default=None)
parser.add_argument("--contrast_anchor_half_width_frac_nyq", type=float, default=None)

parser.add_argument("--contrast_sf_min_cpd", type=float, default=0.0)
parser.add_argument("--contrast_sf_max_cpd", type=float, default=30.0)
parser.add_argument("--contrast_logsf_min_cpd", type=float, default=0.1)
parser.add_argument("--contrast_anchor_half_width_cpd", type=float, default=0.5)

# additional configs:
parser.add_argument('--pretrained', default='', type=str)
parser.add_argument("--class-weights-json-path", default=None, type=str)
parser.add_argument("--label-smoothing", default=0.0, type=float)
parser.add_argument("--resize_to_224", default=0, type=int)
parser.add_argument("--resize_to_256", default=0, type=int)
parser.add_argument("--grayscale-aug", default=1, type=int)
parser.add_argument("--blur-aug", default=1, type=int)
parser.add_argument("--additional-aug", default=1, type=int)
parser.add_argument("--crop-min", default=1.0, type=float)
parser.add_argument("--best-acc1", default=0.0, type=float)

def setup_logging_and_wandb(args):
    dvd_scale_free.utils.init_distributed_mode(args)
    dvd_scale_free.utils.fix_random_seeds(args.seed)

    cudnn.deterministic = True
    cudnn.benchmark = False

    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "dvd_scale_free", "models", "logging", "config.ini"
    )
    fileConfig(config_path)
    logger = logging.getLogger()
    logger.disabled = True

    # scale-free DVD hyperparameters (key one is just progress_logspan_suffix)
    blur_mode_suffix = f"_blur{args.blur_mode}"
    progress_mode_suffix = "" if args.contrast_progress_mode == "logsf_to_amplitude" else f"_pmode{args.contrast_progress_mode}"
    progress_remap_suffix = "" if args.contrast_progress_remap == "log" else f"_premap{args.contrast_progress_remap}"
    progress_logspan_suffix = "" if float(args.contrast_progress_logspan_start) == 5e-3 else f"_plogstart{args.contrast_progress_logspan_start}"
    
    progress_gamma_suffix = "" if float(args.contrast_to_amplitude_gamma) == 1.0 else f"_pgamma{args.contrast_to_amplitude_gamma}"
    csf_suffix = "" if int(args.contrast_use_barten_csf) == 1 else "_noCSF"
    csf_readout_suffix = "_csfReadout" if int(args.contrast_use_csf_for_threshold_readout) == 1 else ""
    csf_mode_suffix = "" if args.contrast_csf_mode == "relative_nyquist" else f"_csfmode{args.contrast_csf_mode}"
    scale_free_suffix = "" if int(args.use_scale_free_geometry) == 1 else "_absGeom"
    acuity_scale_free_suffix = "" if int(args.acuity_use_scale_free_cutoff) == 1 else "_absAcuity"
    soft_mask_suffix = "" if int(args.contrast_use_soft_mask) == 1 else "_hardMask"
    band_range_suffix = "" if args.contrast_band_range_mode == "endpoints" else f"_band{args.contrast_band_range_mode}"
    tau_suffix = "" if float(args.contrast_soft_tau_log10) == 0.10 else f"_tau{args.contrast_soft_tau_log10}"
    fov_suffix = "" if float(args.contrast_fov_deg) == 15.0 else f"_fov{args.contrast_fov_deg}"
    legacy_alpha_dn_suffix = f"_alpha{args.contrast_amplitude_beta}_dn{args.contrast_amplitude_lambda}"

    if args.development_strategy == "dvd":
        net_name = (
            f"{args.arch}_mpe{args.months_per_epoch}"
            f"{legacy_alpha_dn_suffix}_{args.dataset_name}"
            f"_{args.image_size}_{args.lr_scheduler}{args.lr}_dev_{args.development_strategy}"
            f"_b{args.apply_blur}c{args.apply_color}cs{args.apply_contrast}"
            f"_T_{args.time_order}"
            f"{blur_mode_suffix}{progress_mode_suffix}{progress_remap_suffix}"
            f"{progress_gamma_suffix}{progress_logspan_suffix}{csf_suffix}"
            f"{csf_readout_suffix}{csf_mode_suffix}{scale_free_suffix}"
            f"{acuity_scale_free_suffix}{soft_mask_suffix}{band_range_suffix}"
            f"{tau_suffix}{fov_suffix}_seed_{args.seed}"
        )
    else:
        net_name = (
            f"{args.arch}_{args.dataset_name}_{args.image_size}_"
            f"{args.lr_scheduler}{args.lr}_dev_{args.development_strategy}_seed_{args.seed}"
        )

    wandb_run = None
    log_dir = f"logs/{net_name}" if dvd_scale_free.utils.is_main_process() else None

    if dvd_scale_free.utils.is_main_process():
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

    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    return logger, wandb_run, log_dir, net_name


def stack_collate(batch):
    imgs, labels = zip(*batch)
    labels = [lbl if torch.is_tensor(lbl) else torch.tensor(lbl) for lbl in labels]
    return torch.stack(imgs, dim=0), torch.stack(labels, dim=0)


def get_data_loaders(args):
    dataset_name = args.dataset_name
    dataset = SupervisedLearningDataset(args.data, args)
    dataset = dataset.get_dataset(dataset_name)
    train_dataset, val_dataset, _ = dataset["train"], dataset["val"], dataset["test"]

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)

    is_imagenet = (dataset_name == 'imagenet')
    collate_fn = stack_collate if is_imagenet else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size_per_gpu,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        **({'collate_fn': collate_fn} if collate_fn else {})
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size_per_gpu,
        shuffle=None,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False,
        **({'collate_fn': collate_fn} if collate_fn else {})
    )

    return train_loader, val_loader, train_sampler, val_sampler


# ============================================================
# NEW: DVD Build Helpers
# ============================================================
def build_dvd_transformer(args, age_months_curve: Optional[List[float]] = None) -> Optional[DVDTransformer]:
    if args.development_strategy != "dvd":
        return None

    cfg = DVDConfig(
        apply_blur=int(args.apply_blur),
        apply_color=int(args.apply_color),
        apply_contrast=int(args.apply_contrast),
        apply_threshold_color=bool(int(args.apply_threshold_color)),

        image_size=int(args.image_size),
        resize_input=True,
        fully_random=(args.time_order == "fully_random"),
        age_months_curve=age_months_curve,

        blur_mode=str(args.blur_mode),
        contrast_fov_deg=float(args.contrast_fov_deg),
        use_scale_free_geometry=bool(int(args.use_scale_free_geometry)),
        acuity_use_scale_free_cutoff=bool(int(args.acuity_use_scale_free_cutoff)),

        contrast_progress_mode=str(args.contrast_progress_mode),
        contrast_progress_remap=str(args.contrast_progress_remap),
        contrast_progress_logspan_start=float(args.contrast_progress_logspan_start),
        contrast_to_amplitude_gamma=float(args.contrast_to_amplitude_gamma),

        contrast_use_barten_csf=bool(int(args.contrast_use_barten_csf)),
        contrast_use_csf_for_threshold_readout=bool(int(args.contrast_use_csf_for_threshold_readout)),
        contrast_csf_mode=str(args.contrast_csf_mode),

        contrast_use_soft_mask=bool(int(args.contrast_use_soft_mask)),
        contrast_soft_tau_log10=float(args.contrast_soft_tau_log10),

        contrast_band_range_mode=str(args.contrast_band_range_mode),
        contrast_keep_min=float(args.contrast_keep_min),
        contrast_keep_max=float(args.contrast_keep_max),

        contrast_sf_min_frac_nyq=args.contrast_sf_min_frac_nyq,
        contrast_sf_max_frac_nyq=args.contrast_sf_max_frac_nyq,
        contrast_logsf_min_frac_nyq=args.contrast_logsf_min_frac_nyq,
        contrast_anchor_half_width_frac_nyq=args.contrast_anchor_half_width_frac_nyq,

        contrast_sf_min_cpd=float(args.contrast_sf_min_cpd),
        contrast_sf_max_cpd=float(args.contrast_sf_max_cpd),
        contrast_logsf_min_cpd=float(args.contrast_logsf_min_cpd),
        contrast_anchor_half_width_cpd=float(args.contrast_anchor_half_width_cpd),
    )
    return DVDTransformer(cfg)

def build_age_months_curve(args, iters_per_epoch: int) -> List[float]:
    if args.time_order == "fully_random":
        total_steps = int(args.epochs) * int(iters_per_epoch)
        max_age = float(args.months_per_epoch) * float(args.epochs)
        g = torch.Generator()
        g.manual_seed(int(args.seed))
        ages = torch.rand(total_steps, generator=g).tolist()
        return [float(a) * max_age for a in ages]

    return dvd_scale_free.dvd_scale_free.development.generate_age_months_curve(
        args.epochs,
        iters_per_epoch,
        args.months_per_epoch,
        mid_phase=(args.time_order == "mid_phase"),
        shuffle=(args.time_order == "random"),
        seed=args.seed,
    )


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    scaler,
    wandb_run,
    logger,
    epoch,
    age_months_curve,
    args,
    dvdt=None # NEW: Pass instantiated DVD transformer
):
    batch_time = dvd_scale_free.utils.AverageMeter("Time", ":6.3f")
    data_time = dvd_scale_free.utils.AverageMeter("Data", ":6.3f")
    learning_rates = dvd_scale_free.utils.AverageMeter("LR", ":.4e")
    losses = dvd_scale_free.utils.AverageMeter("Loss", ":.4e")
    top1 = dvd_scale_free.utils.AverageMeter("Acc@1", ":6.2f")
    top5 = dvd_scale_free.utils.AverageMeter("Acc@5", ":6.2f")

    progress = dvd_scale_free.utils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rates, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()

    end = time.time()
    iters_per_epoch = len(train_loader)

    for i, (images, target) in enumerate(train_loader):
        it = len(train_loader) * epoch + i

        data_time.update(time.time() - end)

        if args.lr_scheduler == 'cosine':
            lr = dvd_scale_free.utils.adjust_learning_rate(
                optimizer, epoch + i / iters_per_epoch, args
            )
            learning_rates.update(lr)
        elif args.lr_scheduler == '':
            lr = args.lr
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            learning_rates.update(lr)
        else:
            raise NotImplementedError(
                f"Development strategy {args.lr_scheduler} not implemented"
            )

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # NEW: Optimized applying of the global pre-built transformer
        if dvdt is not None:
            age_months = float(age_months_curve[it])
            images = dvdt(images, months=age_months)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = dvd_scale_free.utils.accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        if dvd_scale_free.utils.is_main_process() and wandb_run is not None and it % args.log_freq == 0:
            wandb.log(
                {
                    "train/loss": loss.item(),
                    "train/top1": acc1[0].item(),
                    "train/top5": acc5[0].item(),
                    "lr": lr,
                    "epoch": epoch,
                },
                step=it,
            )
            metrics = progress.display(i)
            logger.info(metrics)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    print(f"Epoch {epoch} time: {batch_time.sum} seconds")


def main():
    args = parser.parse_args()

    logger, wandb_run, log_dir, net_name = setup_logging_and_wandb(args)
    if dvd_scale_free.utils.is_main_process() and log_dir is not None:
        dvd_scale_free.utils.save_config(args, os.path.join(log_dir, "config.yaml"))

    model, linear_keyword = dvd_scale_free.models.loader.create_model(args, logger)
    dvd_scale_free.models.loader.load_pretrained_weights_if_any(args, model, linear_keyword)

    model, optimizer, scaler = dvd_scale_free.models.loader.build_optimizer_and_scaler(args, model)

    dvd_scale_free.models.loader.resume_checkpoint_if_any(
        args, model, optimizer, scaler, logger, log_dir
    )

    if dvd_scale_free.utils.is_main_process() and log_dir is not None and args.start_epoch == 0:
        dvd_scale_free.utils.save_initial_checkpoint(
            log_dir, args, model, optimizer, scaler, logger, net_name
        )

    if dvd_scale_free.utils.is_main_process() and log_dir is not None:
        stats_file = open(os.path.join(log_dir, "stats.txt"), "a", buffering=1)
        logger.info(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_file)
        with open(os.path.join(log_dir, "metadata.txt"), "a") as f:
            yaml.dump(vars(args), f, allow_unicode=True)
            f.write(str(model))

    train_loader, val_loader, train_sampler, val_sampler = get_data_loaders(args)
    print(f"len loaders : {len(train_loader)}  |  {len(val_loader)} |")

    try:
        criterion = dvd_scale_free.utils.get_loss_function(args)
    except Exception:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda(args.gpu)

    if args.evaluate:
        dvd_scale_free.models.eval.validate(val_loader, model, criterion, args.start_epoch, args.gpu)
        return

    logger.info("Main components ready.")
    logger.info(model)
    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Scaler: {scaler}")
    logger.info("Starting model training.")

    best_acc1 = args.best_acc1

    # NEW: Build the DVD components ONCE before training loops
    age_months_curve = build_age_months_curve(args, iters_per_epoch=len(train_loader))
    dvdt = build_dvd_transformer(args, age_months_curve=age_months_curve)

    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)

        train(
            train_loader,
            model,
            criterion,
            optimizer,
            scaler,
            wandb_run,
            logger,
            epoch,
            age_months_curve,
            args,
            dvdt=dvdt # NEW: Pass transformer instance here
        )

        acc1, _ = dvd_scale_free.models.eval.validate(
            val_loader, model, criterion, epoch, args.gpu, wandb_run, logger
        )
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if dvd_scale_free.utils.is_main_process() and log_dir is not None:
            filename = "checkpoint.pth"
            if (epoch + 1) % args.save_checkpoint_every_epochs == 0:
                filename = f"checkpoint_{epoch}.pth"

            checkpoint_dict = {
                "epoch": epoch,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "best_acc1": best_acc1,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "numpy_rng_state": np.random.get_state(),
                "python_rng_state": random.getstate(),
            }

            dvd_scale_free.utils.save_checkpoint(
                checkpoint_dict,
                is_best=is_best,
                filename=os.path.join(log_dir, "weights", filename),
            )
            dvd_scale_free.utils.save_last_checkpoint(
                checkpoint_dict,
                filename=os.path.join(log_dir, "weights", "checkpoint_last.pth"),
            )

    if dvd_scale_free.utils.is_main_process():
        wandb.finish()


if __name__ == "__main__":
    main()