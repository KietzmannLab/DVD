import argparse
import hashlib
import json
import os
import sys
import time
import logging
import random
from dataclasses import asdict
from logging.config import fileConfig

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from dvd_scale_free import DVDConfig, DVDTrainingTransform, InverseConfig, InverseDVD
from dvd_scale_free.dvd_scale_free._cli import adult_anchor, infant_anchor
from dvd_scale_free.dvd_scale_free.development import SPEED_COMPONENTS
from dvd_scale_free.dvd_scale_free.helpers.rrc import rrc_from_args
from dvd_scale_free.dvd_scale_free.helpers.schedule import generate_age_months_curve
from dvd_scale_free.dvd_scale_free.inverse import REFERENCE_BAND_CPD

recipe_option = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
recipe_option.add_argument("--training-recipe", choices=("general", "p004-paper"), default="general",
                           help="opt in to the P004 ImageNet/RN50 paper runner")
parser = argparse.ArgumentParser(description="Model Training", parents=[recipe_option])
parser.add_argument("data", metavar="DIR", nargs="?", help="path to dataset")
parser.add_argument("--dataset-name", default="texture2shape_miniecoset", type=str, help="dataset name")
parser.add_argument("--imagenet-path", help="ImageNet ImageFolder root containing train/ and val/")
parser.add_argument("-a", "--arch", metavar="ARCH", default="resnet50",
                    help="torchvision or timm architecture (default: resnet50)")
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
parser.add_argument("--development_strategy", "--development-strategy", default="adult",
                    choices=["adult", "dvd"], help="use adult inputs or DVD during training")
parser.add_argument("--describe-dvd", action="store_true",
                    help="print resolved DVD configuration as JSON and exit without loading data or GPUs")
parser.add_argument("--time_order", default="chronological", type=str, choices=["chronological", "mid_phase", "random", "fully_random"])
parser.add_argument("--months_per_epoch", default=2, type=float, help="number of months per epoch")

parser.add_argument("--apply_blur", type=int, choices=[0, 1], default=1, help="apply age-dependent acuity")
parser.add_argument("--apply_color", type=int, choices=[0, 1],
                    help="age-dependent color (default: on for forward, off for flipped modes)")
parser.add_argument("--apply_contrast", type=int, choices=[0, 1], default=1,
                    help="apply age-dependent contrast when DVD is selected (default: on)")

dvd_options = parser.add_argument_group("DVD options")
dvd_options.add_argument("--dvd-fov-deg", "--contrast_fov_deg", type=float, default=15.0)
dvd_options.add_argument("--dvd-sigmoid-width-log10", "--contrast_soft_tau_log10", type=float, default=0.10)
dvd_options.add_argument("--dvd-speed", type=float, default=1.0,
                         help="developmental speed V: >1 advances selected curves, <1 delays them (default: 1)")
dvd_options.add_argument("--dvd-speed-components", nargs="*", choices=SPEED_COMPONENTS, default=SPEED_COMPONENTS,
                         help="curves affected by V (default: all three; omit values to leave every curve at V=1)")
dvd_options.add_argument("--dvd-infant-cpd", type=infant_anchor, default=0.23,
                         help="infant anchor, as cpd or 'fundamental' (default: 0.23)")
dvd_options.add_argument("--dvd-adult-cpd", type=adult_anchor, default=None,
                         help="adult anchor, as cpd, 'nyquist', or 'auto' (default: auto)")
dvd_options.add_argument("--dvd-vertical-fov-deg", type=float, help="default: infer from image aspect ratio")
dvd_options.add_argument("--dvd-backend", choices=["auto", "reference", "batched"], default="auto")
dvd_options.add_argument("--dvd-chunk-size", type=int, default=128, help="images per batched FFT chunk")

inverse_options = parser.add_argument_group("Inverse DVD options")
inverse_options.add_argument("--invert-frequency", action="store_true", help="release fine spatial frequencies first")
inverse_options.add_argument("--invert-amplitude", action="store_true", help="release weak amplitudes first")
inverse_options.add_argument("--inverse-birth-cpd", type=float, default=2.)
inverse_options.add_argument("--inverse-quantile", type=float, default=.5)
inverse_options.add_argument("--inverse-taper", type=float, default=.2)
inverse_options.add_argument("--inverse-native-band", action="store_true",
                             help="calibrate to the current Nyquist band instead of the 256 px reference band")
inverse_options.add_argument("--inverse-chunk-size", type=int, default=32)

# additional configs:
parser.add_argument('--pretrained', default='', type=str)
parser.add_argument("--class-weights-json-path", default=None, type=str)
parser.add_argument("--label-smoothing", default=0.0, type=float)
parser.add_argument("--resize_to_224", default=0, type=int)
parser.add_argument("--resize_to_256", default=0, type=int)
parser.add_argument("--grayscale-aug", type=int, choices=[0, 1],
                    help="random grayscale (default: off for flipped modes, on otherwise)")
parser.add_argument("--blur-aug", default=1, type=int)
parser.add_argument("--additional-aug", default=1, type=int)
parser.add_argument("--rrc", action=argparse.BooleanOptionalAction, default=False,
                    help="training-only random resized crop (default: off)")
parser.add_argument("--rrc-scale", type=float, nargs=2, metavar=("MIN", "MAX"), help="crop area fraction (default: .08 1)")
parser.add_argument("--rrc-ratio", type=float, nargs=2, metavar=("MIN", "MAX"), help="crop aspect ratio (default: .75 1.3333)")
parser.add_argument("--crop-min", default=1.0, type=float, help="legacy alias for RRC minimum area; requires --rrc")
parser.add_argument("--best-acc1", default=0.0, type=float)
parser.add_argument("--output-dir", help="explicit run directory for --training-recipe p004-paper")


def parse_args(argv=None):
    chosen, _ = recipe_option.parse_known_args(argv)
    if chosen.training_recipe == "p004-paper":
        from dvd_scale_free.models.paper_runner import RECIPE_DEFAULTS

        return parser.parse_args(argv, namespace=argparse.Namespace(**RECIPE_DEFAULTS))
    return parser.parse_args(argv)

def setup_logging_and_wandb(args):
    import wandb
    import dvd_scale_free.utils

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

    if args.dvd_protocol["method"] == "dvd":
        protocol = json.dumps(args.dvd_protocol, sort_keys=True).encode()
        tag = hashlib.sha256(protocol).hexdigest()[:12]
        net_name = (
            f"{args.arch}_{args.dataset_name}_{args.image_size}_{args.lr_scheduler}{args.lr}"
            f"_dev_dvd_{tag}_seed_{args.seed}"
        )
    else:
        net_name = (
            f"{args.arch}_{args.dataset_name}_{args.image_size}_"
            f"{args.lr_scheduler}{args.lr}_dev_{args.development_strategy}_seed_{args.seed}"
        )

    if args.dvd_protocol.get("rrc") and args.dvd_protocol["method"] != "dvd":
        tag = hashlib.sha256(json.dumps(args.dvd_protocol["rrc"], sort_keys=True).encode()).hexdigest()[:10]
        net_name += f"_rrc_{tag}"

    wandb_run = None
    log_dir = f"logs/{net_name}"

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
    from dvd_scale_free.datasets.dataset_loader import SupervisedLearningDataset

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


def build_dvd_transformer(args):
    flipped = args.invert_frequency or args.invert_amplitude
    if args.development_strategy != "dvd":
        if flipped:
            raise ValueError("Spectrum flips require --development-strategy dvd")
        return None

    color = (not flipped) if args.apply_color is None else bool(args.apply_color)
    if flipped:
        if args.dvd_backend == "reference":
            raise ValueError("Spectrum flips use the batched reflected filter, not the forward reference backend")
        return InverseDVD(InverseConfig(
            invert_frequency=args.invert_frequency,
            invert_amplitude=args.invert_amplitude,
            speed=args.dvd_speed, speed_components=args.dvd_speed_components, fov_deg=args.dvd_fov_deg,
            vertical_fov_deg=args.dvd_vertical_fov_deg,
            birth_frequency_cpd=args.inverse_birth_cpd, amplitude_quantile=args.inverse_quantile,
            taper=args.inverse_taper, chunk_size=args.inverse_chunk_size,
            band_limit_cpd=None if args.inverse_native_band else REFERENCE_BAND_CPD,
            apply_frequency=bool(args.apply_blur), apply_amplitude=bool(args.apply_contrast),
            apply_color=color,
        ))
    return DVDTrainingTransform(
        DVDConfig(
            speed=args.dvd_speed,
            speed_components=args.dvd_speed_components,
            fov_deg=args.dvd_fov_deg,
            vertical_fov_deg=args.dvd_vertical_fov_deg,
            infant_anchor_cpd=args.dvd_infant_cpd,
            adult_anchor_cpd=args.dvd_adult_cpd,
            sigmoid_width_log10=args.dvd_sigmoid_width_log10,
            apply_acuity=bool(args.apply_blur),
            apply_color=color,
            apply_contrast=bool(args.apply_contrast),
        ),
        backend=args.dvd_backend,
        chunk=args.dvd_chunk_size,
    )


def describe_dvd(args, transformer):
    crop = rrc_from_args(args)
    augmentation = {"rrc": asdict(crop)} if crop.enabled else {}
    if transformer is None:
        return {"method": "adult", **augmentation}
    protocol = {
        "method": "dvd",
        "version": 2,
        "config": asdict(transformer.config),
        "epochs": args.epochs,
        "months_per_epoch": args.months_per_epoch,
        "time_order": args.time_order,
        **augmentation,
    }
    if isinstance(transformer, InverseDVD):
        protocol.update(backend="reflected-batched",
                        grayscale_aug=0 if args.grayscale_aug is None else args.grayscale_aug)
    else:
        protocol.update(backend=transformer.backend, chunk_size=transformer.chunk)
        cfg = transformer.config
        if cfg.speed != 1 and cfg.apply_acuity and "acuity" in cfg.speed_components:
            protocol["acuity_clock"] = "age-odds"
    return protocol


def build_age_months_curve(args, iters_per_epoch: int) -> list[float]:
    if args.time_order == "fully_random":
        total_steps = int(args.epochs) * int(iters_per_epoch)
        max_age = float(args.months_per_epoch) * float(args.epochs)
        g = torch.Generator()
        g.manual_seed(int(args.seed))
        ages = torch.rand(total_steps, generator=g).tolist()
        return [float(a) * max_age for a in ages]

    return generate_age_months_curve(
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
    dvdt=None,
):
    import wandb
    import dvd_scale_free.utils

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
    args = parse_args()
    flipped = args.invert_frequency or args.invert_amplitude
    if args.apply_color is None:
        args.apply_color = int(not flipped)
    if args.grayscale_aug is None:
        args.grayscale_aug = int(not flipped)
    if args.rrc and args.dataset_name == "rgbd_texture2shape_miniecoset":
        parser.error("RRC in this trainer supports the RGB pipeline, not the paired RGB-D pipeline")
    try:
        if args.training_recipe == "p004-paper":
            from dvd_scale_free.models import paper_runner

            paper_runner.check_options(args)
        dvdt = build_dvd_transformer(args)
        args.dvd_protocol = describe_dvd(args, dvdt)
        if args.training_recipe == "p004-paper":
            args.dvd_protocol["training_recipe"] = paper_runner.describe(args, dvdt)
        elif args.output_dir is not None:
            raise ValueError("--output-dir is for --training-recipe p004-paper")
    except ValueError as exc:
        parser.error(str(exc))
    if args.describe_dvd:
        print(json.dumps(args.dvd_protocol, indent=2))
        return
    if args.training_recipe == "p004-paper":
        if args.evaluate and not args.resume:
            parser.error("Paper evaluation requires --resume with a trusted checkpoint")
        if not args.evaluate and not args.output_dir:
            parser.error("Paper training requires --output-dir")
        return paper_runner.run(args, dvdt, build_age_months_curve)
    if args.data is None:
        parser.error("data is required for training or evaluation")

    import numpy as np
    import wandb
    import yaml
    import dvd_scale_free.models.eval
    import dvd_scale_free.models.loader
    import dvd_scale_free.utils

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

    age_months_curve = build_age_months_curve(args, iters_per_epoch=len(train_loader)) if dvdt else None

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
            dvdt=dvdt,
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
                "dvd_protocol": args.dvd_protocol,
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
