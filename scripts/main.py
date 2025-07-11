
import argparse
import math
import os
import sys
import time
import yaml
import wandb 
import logging
from logging.config import fileConfig
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as torchvision_models

import dvd.utils
import dvd.dvd.development
import dvd.models.loader
import dvd.models.eval
from dvd.datasets.dataset_loader import SupervisedLearningDataset
from dvd.dvd.development import DVDTransformer, DVDConfig, AgeCurve

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
    default=0,
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
                    help='development strategy (default: dvd)',
                    )
parser.add_argument(
    "--time-order",
    default="chronological",
    type=str,
    choices=["chronological", "mid_phase", "random", "fully_random"],
    help="time order of the batches",
)
parser.add_argument(
    "--months-per-epoch",
    default=2,
    type=float,
    help="number of months per epoch",
)
parser.add_argument(
    "--contrast_amplitude_beta",
    default=0.1,
    type=float,
    help="beta in the paper, determines the base amplitude threshold in the frequency domain to map the initial contrast sensitivity in the spatial domain at birth,",
)
parser.add_argument(
    "--contrast_amplitude_lambda",
    default=150,
    type=float,
    help="decrease contrast drop speed (default: 100)",
)

# Ablations
parser.add_argument('--apply_blur', type=int, default=1, help='Flag to apply blur to images')
parser.add_argument('--apply_color', type=int, default=1, help='Flag to apply color changes')
parser.add_argument('--apply_threshold_color', type=int, default=0, help='Flag to apply threshold color changes')
parser.add_argument('--apply_contrast', type=int, default=1, help='Flag to apply contrast adjustments')
parser.add_argument('--apply_contrast_by_percentile', type=int, default=0, help='Flag to apply contrast adjustments in percentile')



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

# resize to 224
parser.add_argument(
    "--resize_to_224",
    default=0,
    type=int,
    help="Resize images to 224x224",
) # args.resize_to_224
parser.add_argument(
    "--resize_to_256",
    default=0,
    type=int,
    help="Resize images to 256x256",
) # args.resize_to_256

# no grayscale
parser.add_argument(
    "--grayscale-aug",
    default=1,
    type=int,
    help="apply grayscale transformation",
) # args.grayscale_aug
parser.add_argument(
    "--blur-aug",
    default=1,
    type=int,
    help="apply blur transformation",
) # args.blur_aug

# best_acc1
parser.add_argument("--best-acc1", default=0.0, type=float,
                    help="Best accuracy achieved so far")

def setup_logging_and_wandb(args):
    """
    Sets up logging and initializes WandB run if main process.
    Returns logger, wandb_run (or None), and log_dir.
    """
    # Initialize distributed training
    dvd.utils.init_distributed_mode(args)
    dvd.utils.fix_random_seeds(args.seed)
    # If you really want bit-for-bit reproducibility (slower), enable:
    # cudnn.deterministic = True
    cudnn.benchmark = True # find the best algorithm to use for your hardware, but it can introduce some variability

    # Setup logging
    fileConfig("dvd/models/logging/config.ini")
    logger = logging.getLogger()
    logger.disabled = True  # Will enable if main process

    if args.development_strategy == 'dvd':
        net_name = (
            f'{args.arch}_mpe{args.months_per_epoch}_alpha{args.contrast_amplitude_beta}'
            f'_dn{args.contrast_amplitude_lambda}_{args.dataset_name}'
            f'{args.image_size}_{args.lr_scheduler}{args.lr}_dev_{args.development_strategy}'
            f'_b{args.apply_blur}c{args.apply_color}cs{args.apply_contrast}'
            f'_T_{args.time_order}_seed_{args.seed}'
        ) 
    else:
        net_name = f'{args.arch}_{args.dataset_name}_{args.image_size}_{args.lr_scheduler}{args.lr}_dev_{args.development_strategy}_seed_{args.seed}'
    
    if not args.blur_aug:
        net_name += f'_no_blur_aug' #* just debug
    if args.apply_threshold_color:
        net_name += f'_threshold_color'

    wandb_run = None

    if dvd.utils.is_main_process():
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


def stack_collate(batch):
    imgs, labels = zip(*batch)
    # imgs = [img if torch.is_tensor(img) else torch.tensor(img) for img in imgs]
    labels = [lbl if torch.is_tensor(lbl) else torch.tensor(lbl) for lbl in labels]
    return torch.stack(imgs,   dim=0), \
           torch.stack(labels, dim=0)
    
def get_data_loaders(args):
    """
    Retrieves the training and validation datasets and wraps them into DataLoaders 
    with distributed sampling.
    """
    dataset_name = args.dataset_name
    dataset = SupervisedLearningDataset(args.data, args) # Load dataset from path (args.data) & all
    dataset = dataset.get_dataset(dataset_name)
    train_dataset, val_dataset, _ = dataset["train"], dataset["val"], dataset["test"]

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    print(f"train_sampler: {train_sampler}")
    print(f"val_sampler: {val_sampler}")

    # 3) decide if we need our custom collate *only* for ImageNet
    is_imagenet = (dataset_name == 'imagenet')
    collate_fn  = stack_collate if is_imagenet else None

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

    val_loader =  torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size_per_gpu,
        shuffle=None,
        num_workers=args.workers,
        pin_memory=True,
        sampler=val_sampler,
        drop_last=False,                 # <-- changed #* for small dataset, num/2<512 cannot drop
        **({'collate_fn': collate_fn} if collate_fn else {})
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
    batch_time = dvd.utils.AverageMeter("Time", ":6.3f")
    data_time = dvd.utils.AverageMeter("Data", ":6.3f")
    learning_rates = dvd.utils.AverageMeter("LR", ":.4e")
    losses = dvd.utils.AverageMeter("Loss", ":.4e")
    top1 = dvd.utils.AverageMeter("Acc@1", ":6.2f")
    top5 = dvd.utils.AverageMeter("Acc@5", ":6.2f")

    # Progress meter display settings
    progress = dvd.utils.ProgressMeter(
        len(train_loader),
        [batch_time, data_time, learning_rates, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()

    end = time.time()
    iters_per_epoch = len(train_loader)

    # Generate age months curve to map batches to age months for DVD
    age_months_curve = AgeCurve.generate(
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
            lr = dvd.utils.adjust_learning_rate(
                optimizer, epoch + i / iters_per_epoch, args
            )
            learning_rates.update(lr)
        elif args.lr_scheduler == '':
            lr = args.lr
        else:
            raise NotImplementedError(
                f"Development strategy {args.lr_scheduler} not implemented"
            )

        # Get age in months (for DVD transformations) | epoch start from 1 so -1
        age_months = age_months_curve[(epoch -0) * len(train_loader) + i]
        

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # Experience across visual development
        if args.development_strategy == "dvd":
            # images = dvd.dvd.development.DVDTransformer().apply_fft_transformations(
            #     images,
            #     age_months,
            #     apply_blur=args.apply_blur, 
            #     apply_color=args.apply_color, 
            #     apply_contrast=args.apply_contrast,
            #     contrast_amplitude_beta=args.contrast_amplitude_beta,
            #     contrast_amplitude_lambda = args.contrast_amplitude_lambda,
            #     apply_threshold_color=args.apply_threshold_color,
            #     image_size=args.image_size,
            #     fully_random=(args.time_order == "fully_random"), # just for control models
            #     age_months_curve= age_months_curve,
            #     verbose=False,
            # )
            dvdt =  DVDTransformer(DVDConfig(
                                    blur=args.apply_blur, color=args.apply_color, contrast=args.apply_contrast,
                                    beta=args.contrast_amplitude_beta, lam=args.contrast_amplitude_lambda, 
                                    threshold_color=args.apply_threshold_color,
                                    image_size=args.image_size,
                                    by_percentile=args.apply_contrast_by_percentile,  
                                )
                            )
            images = dvdt(images, age_months, curriculum=age_months_curve, randomise=(args.time_order == "fully_random"), 
                           verbose=False,)
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
        acc1, acc5 = dvd.utils.accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # Log with wandb if main process
        if dvd.utils.is_main_process() and wandb_run is not None and it % args.log_freq == 0:
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
    if dvd.utils.is_main_process() and log_dir is not None:
        dvd.utils.save_config(args, os.path.join(log_dir, f"config.yaml")) # Saving config setup

    # 2) Create and setup model
    model, linear_keyword = dvd.models.loader.create_model(args, logger)
    dvd.models.loader.load_pretrained_weights_if_any(args, model, linear_keyword)
    
    # Optionally compile the model (PyTorch 2.0+) to speed up training
    if hasattr(torch, "compile"):
        model = torch.compile(model)

    # 3) Build optimizer and FP16 scaler
    model, optimizer, scaler = dvd.models.loader.build_optimizer_and_scaler(args, model)
    if dvd.utils.is_main_process() and log_dir is not None:
        dvd.utils.save_initial_checkpoint(log_dir, args, model, optimizer, scaler, logger, net_name) 
            
    # 4) Possibly resume checkpoint
    if dvd.utils.is_main_process() and log_dir is not None:
        dvd.models.loader.resume_checkpoint_if_any(args, model, optimizer, scaler, logger, log_dir)

    # Prepare to log stats if main process
    if dvd.utils.is_main_process() and log_dir is not None:
        stats_file = open(os.path.join(log_dir, "stats.txt"), "a", buffering=1)
        logger.info(" ".join(sys.argv))
        print(" ".join(sys.argv), file=stats_file)
        with open(os.path.join(log_dir, "metadata.txt"), "a") as f:
            yaml.dump(args, f, allow_unicode=True)
            f.write(str(model))
        
    # 5) Get data loaders
    train_loader, val_loader, train_sampler, val_sampler = get_data_loaders(args)
    print(f"len loaders : {len(train_loader)}  |  {len(val_loader)} |")
    # import pdb;pdb.set_trace()
    #TODO fix len(val_loader) is 0
    

    # 6) Optionally only evaluate
    if args.evaluate:
        dvd.models.eval.validate(val_loader, model, criterion, epoch, args.gpu)
        return

    logger.info("Main components ready.")
    logger.info(model)
    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Scaler: {scaler}")
    logger.info("Starting model training.")

    best_acc1 = args.best_acc1
    try:
        criterion = dvd.utils.get_loss_function(args)
    except:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing).cuda(args.gpu)

    for epoch in range(args.start_epoch, args.epochs):
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
        acc1, _ = dvd.models.eval.validate(val_loader, model, criterion, epoch, args.gpu, wandb_run, logger)
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if dvd.utils.is_main_process() and log_dir is not None:
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
            dvd.utils.save_checkpoint(
                checkpoint_dict,
                is_best=is_best,
                filename=os.path.join(log_dir, 'weights', filename),
            )
            dvd.utils.save_last_checkpoint(
                checkpoint_dict,
                filename=os.path.join(log_dir, 'weights', "checkpoint_last.pth"),
            )

    if dvd.utils.is_main_process():
        wandb.finish()  


if __name__ == "__main__":
    main()
