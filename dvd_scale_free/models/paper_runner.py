"""ResNet-50 ImageNet training with the paper augmentation and DVD schedule."""

from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.modules._functions import SyncBatchNorm

from ..dvd_scale_free import DVDConfig, DVDTrainingTransform
from .protocol import PAPER_RECIPE, same_dvd_protocol


RECIPE_DEFAULTS = dict(dataset_name="imagenet", development_strategy="dvd", epochs=150,
                       workers=24, world_size=2, log_freq=50, dvd_backend="batched",
                       apply_color=1, grayscale_aug=1)
SHARED_IMAGENET = Path("/share/klab/datasets/cached_datasets/imagenet_p004")
STEPS_PER_EPOCH = 1251
_FIXED = dict(arch="resnet50", dataset_name="imagenet", image_size=256, batch_size_per_gpu=512,
              optimizer="adam", lr=1e-4, weight_decay=1e-6, lr_scheduler="", warmup_epochs=0,
              label_smoothing=0.0, world_size=2, dist_url="env://", epochs=150, months_per_epoch=2.0,
              time_order="chronological", development_strategy="dvd", apply_blur=1, apply_color=1,
              apply_contrast=1, grayscale_aug=1, blur_aug=1, additional_aug=1, resize_to_224=0,
              resize_to_256=0, rrc=False, crop_min=1.0, pretrained="", start_epoch=0, best_acc1=0.0)


def check_options(args):
    for key, expected in _FIXED.items():
        actual = getattr(args, key)
        if actual != expected:
            raise ValueError(f"The {PAPER_RECIPE} recipe requires {key}={expected!r}, got {actual!r}")
    if args.workers < 0 or args.log_freq <= 0 or args.save_checkpoint_every_epochs <= 0:
        raise ValueError("Invalid worker count or logging/checkpoint interval")


def describe(args, dvd):
    check_options(args)
    if (not isinstance(dvd, DVDTrainingTransform) or dvd.config != DVDConfig()
            or dvd.backend != "batched" or dvd.chunk != 128):
        raise ValueError(f"The {PAPER_RECIPE} recipe requires forward V=1 DVD with both spatial stages, "
                         "default anchors, 15 degrees, and the batched backend")
    return dict(name=PAPER_RECIPE, version=1, seed=args.seed, world_size=2, batch_size_per_rank=512,
                steps_per_epoch=STEPS_PER_EPOCH, train_workers=args.workers, val_workers=8,
                optimizer={"name": "adam", "lr": args.lr, "weight_decay": args.weight_decay, "fused": True},
                augmentation="paper-kornia-batched-gpu-before-dvd", effective_input="full-frame-256",
                rrc=False, criterion="class-weighted-cross-entropy", amp="fp16",
                memory_format="channels_last", batch_norm="full-batch-syncbn",
                validation="unfiltered-eager-global-50000", resume="rank-rng-and-input-manifest")


class _FullBatchSyncNorm(SyncBatchNorm):
    @staticmethod
    def forward(ctx, x, weight, bias, running_mean, running_var, eps, momentum, group, world):
        # The fixed sampler supplies the same nonempty batch size to both CUDA ranks.
        if not x.is_cuda or not x.numel() or dist.get_backend(group) != "nccl":
            raise RuntimeError("Full-batch SyncBN requires nonempty NCCL CUDA ranks")
        if not x.is_contiguous(memory_format=torch.channels_last):
            x = x.contiguous()
        weight = weight.contiguous() if weight is not None else None
        channels = x.shape[1]
        mean, invstd = torch.batch_norm_stats(x, eps)
        count = mean.new_full((1,), x.numel() // channels)
        combined = torch.cat((mean, invstd, count))
        merged = combined.new_empty(world, combined.numel())
        dist.all_gather_into_tensor(merged.flatten(), combined, group)
        means, invstds, counts = merged.split(channels, dim=1)
        backward_counts = counts.to(torch.int32).contiguous()
        counts = counts.flatten().contiguous()
        if running_mean is not None:
            counts = counts.to(running_mean.dtype)
        mean, invstd = torch.batch_norm_gather_stats_with_counts(
            x, means.contiguous(), invstds.contiguous(), running_mean, running_var, momentum, eps, counts)
        ctx.save_for_backward(x, weight, mean, invstd, backward_counts)
        ctx.process_group = group
        return torch.batch_norm_elemt(x, weight, bias, mean, invstd, eps)


def paper_optimizer(parameters, *, lr=1e-4, weight_decay=1e-6):
    return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay, fused=True)


def _init_cuda(args):
    args.rank = int(os.environ.get("RANK", -1))
    args.gpu = int(os.environ.get("LOCAL_RANK", -1))
    if (args.rank not in (0, 1) or args.gpu != args.rank or int(os.environ.get("WORLD_SIZE", 1)) != 2
            or int(os.environ.get("LOCAL_WORLD_SIZE", 1)) != 2
            or not torch.cuda.is_available() or not dist.is_nccl_available() or torch.cuda.device_count() < 2):
        raise ValueError(f"The {PAPER_RECIPE} recipe requires torchrun --standalone --nproc-per-node=2 and two CUDA GPUs")
    for name, value in dict(NCCL_SOCKET_IFNAME="lo", NCCL_NET="Socket", NCCL_P2P_DISABLE="0",
                            NCCL_CUMEM_ENABLE="0", NCCL_CUMEM_HOST_ENABLE="0").items():
        os.environ.setdefault(name, value)
    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda", args.gpu)
    dist.init_process_group("nccl", timeout=timedelta(seconds=300), device_id=device)
    dist.barrier(device_ids=[args.gpu])
    return device


def _headroom():
    if os.environ.get("SLURM_JOB_ID"):
        requested = os.environ.get("SLURM_MEM_PER_NODE", "")
        per_cpu = os.environ.get("SLURM_MEM_PER_CPU", "")
        cpus = os.environ.get("SLURM_CPUS_ON_NODE", "")
        budget_mib = int(requested) if requested.isdecimal() else None
        if budget_mib is None and per_cpu.isdecimal() and cpus.isdecimal():
            budget_mib = int(per_cpu) * int(cpus)
        if budget_mib is not None and 0 < budget_mib < 80 * 1024:
            raise RuntimeError(f"Paper loader: Slurm grants {budget_mib / 1024:.1f} GiB RAM; "
                               "request at least 80 GiB (--mem=105G matches the reference allocation).")
    meminfo = Path("/proc/meminfo")
    shm = Path("/dev/shm")
    if meminfo.is_file() and shm.is_dir():
        available = next(int(line.split()[1]) for line in meminfo.read_text().splitlines()
                         if line.startswith("MemAvailable:")) / 1024**2
        stat = os.statvfs(shm)
        free_shm = stat.f_bavail * stat.f_frsize / 1024**3
        if available < 80 or free_shm < 24:
            raise RuntimeError(f"Paper loader preflight: need 80 GiB available RAM and 24 GiB free IPC shm; "
                               f"found {available:.1f} and {free_shm:.1f} GiB. No dataset is copied into RAM.")


@torch.no_grad()
def validate(loader, model, criterion, device, *, expected_count):
    """Compute full-set weighted loss and exact hit counts, never rank averages."""
    model.eval()
    stats = torch.zeros(5, device=device, dtype=torch.float64)
    for images, labels in loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = model(images)
            loss = criterion(output, labels)
        hits = output.topk(5, dim=1).indices.eq(labels[:, None])
        weight = criterion.weight[labels].sum() if criterion.weight is not None else labels.numel()
        stats[0] += loss.double() * weight
        stats[1] += weight
        stats[2] += hits[:, 0].sum()
        stats[3] += hits.any(dim=1).sum()
        stats[4] += labels.numel()
    if dist.is_initialized():
        dist.all_reduce(stats)
    loss, weight, top1_hits, top5_hits, samples = stats.tolist()
    if samples != expected_count or weight <= 0 or not math.isfinite(loss):
        raise RuntimeError(f"Invalid validation: samples={samples:g}/{expected_count}, loss={loss}, weight={weight}")
    return dict(samples=int(samples), top1=100*top1_hits/samples, top5=100*top5_hits/samples, loss=loss/weight)


def update_batch(model, optimizer, scaler, criterion, x, target):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(x.device.type, dtype=torch.float16, enabled=x.is_cuda):
        logits = model(x)
        loss = criterion(logits, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return logits.detach(), loss.detach()


def train_epoch(args, device, loader, model, optimizer, scaler, criterion, augment, dvd, ages, epoch):
    model.train()
    stream = torch.cuda.current_stream(device)
    prep = torch.cuda.Stream(device=device)
    prep.wait_stream(stream)
    stats = torch.zeros(3, device=device, dtype=torch.float64)
    samples = 0
    for step, (raw, labels) in enumerate(loader):
        index = epoch * len(loader) + step
        with torch.no_grad(), torch.cuda.stream(prep):
            x = raw.to(device, non_blocking=True)
            target = labels.to(device, non_blocking=True)
            x = dvd(augment(x), months=ages[index]).contiguous(memory_format=torch.channels_last)
        stream.wait_stream(prep)
        x.record_stream(stream)
        target.record_stream(stream)
        logits, loss = update_batch(model, optimizer, scaler, criterion, x, target)
        hits = logits.topk(5, dim=1).indices.eq(target[:, None])
        stats[0] += loss.double()*len(x)
        stats[1] += hits[:, 0].sum()
        stats[2] += hits.any(dim=1).sum()
        samples += len(x)
        if (step + 1 == 16 or (step + 1) % args.log_freq == 0) and args.rank == 0:
            print(f"Paper epoch {epoch+1}, step {step+1}/{len(loader)}, age {ages[index]:.3f} mo", flush=True)
    loss, correct, correct5 = stats.tolist()
    if samples != len(loader) * args.batch_size_per_gpu or not math.isfinite(loss):
        raise RuntimeError("Invalid paper training epoch")
    return dict(samples_per_rank=samples, rank0_loss=loss/samples, rank0_top1=100*correct/samples,
                rank0_top5=100*correct5/samples)


def _local_rng(device):
    return dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                numpy=np.random.get_state(), python=random.getstate())


def _restore_rng(state, device):
    torch.set_rng_state(state["cpu"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"], device)
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])


def _check_checkpoint(state, args, manifest, ages):
    if "dvd_protocol" in state:
        if not same_dvd_protocol(state["dvd_protocol"], args.dvd_protocol) or state.get("input_manifest") != manifest:
            raise ValueError("Checkpoint recipe or ImageNet manifest differs")
    else:
        native = state.get("args", {})
        keys = ("arch", "dataset_name", "image_size", "batch_size_per_gpu", "optimizer", "lr", "weight_decay",
                "lr_scheduler", "label_smoothing", "world_size", "epochs", "months_per_epoch", "time_order",
                "development_strategy", "apply_blur", "apply_color", "seed", "grayscale_aug", "blur_aug",
                "additional_aug", "crop_min")
        if (state.get("training_variant") != "linear_v1" or native.get("apply_contrast") != 0
                or any(native.get(key) != getattr(args, key) for key in keys)):
            raise ValueError("Not a matching native paper checkpoint (linear_v1)")
    epoch = int(state["epoch"])
    if not -1 <= epoch < args.epochs or state.get("age_months_curve") != ages:
        raise ValueError("Checkpoint epoch or age schedule differs")
    if args.evaluate:
        return
    if len(state.get("rank_rng", ())) != args.world_size or state.get("scaler") is None:
        raise ValueError("Continuing training requires the saved state of both ranks and the GradScaler")
    groups = state["optimizer"]["param_groups"]
    if not groups or any(group["lr"] != args.lr or group["weight_decay"] != args.weight_decay
                         or not group.get("fused") for group in groups):
        raise ValueError("Checkpoint does not use the matching fused Adam settings")


def _save_atomic(path, payload):
    temp = path.with_name(f".{path.name}.{os.getpid()}.building")
    try:
        with temp.open("wb") as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _alias_atomic(source, target):
    temp = target.with_name(f".{target.name}.{os.getpid()}.building")
    try:
        os.link(source, temp)
        temp.replace(target)
    finally:
        temp.unlink(missing_ok=True)


def _save_run_state(path, args, model, optimizer, scaler, device, manifest, ages, epoch, best):
    rank_rng = [None] * dist.get_world_size()
    dist.all_gather_object(rank_rng, _local_rng(device))
    if args.rank == 0:
        _save_atomic(path, dict(epoch=epoch, arch=args.arch, state_dict=model.module.state_dict(),
                                optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                                best_acc1=best, rank_rng=rank_rng, age_months_curve=ages,
                                dvd_protocol=args.dvd_protocol, input_manifest=manifest, args=dict(vars(args))))


def _output_dir(args):
    root = Path(args.output_dir).resolve()
    latest = root / "weights/checkpoint_last.pth"
    init = root / "weights/checkpoint_init.pth"
    if (latest.exists() and (not args.resume or Path(args.resume).resolve() != latest)
            or not latest.exists() and init.exists() and (not args.resume or Path(args.resume).resolve() != init)):
        raise ValueError("Output contains another paper run. Resume its last checkpoint or select a new output directory.")
    return root


def run(args, dvd, build_ages):
    import kornia
    import PIL
    import torchvision
    from ..datasets.paper_imagenet import VAL_IMAGES, paper_loaders
    from ..utils import calculate_class_weights_from_imagefolder, fix_random_seeds
    from .loader import create_model

    root = Path(args.imagenet_path or args.data or SHARED_IMAGENET).resolve()
    if not (root / "train").is_dir() or not (root / "val").is_dir():
        raise ValueError(f"Point --imagenet-path to the full ImageNet-1K train/val tree: {root}")
    output = None if args.evaluate else _output_dir(args)
    _headroom()
    device = _init_cuda(args)
    from torch.nn.modules import batchnorm

    previous = batchnorm.sync_batch_norm
    batchnorm.sync_batch_norm = _FullBatchSyncNorm
    try:
        fix_random_seeds(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        model, _ = create_model(args)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device, memory_format=torch.channels_last)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu,
                                                     broadcast_buffers=False)
        optimizer = paper_optimizer(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scaler = torch.amp.GradScaler("cuda")
        train, val, augment, manifest = paper_loaders(args, root)
        if len(train) != STEPS_PER_EPOCH:
            raise RuntimeError(f"Expected {STEPS_PER_EPOCH} ImageNet batches per rank; got {len(train)}")
        manifests = [None] * args.world_size
        dist.all_gather_object(manifests, manifest)
        if any(item != manifest for item in manifests):
            raise RuntimeError("ImageNet IDs or JPEG spot checks differ between ranks")
        augment = augment.to(device)
        criterion = nn.CrossEntropyLoss(weight=calculate_class_weights_from_imagefolder(train.dataset).to(device))
        ages = build_ages(args, len(train))
        start, best = 0, 0.0
        if args.resume:
            # Only run-owned, trusted checkpoints contain Python/NumPy RNG objects.
            state = torch.load(args.resume, map_location="cpu", mmap=True, weights_only=False)
            _check_checkpoint(state, args, manifest, ages)
            model.module.load_state_dict(state["state_dict"], strict=True)
            start, best = int(state["epoch"]) + 1, float(state["best_acc1"])
            if not args.evaluate:
                optimizer.load_state_dict(state["optimizer"])
                scaler.load_state_dict(state["scaler"])
                _restore_rng(state["rank_rng"][args.rank], device)
            del state
        if args.evaluate:
            result = validate(val, model, criterion, device, expected_count=VAL_IMAGES)
            if args.rank == 0:
                print(json.dumps(dict(epoch=start, **result)), flush=True)
            return

        weights = output / "weights"
        last = weights / "checkpoint_last.pth"
        if args.rank == 0:
            weights.mkdir(parents=True, exist_ok=True)
            details = dict(protocol=args.dvd_protocol, input_manifest=manifest, data=str(root),
                           software=dict(torch=torch.__version__, torchvision=torchvision.__version__,
                                         kornia=kornia.__version__, pillow=PIL.__version__, cuda=torch.version.cuda),
                           gpus=[torch.cuda.get_device_name(i) for i in range(2)], resume=args.resume)
            (output / "protocol.json").write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
        if start == 0 and not args.resume:
            _save_run_state(weights / "checkpoint_init.pth", args, model, optimizer, scaler, device, manifest, ages, -1, best)
        for epoch in range(start, args.epochs):
            tick = time.monotonic()
            train.sampler.set_epoch(epoch)
            training = train_epoch(args, device, train, model, optimizer, scaler, criterion, augment, dvd, ages, epoch)
            results = validate(val, model, criterion, device, expected_count=VAL_IMAGES)
            is_best = results["top1"] > best
            best = max(best, results["top1"])
            _save_run_state(last, args, model, optimizer, scaler, device, manifest, ages, epoch, best)
            if args.rank == 0:
                if is_best:
                    _alias_atomic(last, weights / "checkpoint_best.pth")
                if (epoch + 1) % args.save_checkpoint_every_epochs == 0:
                    _alias_atomic(last, weights / f"checkpoint_{epoch}.pth")
                line = json.dumps(dict(epoch=epoch+1, training=training, validation=results, best=best,
                                       seconds=time.monotonic()-tick))
                print(line, flush=True)
                with (output / "metrics.jsonl").open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            dist.barrier(device_ids=[args.gpu])
    finally:
        batchnorm.sync_batch_norm = previous
        dist.destroy_process_group()
