"""Native paper JPEG geometry, ordered ImageNet batches, and per-image GPU augmentation."""

import hashlib
import json
import os
from pathlib import Path

import kornia.augmentation as K
from kornia.enhance import equalize
import numpy as np
import torch
from torch import nn
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

from .dataset_loader import SupervisedLearningDataset


IMAGE_SIZE = 256
TRAIN_IMAGES, VAL_IMAGES, N_CLASSES = 1_281_167, 50_000, 1000
PAPER_OPS = ("RandomHorizontalFlip", "RandomRotation", "RandomBrightness", "RandomGrayscale",
             "RandomEqualize", "RandomPerspective", "RandomSharpness", "RandomGaussianBlur")


class _FullFrameResize:
    def __call__(self, image):
        x = transforms.functional.to_tensor(image)
        return interpolate(x[None], (IMAGE_SIZE, IMAGE_SIZE), mode="bilinear",
                           align_corners=True, antialias=False)[0]


class _PackedSamples:
    def __init__(self, samples):
        # Store paths compactly so forked JPEG workers do not copy millions of Python objects.
        self.paths = np.asarray([os.fsencode(path) for path, _ in samples])
        self.labels = np.asarray([target for _, target in samples], dtype=np.int32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return os.fsdecode(self.paths[i]), int(self.labels[i])


def _batched_equalize(x):
    channels = x.reshape(-1, IMAGE_SIZE * IMAGE_SIZE)
    raw = channels * 255.0
    bins = ((raw * 256) / raw.new_tensor(255)).to(torch.int64).clamp_(0, 255)
    pieces = bins.reshape(len(channels), -1, 1024)
    counts = torch.zeros((*pieces.shape[:-1], 256), dtype=torch.int32, device=x.device)
    counts.scatter_add_(-1, pieces, torch.ones_like(pieces, dtype=torch.int32))
    hist = counts.sum(1, dtype=torch.int32)
    last = (hist.ne(0) * torch.arange(256, device=x.device)).amax(1, keepdim=True)
    step = (channels.shape[1] - hist.gather(1, last)) // 255
    cdf = hist.cumsum(1, dtype=torch.int32) - hist
    lut = ((cdf + step // 2) // step.clamp_min(1)).clamp_(0, 255)
    adjusted = lut.gather(1, raw.to(torch.int64)).to(dtype=x.dtype)
    return (torch.where(step != 0, adjusted, raw) / 255).reshape_as(x)


def _paper_equalize(x):
    if not x.is_cuda or x.dtype != torch.float32 or x.ndim != 4 or x.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
        return equalize(x)
    channels = x.reshape(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
    lo, hi = channels.flatten(1).aminmax(dim=1)
    valid = (lo >= 0) & (hi <= 1)
    keep = valid.nonzero().flatten()
    if not len(keep):
        return equalize(x)
    if len(keep) == len(channels):
        return _batched_equalize(x)
    # Kornia's reference handles tiny overshoots independently per channel.
    other = (~valid).nonzero().flatten()
    out = torch.empty_like(channels)
    out.index_copy_(0, keep, _batched_equalize(channels.index_select(0, keep)))
    out.index_copy_(0, other, equalize(channels.index_select(0, other)))
    return out.reshape_as(x)


class _PaperEqualize(K.RandomEqualize):
    def apply_transform(self, input, params, flags, transform=None):
        return _paper_equalize(input)


def paper_transforms(args):
    source = SupervisedLearningDataset(".", args)
    reference = source.get_supervised_pipeline_transform(train=False)
    original = list(source.get_supervised_pipeline_transform(train=True).aug_pipe.children())
    if tuple(type(op).__name__ for op in original) != ("ConvertImageDtype", "Resize", *PAPER_OPS):
        raise RuntimeError("The paper augmentation sequence changed")
    ops = original[2:]
    if any(op.same_on_batch for op in ops):
        raise RuntimeError("Paper augmentation must draw parameters independently per image")
    op = ops[4]
    ops[4] = _PaperEqualize(same_on_batch=op.same_on_batch, p=op.p, keepdim=op.keepdim)
    pil_resize = transforms.Resize(IMAGE_SIZE)
    return (transforms.Compose([pil_resize, _FullFrameResize()]),
            transforms.Compose([pil_resize, reference]), nn.Sequential(*ops))


def _manifest(dataset):
    ordered = hashlib.sha256()
    prefix = os.fspath(dataset.root) + os.sep
    ordered.update(json.dumps(dataset.class_to_idx, sort_keys=True, separators=(",", ":")).encode())
    for path, label in dataset.samples:
        ordered.update(os.fsencode(path.removeprefix(prefix)))
        ordered.update(label.to_bytes(4, "little") + b"\0")
    jpeg = hashlib.sha256()
    for i in range(64):
        path, _ = dataset.samples[i * len(dataset) // 64]
        jpeg.update(hashlib.sha256(Path(path).read_bytes()).digest())
    return dict(samples=len(dataset), classes=len(dataset.classes), ordered_ids_sha256=ordered.hexdigest(),
                jpeg_spotcheck_n=64, jpeg_spotcheck_sha256=jpeg.hexdigest())


def _stack(batch):
    x, y = zip(*batch)
    return torch.stack(x), torch.tensor(y, dtype=torch.int64)


def paper_loaders(args, root):
    resize, reference, augment = paper_transforms(args)
    train, val = [ImageFolder(os.fspath(root / split), transform=resize) for split in ("train", "val")]
    if (len(train), len(val), len(train.classes)) != (TRAIN_IMAGES, VAL_IMAGES, N_CLASSES):
        raise ValueError(f"Expected the full ImageNet-1K splits; got {len(train)} train, {len(val)} val, "
                         f"{len(train.classes)} classes")
    if train.class_to_idx != val.class_to_idx:
        raise ValueError("ImageNet train and validation class indices differ")
    manifest = {}
    for split, dataset in (("train", train), ("val", val)):
        for j in range(32):
            path, _ = dataset.samples[j * len(dataset) // 32]
            image = dataset.loader(path)
            torch.testing.assert_close(resize(image), reference(image), atol=0, rtol=0)
        manifest[split] = _manifest(dataset)
        dataset.samples = dataset.imgs = _PackedSamples(dataset.samples)

    loaders = []
    for i, dataset in enumerate((train, val)):
        training = i == 0
        workers = args.workers if training else 8
        sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank, shuffle=training)
        loaders.append(DataLoader(dataset, batch_size=args.batch_size_per_gpu, sampler=sampler,
                                  drop_last=training, num_workers=workers, pin_memory=True,
                                  generator=torch.Generator().manual_seed(args.seed + 2*args.rank + i),
                                  prefetch_factor=1 if workers else None, persistent_workers=training and workers > 0,
                                  collate_fn=_stack))
    return *loaders, augment, manifest
