#!/usr/bin/env python3
"""Minimal inference example for released DVD ResNet50 checkpoints.

DVD checkpoints were trained with RGB inputs in [0, 1], without mean/std
normalization. This script therefore uses Resize((256, 256)) + ToTensor().

Example:
    python scripts/eval_new_images.py \
        --checkpoint logs/open_source_weights/resnet50_ecoset_DVD_B/weights/checkpoint_best.pth \
        --images assets/example_stimuli/*.jpeg
"""

import argparse
import glob
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from torchvision.models import resnet50


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = ROOT / "logs/open_source_weights/resnet50_ecoset_DVD_B/weights/checkpoint_best.pth"
CLASS_FILES = {
    "ecoset_square256": ROOT / "dvd/datasets/ecoset/lookup_ecoset.json",
    "imagenet": ROOT / "dvd/datasets/imagenet/imagenet_classnames.json",
}
NUM_CLASSES = {
    "texture2shape_miniecoset": 112,
    "ecoset_square256": 565,
    "imagenet": 1000,
}
DATASET_FROM_CLASSES = {v: k for k, v in NUM_CLASSES.items()}
IMAGE_EXTS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
PREFIXES = ("module.", "_orig_mod.", "model.", "net.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--dataset", choices=["auto", *NUM_CLASSES], default="auto")
    parser.add_argument("--num-classes", type=int)
    parser.add_argument("--class-names", type=Path)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    args = parser.parse_args()
    for name in ("image_size", "top_k", "batch_size"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    return args


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def is_state_dict(value):
    return isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values())


def load_state_dict(path):
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")

    for key in ("state_dict", "model_state_dict", "model", "net"):
        if isinstance(ckpt, dict) and is_state_dict(ckpt.get(key)):
            state_dict = ckpt[key]
            break
    else:
        if not is_state_dict(ckpt):
            keys = list(ckpt) if isinstance(ckpt, dict) else type(ckpt)
            raise KeyError(f"Could not find state_dict in checkpoint keys: {keys}")
        state_dict = ckpt

    clean = {}
    for key, value in state_dict.items():
        old_key = None
        while key != old_key:
            old_key = key
            for prefix in PREFIXES:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
        clean[key] = value
    return clean


def infer_num_classes(state_dict, dataset, override):
    if override is not None:
        return override
    if torch.is_tensor(state_dict.get("fc.weight")):
        return int(state_dict["fc.weight"].shape[0])
    return NUM_CLASSES[dataset] if dataset != "auto" else NUM_CLASSES["ecoset_square256"]


def make_resnet50(num_classes):
    try:
        model = resnet50(weights=None)
    except TypeError:
        model = resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def clean_name(value):
    if isinstance(value, dict):
        value = value.get("category", value)
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    value = value.decode("utf-8") if isinstance(value, bytes) else str(value)
    if (value.startswith("b'") and value.endswith("'")) or (
        value.startswith('b"') and value.endswith('"')
    ):
        value = value[2:-1]
    return value.replace("_", " ")


def load_class_names(dataset, num_classes, path=None):
    path = path or CLASS_FILES.get(dataset)
    if path is None:
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        names = [clean_name(x) for x in data]
    else:
        try:
            keys = [int(k) for k in data]
            offset = 1 if min(keys) == 1 and max(keys) == len(keys) else 0
            indexed = {int(k) - offset: v for k, v in data.items()}
            names = [clean_name(indexed.get(i, f"class_{i}")) for i in range(max(indexed) + 1)]
        except ValueError:
            names = [clean_name(x) for x in data.values()]

    if len(names) != num_classes:
        print(f"[WARN] {path} has {len(names)} names, but model has {num_classes} outputs.")
        return None
    return names


def expand_images(inputs):
    images = []
    for item in inputs:
        paths = [Path(x) for x in sorted(glob.glob(item, recursive=True))] or [Path(item)]
        for path in paths:
            path = path.expanduser()
            if path.is_dir():
                images += [p for p in sorted(path.rglob("*")) if p.suffix.lower() in IMAGE_EXTS]
            elif path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                images.append(path)
            else:
                raise FileNotFoundError(f"Image path not found or unsupported: {path}")
    if not images:
        raise FileNotFoundError("No supported image files found.")
    return images


def main():
    args = parse_args()
    ckpt_path = args.checkpoint.expanduser()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_dict = load_state_dict(ckpt_path)
    num_classes = infer_num_classes(state_dict, args.dataset, args.num_classes)
    dataset = DATASET_FROM_CLASSES.get(num_classes, "unknown") if args.dataset == "auto" else args.dataset
    class_names = load_class_names(dataset, num_classes, args.class_names)

    model = make_resnet50(num_classes)
    model.load_state_dict(state_dict, strict=True)
    device = pick_device(args.device)
    model.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
    ])
    image_paths = expand_images(args.images)
    top_k = min(args.top_k, num_classes)

    print(f"Loaded: {ckpt_path}")
    print(f"Model: torchvision ResNet50 | outputs={num_classes} | labels={dataset}")
    print(f"Preprocess: resize=({args.image_size}, {args.image_size}), RGB in [0, 1]")
    print(f"Device: {device}")

    with torch.inference_mode():
        for start in range(0, len(image_paths), args.batch_size):
            paths = image_paths[start : start + args.batch_size]
            images = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths]).to(device)
            logits = model(images)
            probs, ids = logits.softmax(1).topk(top_k, dim=1)

            for path, image_ids, image_probs, image_logits in zip(paths, ids.cpu(), probs.cpu(), logits.cpu()):
                print(f"\n{path}")
                for rank, (class_id, prob) in enumerate(zip(image_ids.tolist(), image_probs.tolist()), 1):
                    name = class_names[class_id] if class_names else f"class_{class_id}"
                    logit = float(image_logits[class_id])
                    print(f"  {rank:>2}. class_id={class_id:<4d} prob={prob:.4f} logit={logit:.4f} {name}")


if __name__ == "__main__":
    main()
