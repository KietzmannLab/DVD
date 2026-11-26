"""Run the paper trainer on a small synthetic split with two GPUs, then resume."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from PIL import Image
import torch
from torch import nn

from dvd_scale_free.datasets import paper_imagenet
from dvd_scale_free.models import loader, paper_runner
from scripts import main as cli


def fixture(root):
    for split in ("train", "val"):
        for label in range(6):
            folder = root / split / f"{label:02d}"
            folder.mkdir(parents=True)
            for i in range(4 if split == "train" else 1):
                Image.new("RGB", (167+i*13, 194), (34+label*26, 76+i*20, 162)).save(folder / f"{i}.jpeg")


def small_model(args):
    model = nn.Sequential(nn.Conv2d(3, 16, 5, stride=4), nn.BatchNorm2d(16), nn.ReLU(),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 6))
    return model, "5"


def worker(root, *, resume):
    argv = ["--training-recipe", "p004-paper", "--workers", "0", "--seed", "5",
            "--imagenet-path", str(root / "fixture"), "--output-dir", str(root / "run"),
            "--save-checkpoint-every-epochs", "1"]
    if resume:
        argv += ["--resume", str(root / "run/weights/checkpoint_last.pth")]
    args = cli.parse_args(argv)
    args.apply_color = args.grayscale_aug = 1
    dvd = cli.build_dvd_transformer(args)
    recipe = paper_runner.describe(args, dvd)
    args.dvd_protocol = cli.describe_dvd(args, dvd)
    args.dvd_protocol["training_recipe"] = dict(recipe, fixture_only=True)
    # Only the fixture uses reduced lengths and a tiny model. Use the real
    # augmentation, SyncBN, fused optimizer, validator, and checkpoint path.
    args.epochs = 2 if resume else 1
    args.batch_size_per_gpu = 2
    epoch_fn = paper_runner.train_epoch

    def checked_resume(args, device, batches, model, optim, scaler, *rest):
        if resume:
            old = torch.load(args.resume, map_location="cpu", weights_only=False, mmap=True)
            torch.testing.assert_close(optim.state_dict(), old["optimizer"], rtol=0, atol=0, check_device=False)
            assert scaler.state_dict() == old["scaler"]
            assert torch.equal(torch.get_rng_state(), old["rank_rng"][args.rank]["cpu"])
            assert torch.equal(torch.cuda.get_rng_state(device), old["rank_rng"][args.rank]["cuda"])
        return epoch_fn(args, device, batches, model, optim, scaler, *rest)

    with patch.multiple(paper_imagenet, TRAIN_IMAGES=24, VAL_IMAGES=6, N_CLASSES=6), \
            patch.multiple(paper_runner, STEPS_PER_EPOCH=6, _headroom=lambda: None, train_epoch=checked_resume), \
            patch.object(loader, "create_model", small_model):
        paper_runner.run(args, dvd, lambda args, n: [i * 2 / n for i in range(2*n)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    if "RANK" in os.environ:
        return worker(root, resume=args.resume)
    root.mkdir()
    fixture(root / "fixture")
    argv = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=2",
            __file__, str(root)]
    subprocess.run(argv, check=True)
    subprocess.run([*argv, "--resume"], check=True)
    weights = root / "run/weights"
    first, state = (torch.load(weights / name, map_location="cpu", weights_only=False)
                    for name in ("checkpoint_0.pth", "checkpoint_last.pth"))
    records = [json.loads(line) for line in (root / "run/metrics.jsonl").read_text().splitlines()]
    assert state["epoch"] == 1 and len(state["rank_rng"]) == 2
    assert [line["epoch"] for line in records] == [1, 2]
    assert all(line["validation"]["samples"] == 6 for line in records)
    # FP16 overflow can skip updates; the serialized steps must continue, not reset.
    earlier = first["optimizer"]["state"]
    updates = {int(slot["step"]) for slot in state["optimizer"]["state"].values()}
    for key, slot in state["optimizer"]["state"].items():
        assert 0 < int(earlier[key]["step"]) < int(slot["step"]) <= int(earlier[key]["step"]) + 6
    print(json.dumps(dict(passed=True, resumed=True, gpu_ranks=2, batches_per_rank=12,
                          optimizer_updates=sorted(updates), epochs=records), indent=2))


if __name__ == "__main__":
    main()
