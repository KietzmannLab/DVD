"""Paper protocol, sampler, optimizer, and two-rank validation/resume checks."""

from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch import nn
from torch.multiprocessing import spawn

from dvd_scale_free.datasets import paper_imagenet as data
from dvd_scale_free.datasets.dataset_loader import SupervisedLearningDataset
from dvd_scale_free.models import paper_runner as paper
from scripts import main


ROOT = Path(__file__).resolve().parents[1]


def cli(*args):
    return subprocess.run([sys.executable, "-m", "scripts.main", *args], cwd=ROOT,
                          capture_output=True, text=True)


def paper_args(*flags):
    args = main.parse_args(["--training-recipe", "p004-paper", *flags])
    args.apply_color = args.grayscale_aug = 1
    dvd = main.build_dvd_transformer(args)
    args.dvd_protocol = main.describe_dvd(args, dvd)
    args.dvd_protocol["training_recipe"] = paper.describe(args, dvd)
    return args


def distributed_check(rank, root):
    torch.set_num_threads(1)
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0" if sys.platform == "darwin" else "lo")
    device = torch.device("cpu")
    folder = Path(root)
    dist.init_process_group("gloo", init_method=(folder / "rendezvous").as_uri(), rank=rank, world_size=2,
                            timeout=timedelta(seconds=40))
    try:
        criterion = nn.CrossEntropyLoss(weight=torch.tensor([1., 2., 3., 4., 5., 6.]))
        # Rank 0 is perfect. Rank 1 has more samples and misses all of them.
        logits = torch.tensor([[8., 0., 1., 2., 3., 4.]]).repeat(rank+1, 1)
        labels = torch.full((rank+1,), 0 if rank == 0 else 1, dtype=torch.long)
        report = paper.validate([(logits, labels)], nn.Identity(), criterion, device, expected_count=3)
        assert report["samples"] == 3 and report["top1"] == 100/3 and report["top5"] == 100/3
        combined = criterion(logits[:1].repeat(3, 1), torch.tensor([0, 1, 1])).item()
        assert abs(report["loss"] - combined) < 2e-6
        args = paper_args()
        args.rank = rank
        model = nn.parallel.DistributedDataParallel(nn.Linear(4, 6))
        optimizer = paper.paper_optimizer(model.parameters())
        scaler = torch.amp.GradScaler("cpu", enabled=False)
        x, y = torch.full((2, 4), 0.3), torch.tensor([0, 1])
        before = model.module.weight.detach().clone()
        paper.update_batch(model, optimizer, scaler, criterion, x, y)
        assert not torch.equal(model.module.weight, before)
        torch.manual_seed(100 + rank)
        np.random.seed(200 + rank)
        ages, manifest = [0., 1.], {"train": {"ordered_ids_sha256": "train"}}
        last = folder / "checkpoint_last.pth"
        paper._save_run_state(last, args, model, optimizer, scaler, device, manifest, ages, 0, report["top1"])
        dist.barrier()
        expected_next = torch.rand(3)
        state = torch.load(last, map_location="cpu", mmap=True, weights_only=False)
        paper._check_checkpoint(state, args, manifest, ages)
        assert not torch.equal(state["rank_rng"][0]["cpu"], state["rank_rng"][1]["cpu"])
        paper._restore_rng(state["rank_rng"][rank], device)
        torch.testing.assert_close(torch.rand(3), expected_next, rtol=0, atol=0)
        if rank == 0:
            alias = folder / "checkpoint_best.pth"
            paper._alias_atomic(last, alias)
            assert alias.stat().st_ino == last.stat().st_ino
            assert state["best_acc1"] == report["top1"]
    finally:
        dist.destroy_process_group()


class PaperRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_paper_cli_resolves_native_recipe_and_refuses_disabled_contrast(self):
        args = ("--training-recipe", "p004-paper", "--describe-dvd")
        result = cli(*args)
        self.assertEqual(result.returncode, 0, result.stderr)
        protocol = json.loads(result.stdout)
        self.assertTrue(protocol["config"]["apply_contrast"])
        self.assertFalse(protocol["config"]["unfiltered_at_300"])
        self.assertEqual(protocol["epochs"], 150)
        recipe = protocol["training_recipe"]
        self.assertEqual(recipe["optimizer"], dict(name="adam", lr=1e-4, weight_decay=1e-6, fused=True))
        self.assertEqual(recipe["validation"], "unfiltered-eager-global-50000")
        for bad in (("--apply_contrast", "0"), ("--rrc",), ("--epochs", "300"), ("--wd", "0")):
            result = cli(*args, *bad)
            self.assertEqual(result.returncode, 2)
            self.assertIn("p004-paper requires", result.stderr)

    def test_paper_batch_prefix_and_random_draws(self):
        args = paper_args()
        resize, reference, batch = data.paper_transforms(args)
        image = Image.fromarray(np.random.default_rng(9).integers(0, 256, (83, 121, 3), dtype=np.uint8))
        torch.testing.assert_close(resize(image), reference(image), rtol=0, atol=0)
        source = SupervisedLearningDataset(".", args).get_supervised_pipeline_transform(train=True)
        native = nn.Sequential(*list(source.aug_pipe.children())[2:])
        for seed in (11, 391):
            torch.manual_seed(seed)
            x = resize(image).expand(4, -1, -1, -1).contiguous()
            reference_result = native(x)
            reference_rng = torch.get_rng_state()
            torch.manual_seed(seed)
            torch.testing.assert_close(batch(x), reference_result, rtol=0, atol=0)
            self.assertTrue(torch.equal(torch.get_rng_state(), reference_rng))

    def test_imagefolder_manifest_and_sampler_protocol(self):
        args = paper_args("--workers", "0")
        args.rank, args.batch_size_per_gpu = 1, 2
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for split in ("train", "val"):
                for cls in ("b", "a"):
                    folder = root / split / cls
                    folder.mkdir(parents=True)
                    for i in range(3):
                        Image.new("RGB", (65+i, 87), color=(44+30*i, 156, 114)).save(folder / f"{i}.jpeg")
            with patch.multiple(data, TRAIN_IMAGES=6, VAL_IMAGES=6, N_CLASSES=2):
                train, val, _, first = data.paper_loaders(args, root)
                self.assertEqual(list(val.sampler), [1, 3, 5])
                self.assertEqual(train.generator.initial_seed(), 2)
                self.assertEqual(val.generator.initial_seed(), 3)
                self.assertTrue(train.drop_last)
                self.assertFalse(val.drop_last or val.persistent_workers)
                self.assertEqual(val.num_workers, 8)
                self.assertEqual(train.dataset.class_to_idx, {"a": 0, "b": 1})
                x, y = next(iter(train))
                self.assertEqual((tuple(x.shape), y.dtype), ((2, 3, 256, 256), torch.int64))
                Image.new("RGB", (65, 87), color=(0, 0, 0)).save(root / "val/a/0.jpeg")
                _, _, _, second = data.paper_loaders(args, root)
                self.assertEqual(first["val"]["ordered_ids_sha256"], second["val"]["ordered_ids_sha256"])
                self.assertNotEqual(first["val"]["jpeg_spotcheck_sha256"], second["val"]["jpeg_spotcheck_sha256"])

    def test_native_checkpoint_guard(self):
        args = paper_args()
        state = dict(args={**vars(args), "apply_contrast": 0}, training_variant="linear_v1", epoch=147,
                     age_months_curve=[0., 1.], rank_rng=[{}, {}], scaler={},
                     optimizer={"param_groups": [{"lr": 1e-4, "fused": True, "weight_decay": 1e-6}]})
        paper._check_checkpoint(state, args, {}, [0., 1.])
        with self.assertRaisesRegex(ValueError, "native P004"):
            paper._check_checkpoint({**state, "training_variant": "linear_v05"}, args, {}, [0., 1.])
        with self.assertRaisesRegex(ValueError, "age schedule"):
            paper._check_checkpoint(state, args, {}, [0., 2.])
        with self.assertRaisesRegex(ValueError, "both ranks"):
            paper._check_checkpoint({**state, "rank_rng": []}, args, {}, [0., 1.])
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            paper._check_checkpoint({**state, "dvd_protocol": args.dvd_protocol}, args, {"train": "new"}, [0., 1.])

    def test_small_slurm_memory_request_fails_before_loading(self):
        limits = ({"SLURM_MEM_PER_NODE": "24576"}, {"SLURM_MEM_PER_CPU": "4096", "SLURM_CPUS_ON_NODE": "6"})
        for limit in limits:
            with patch.dict(os.environ, {"SLURM_JOB_ID": "1", **limit}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "Slurm grants 24.0 GiB RAM"):
                    paper._headroom()

    def test_two_ranks_reduce_validation_and_restore_rng(self):
        if not dist.is_gloo_available():
            self.skipTest("PyTorch was built without Gloo")
        with TemporaryDirectory() as directory:
            spawn(distributed_check, (directory,), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
