"""Training integration, the legacy opt-in, and the single-image command."""

import json
import logging
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from PIL import Image
import torch
from torch import nn

from dvd_scale_free import utils
from dvd_scale_free.datasets.dataset_loader import SupervisedLearningDataset
from dvd_scale_free.dvd_scale_free.development import DVDConfig, DVDTransformer
from dvd_scale_free.models.loader import _load_resume_checkpoint, _same_dvd_protocol
from scripts import main


ROOT = Path(__file__).resolve().parents[1]


def command(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT, check=True, text=True, capture_output=True)


class DVDCommandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_describe_uses_new_default_only_for_dvd(self):
        bare = command("-m", "scripts.main", "--describe-dvd")
        self.assertEqual(json.loads(bare.stdout), {"method": "adult"})
        latest = command("-m", "scripts.main", "--describe-dvd", "--development_strategy", "dvd")
        protocol = json.loads(latest.stdout)
        self.assertEqual(protocol["method"], "endpoint-linear")
        self.assertEqual(protocol["config"]["speed"], 1)
        self.assertTrue(protocol["config"]["apply_contrast"])
        self.assertEqual(protocol["config"]["speed_components"], ["acuity", "contrast", "color"])
        self.assertIsNone(protocol["config"]["adult_anchor_cpd"])
        self.assertFalse(protocol["config"]["unfiltered_at_300"])

    def test_cli_speed_components_and_v1_checkpoint_compatibility(self):
        def describe(flags):
            args = main.parser.parse_args(["--development_strategy", "dvd", *flags])
            return main.describe_dvd(args, main.build_dvd_transformer(args))

        def old(protocol):
            return dict(protocol, version=1, config={k: v for k, v in protocol["config"].items()
                                                     if k != "speed_components"})

        default = describe([])
        self.assertTrue(_same_dvd_protocol(old(default), default))
        speed = ["--endpoint-speed", ".5"]
        selective = describe([*speed, "--endpoint-speed-components", "contrast"])
        self.assertEqual(selective["config"]["speed_components"], ("contrast",))
        self.assertTrue(_same_dvd_protocol(old(selective), selective))
        advanced = describe(speed)
        self.assertEqual(advanced["acuity_clock"], "age-odds")
        self.assertFalse(_same_dvd_protocol(old(selective), advanced))
        earlier_v2 = {k: v for k, v in advanced.items() if k != "acuity_clock"}
        self.assertFalse(_same_dvd_protocol(earlier_v2, advanced))
        flipped = describe([*speed, "--invert-frequency", "--invert-amplitude"])
        self.assertTrue(_same_dvd_protocol(old(flipped), flipped))
        flipped_color = describe([*speed, "--invert-frequency", "--apply_color", "1"])
        self.assertFalse(_same_dvd_protocol(old(flipped_color), flipped_color))

    def test_legacy_cli_reproduces_original_api_with_matching_flags(self):
        args = main.parser.parse_args(["--development_strategy", "dvd", "--dvd-method", "legacy",
                                       "--cs_logspan_start", ".02", "--contrast_use_barten_csf", "0",
                                       "--image-size", "64"])
        legacy = main.build_dvd_transformer(args)
        original = DVDTransformer(DVDConfig(image_size=64, cs_logspan_start=.02))
        x = torch.rand(2, 3, 64, 64)
        torch.testing.assert_close(legacy(x, 16), original(x, 16), rtol=0, atol=0)

    def test_image_command_preserves_size_and_records_anchor(self):
        with TemporaryDirectory() as directory:
            src, dst = (Path(directory) / name for name in ("in.png", "out.png"))
            image = Image.new("RGB", (96, 64), color=(128, 112, 94))
            image.save(src)
            r = command("-m", "scripts.dvd_image", str(src), str(dst), "--months", "300",
                        "--adult-anchor-cpd", "auto", "--unfiltered-at-300", "--speed", "2",
                        "--speed-components", "color")
            meta = json.loads(r.stdout)
            self.assertTrue(meta["unfiltered"])
            self.assertTrue(meta["adult_anchor_clamped"])
            self.assertEqual(meta["speed_components"], ["color"])
            with Image.open(dst) as actual:
                self.assertEqual(actual.size, image.size)
                self.assertEqual(actual.tobytes(), image.tobytes())

    def test_imagenet_path_uses_supplied_splits(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "example-imagenet"
            for split in ("train", "val"):
                category = path / split / "sample-category"
                category.mkdir(parents=True)
                Image.new("RGB", (96, 64), color=(128, 112, 94)).save(category / "sample.png")
            args = main.parser.parse_args([directory, "--dataset-name", "imagenet",
                                           "--imagenet-path", str(path), "--workers", "0"])
            ds = SupervisedLearningDataset(directory, args).get_dataset("imagenet")
            self.assertEqual(Path(ds["train"].root), path / "train")
            image, target = ds["val"][0]
            self.assertEqual((image.shape, target), (torch.Size([3, 256, 256]), 0))

    def test_training_steps_and_checkpoint_protocol(self):
        args = main.parser.parse_args(["unused", "--development_strategy", "dvd", "--epochs", "150"])
        args.gpu = None
        dvdt = main.build_dvd_transformer(args)
        args.dvd_protocol = main.describe_dvd(args, dvdt)
        ages = main.build_age_months_curve(args, 1251)
        self.assertEqual((ages[0], ages[1251], len(ages)), (0, 2, 150 * 1251))
        self.assertAlmostEqual(ages[-1], 300 - 2 / 1251)

        x = torch.rand(2, 3, 224, 224)
        model = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 6))
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        old_weights = model[-1].weight.detach().clone()
        main.train([(x, torch.tensor([0, 1]))], model, nn.CrossEntropyLoss(), opt, None, None,
                   logging.getLogger(__name__), 0, ages, args, dvdt=dvdt)
        self.assertFalse(torch.equal(model[-1].weight.detach(), old_weights))
        with torch.no_grad():
            self.assertTrue(torch.isfinite(model(x)).all())  # An unfiltered validation batch.

        with TemporaryDirectory() as directory:
            path = Path(directory) / "weights" / "checkpoint_last.pth"
            utils.save_last_checkpoint(
                {"state_dict": model.state_dict(), "dvd_protocol": args.dvd_protocol}, str(path))
            result = _load_resume_checkpoint(path, args)
            torch.testing.assert_close(result["state_dict"]["2.weight"], model[-1].weight)
            args.dvd_protocol = {**args.dvd_protocol, "time_order": "random"}
            with self.assertRaisesRegex(ValueError, "DVD protocol differs"):
                _load_resume_checkpoint(path, args)


if __name__ == "__main__":
    unittest.main()
