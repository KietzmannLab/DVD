"""Checks for the optional Recipe B RGB image policy."""

import unittest

import torch

from dvd_scale_free.datasets.augmentations import recipe_b_rgb_augmentation


class RecipeBAugmentationTests(unittest.TestCase):
    def test_large_crop_policy_and_validation(self):
        augment = recipe_b_rgb_augmentation(64)
        ops = [op for group in augment for op in group.children()]
        self.assertEqual([(type(op).__name__, op.p) for op in ops], [
            ("RandomResizedCrop", 1.0), ("RandomHorizontalFlip", 0.25), ("RandomRotation", 0.25),
            ("RandomPerspective", 0.5), ("RandomGaussianBlur", 0.5),
        ])
        self.assertTrue(all(not op.same_on_batch for op in ops))
        self.assertEqual(tuple(ops[0]._param_generator.scale), (0.7, 1.0))
        self.assertEqual(tuple(ops[0]._param_generator.ratio), (0.75, 4 / 3))
        x = torch.rand((4, 3, 64, 64), generator=torch.Generator().manual_seed(19))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(12)
            result = augment(x)
        self.assertEqual(result.shape, x.shape)
        self.assertTrue(torch.isfinite(result).all())
        self.assertFalse(torch.equal(result, x))
        self.assertIs(recipe_b_rgb_augmentation(64, train=False)(x), x)
