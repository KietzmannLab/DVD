"""Optional RGB augmentation policies."""

import kornia.augmentation as K
from torch import nn

from ..dvd_scale_free.helpers.rrc import RRCConfig, training_rrc


def recipe_b_rgb_augmentation(image_size=256, *, train=True):
    """Build the large-crop RGB augmentation used before DVD in Recipe B."""
    if not train:
        return nn.Identity()
    crop = training_rrc(image_size, RRCConfig(enabled=True, scale=(0.7, 1.0)))
    return nn.Sequential(
        K.AugmentationSequential(
            crop, K.RandomHorizontalFlip(p=0.25), K.RandomRotation(degrees=15.0, p=0.25),
            data_keys=["image"],
        ),
        K.AugmentationSequential(K.RandomPerspective(distortion_scale=0.5, p=0.5), data_keys=["image"]),
        K.AugmentationSequential(
            K.RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0), p=0.5), data_keys=["image"],
        ),
    )
