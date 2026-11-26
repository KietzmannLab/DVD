"""Opt-in random resized crop for training, independent of the DVD curriculum."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class RRCConfig:
    enabled: bool = False
    scale: tuple[float, float] = (.08, 1.)
    ratio: tuple[float, float] = (.75, 4/3)

    def __post_init__(self):
        if len(self.scale) != 2 or not 0 < self.scale[0] <= self.scale[1] <= 1:
            raise ValueError("RRC area fractions must satisfy 0 < min <= max <= 1")
        if len(self.ratio) != 2 or not 0 < self.ratio[0] <= self.ratio[1] < math.inf:
            raise ValueError("RRC aspect ratios must be finite, positive, and ordered")


def training_rrc(size, config: RRCConfig | None = None, *, train=True):
    """Return a Kornia crop, or None when off or evaluating."""
    if not (train and config is not None and config.enabled):
        return None
    from kornia.augmentation import RandomResizedCrop

    size = (size, size) if isinstance(size, int) else size
    return RandomResizedCrop(size, scale=config.scale, ratio=config.ratio, p=1., same_on_batch=False)


def rrc_from_args(args):
    base = RRCConfig()
    legacy_min = getattr(args, "crop_min", 1.)
    return RRCConfig(
        enabled=getattr(args, "rrc", False),
        scale=tuple(getattr(args, "rrc_scale", None) or ((legacy_min, 1.) if legacy_min < 1 else base.scale)),
        ratio=tuple(getattr(args, "rrc_ratio", None) or base.ratio),
    )
