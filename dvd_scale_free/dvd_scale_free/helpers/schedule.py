"""Age schedules shared by the training recipes."""

import random


def generate_age_months_curve(
    total_epochs: int,
    len_train_loader: int,
    months_per_epoch: float,
    shuffle: bool = False,
    seed: int | None = None,
    mid_phase: bool = False,
) -> list[float]:
    """Generate one age in months per training batch."""
    curve = [
        ep * months_per_epoch + b * months_per_epoch / len_train_loader
        for ep in range(total_epochs)
        for b in range(len_train_loader)
    ]

    if mid_phase:
        first = sorted(curve[::2], reverse=True)
        second = curve[1::2]
        curve = first + second

    if shuffle:
        if seed is not None:
            random.seed(seed)
        random.shuffle(curve)

    return curve
