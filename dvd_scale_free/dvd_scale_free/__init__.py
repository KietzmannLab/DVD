"""DVD and its optional training controls."""

from .development import DVDConfig, DVDTransformer
from .development_fast import DVDTrainingTransform
from .inverse import InverseConfig, InverseDVD


def make_curriculum(*, invert_frequency=False, invert_amplitude=False, **options):
    """Create forward DVD or select the inverted frequency and amplitude gates."""
    if invert_frequency or invert_amplitude:
        return InverseDVD(InverseConfig(
            invert_frequency=invert_frequency, invert_amplitude=invert_amplitude, **options))
    return DVDTrainingTransform(DVDConfig(**options))
