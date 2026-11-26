"""Endpoint Linear DVD and its optional training controls."""

from .endpoint_linear import EndpointConfig, EndpointLinearDVD
from .endpoint_linear_fast import EndpointLinearTrainingTransform
from .inverse_endpoint_linear import InverseConfig, InverseEndpointDVD


def make_curriculum(*, invert_frequency=False, invert_amplitude=False, **options):
    """Neither flip: original Endpoint Linear. Both flips: full-colour inverse-DVD."""
    if invert_frequency or invert_amplitude:
        return InverseEndpointDVD(InverseConfig(
            invert_frequency=invert_frequency, invert_amplitude=invert_amplitude, **options))
    return EndpointLinearTrainingTransform(EndpointConfig(**options))
