"""P004-compatible batched Fourier path for square, default-geometry training input."""

from bisect import bisect_left
from dataclasses import dataclass
from functools import lru_cache
import math

import torch

from .development import (
    AMPLITUDE_FLOOR, DVDTransformer, _age, _blur, _grid, _luminance,
    _stage_timing, acuity_sigma_degrees, paper_color_weight,
)


@dataclass(frozen=True)
class _Basis:
    indices: torch.Tensor
    weights: torch.Tensor
    counts: torch.Tensor
    left: torch.Tensor
    right: torch.Tensor
    fractions: torch.Tensor


@lru_cache(maxsize=4)
def _basis(side, device):
    ids, _, counts, centers, band_ids, _ = _grid(side, side, 15., 15., device)
    populated = band_ids.tolist()
    indices = ids.reshape(side, side)[:, :side//2+1].reshape(-1)
    geometry = DVDTransformer()._geometry((side, side), (side, side))
    left, right, fractions = [], [], []
    for anchor in (geometry["infant_anchor_cpd"], geometry["adult_anchor_cpd"]):
        j = bisect_left(centers, anchor)
        if not 0 < j < len(centers):
            raise ValueError("Batched DVD requires two in-band anchors")
        lo, hi = math.log(centers[j-1]), math.log(centers[j])
        left.append(populated[j-1])
        right.append(populated[j])
        fractions.append((math.log(anchor) - lo) / (hi - lo))
    # Count conjugate pairs exactly as in the full FFT radial average.
    weights = torch.full((side, side//2+1), 2., dtype=torch.float32, device=device)
    weights[:, 0] = 1
    if side % 2 == 0:
        weights[:, -1] = 1
    weights[0, 0] = 0
    return _Basis(indices, weights.reshape(-1), counts.float().clamp_min(1),
                  torch.tensor(left, device=device), torch.tensor(right, device=device),
                  torch.tensor(fractions, dtype=torch.float32, device=device))


def supports_batched(config):
    return (config.fov_deg == 15. and config.vertical_fov_deg in (None, 15.)
            and config.infant_anchor_cpd == .23 and config.adult_anchor_cpd is None
            and config.sigmoid_width_log10 == .1 and config.image_size is None
            and config.apply_acuity and config.apply_contrast and not config.unfiltered_at_300)


@torch.no_grad()
def _filter_spatial(images, sigma, progress, chunk):
    side = images.shape[-1]
    basis = _basis(side, images.device)
    floor = AMPLITUDE_FLOOR
    out = torch.empty_like(images, memory_format=torch.contiguous_format)
    for start in range(0, len(images), chunk):
        x = _blur(images[start:start+chunk], sigma, 15., 15.)
        y = _luminance(x)
        y_mean = y.mean((-2,-1), keepdim=True)
        amp = torch.fft.rfft2(y-y_mean).abs().mul_(2/(side*side*y_mean.clamp_min(1e-10)))
        radial = x.new_zeros((len(x), basis.counts.numel()))
        radial.scatter_add_(1, basis.indices.expand(len(x), -1), amp.flatten(1)*basis.weights)
        radial.div_(basis.counts)
        log_amp = radial.clamp_min(floor).log()
        anchors = torch.lerp(log_amp.index_select(1,basis.left), log_amp.index_select(1,basis.right),
                             basis.fractions).exp().clamp_min_(floor)
        threshold = torch.lerp(anchors[:, :1], anchors[:, 1:], progress)
        gains = ((log_amp-threshold.log())/(.1*math.log(10))).sigmoid_()
        gains.masked_fill_(radial.amax(dim=1, keepdim=True) <= floor, 1)
        mask = gains.index_select(1, basis.indices).reshape(len(x), 1, side, side//2+1)
        mask[:, :, 0, 0] = 1
        mean = x.mean((-2,-1), keepdim=True)
        spectrum = torch.fft.rfft2(x-mean).mul_(mask)
        out[start:start+len(x)] = torch.fft.irfft2(spectrum, s=(side,side)).add_(mean)
    return out


class DVDTrainingTransform(DVDTransformer):
    """Auto-dispatch to the batched kernel where it matches the requested settings."""

    def __init__(self, config=None, *, backend="auto", chunk=128):
        super().__init__(config)
        if backend not in ("auto", "reference", "batched") or type(chunk) is not int or chunk < 1:
            raise ValueError("Use backend auto/reference/batched and a positive integer FFT chunk size")
        if backend == "batched" and not supports_batched(self.config):
            raise ValueError("Batched DVD requires the default physical geometry and both spatial stages")
        self.backend = backend
        self.chunk = chunk

    def backend_for(self, image):
        if (self.backend != "reference" and supports_batched(self.config)
                and isinstance(image, torch.Tensor) and image.ndim == 4 and image.shape[0] > 0
                and image.shape[1] in (1,3) and image.shape[2] == image.shape[3] >= 224
                and image.dtype == torch.float32 and image.device.type in ("cpu","cuda")):
            return "batched"
        if self.backend == "batched":
            raise ValueError("Batched DVD expects nonempty square float32 NCHW input at least 224 px on CPU/CUDA")
        return "reference"

    @torch.no_grad()
    def forward(self, image, months, *, return_diagnostics=False):
        if return_diagnostics or self.backend_for(image) == "reference":
            return super().forward(image, months, return_diagnostics=return_diagnostics)
        timing = _stage_timing(_age(months), self.config.speed, self.config.speed_components)
        with torch.autocast(image.device.type, enabled=False):
            out = _filter_spatial(image, acuity_sigma_degrees(timing.acuity_months), timing.contrast_progress, self.chunk)
            if self.config.apply_color and out.shape[1] == 3:
                a = paper_color_weight(timing.color_months)
                out = a*out + (1-a)*_luminance(out)
            return out.clamp_(0,1) if self.config.clamp_output else out
