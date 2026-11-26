"""Reflected-spectrum Endpoint Linear controls, including the full-colour inverse."""

from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
import math

import torch
from torch import nn

from .endpoint_linear import _age, _luminance, contrast_sensitivity, paper_color_weight


REFERENCE_BAND_CPD = .98 * 256 / 30


@dataclass(frozen=True)
class InverseConfig:
    invert_frequency: bool = True
    invert_amplitude: bool = True
    speed: float = 1.0  # V > 1 releases the spectrum sooner; V < 1 releases it later.
    fov_deg: float = 15.
    vertical_fov_deg: float | None = None
    birth_frequency_cpd: float = 2.
    amplitude_quantile: float = .5
    taper: float = .2
    # None uses the native common band. The default holds the 256-px calibration
    # fixed when transferring to higher resolutions, without exceeding Nyquist.
    band_limit_cpd: float | None = REFERENCE_BAND_CPD
    apply_frequency: bool = True
    apply_amplitude: bool = True
    apply_color: bool = False
    chunk_size: int = 32

    def __post_init__(self):
        if not (self.invert_frequency or self.invert_amplitude):
            raise ValueError("Use EndpointLinearDVD for the original forward curriculum")
        for name in ("speed", "fov_deg", "vertical_fov_deg", "birth_frequency_cpd", "band_limit_cpd"):
            value = getattr(self, name)
            if value is None and name in ("vertical_fov_deg", "band_limit_cpd"):
                continue
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.amplitude_quantile < 1 or not 0 < self.taper < 1:
            raise ValueError("Amplitude quantile and taper must be strictly between zero and one")
        if self.band_limit_cpd is not None and self.birth_frequency_cpd >= self.band_limit_cpd:
            raise ValueError("Birth frequency must be below the calibration band limit")
        if type(self.chunk_size) is not int or self.chunk_size < 1:
            raise ValueError("chunk_size must be a positive integer")


def _smooth(z):
    z = z.clamp(0, 1)
    return z.square() * (3 - 2*z)


def _reflect(x):
    x = torch.cat((x, x.flip(-1)), -1)
    return torch.cat((x, x.flip(-2)), -2)


@lru_cache(maxsize=6)
def _basis(h, w, fov_x, fov_y, start, ceiling, device, dtype):
    fy = torch.fft.fftfreq(2*h, d=fov_y/h, device=device, dtype=dtype)
    fx = torch.fft.rfftfreq(2*w, d=fov_x/w, device=device, dtype=dtype)
    freq = torch.hypot(fy[:, None], fx[None, :])
    in_band = (freq >= start) & (freq <= ceiling)
    extra = in_band.clone()
    extra[:, (0, -1)] = False
    # Count the omitted conjugates to get the full-spectrum training quantile.
    indices = torch.cat((in_band.flatten().nonzero().flatten(), extra.flatten().nonzero().flatten()))
    return freq, indices


def _quantile(x, q):
    rank = (x.shape[1]-1)*q
    i, j = math.floor(rank), math.ceil(rank)
    lo = x.kthvalue(i+1, dim=1).values
    if i == j:
        return lo
    hi = x.kthvalue(j+1, dim=1).values
    return (lo+hi)*.5 if j-rank == .5 else torch.lerp(lo, hi, rank-i)


class InverseEndpointDVD(nn.Module):
    """CHW/BCHW floats in [0, 1]; output keeps its size, with identity at 300 months."""

    def __init__(self, config: InverseConfig | None = None):
        super().__init__()
        self.config = InverseConfig() if config is None else config

    @torch.no_grad()
    def forward(self, image, months, *, return_diagnostics=False):
        age = _age(months)
        if not isinstance(image, torch.Tensor) or not image.is_floating_point():
            raise TypeError("Expected a floating-point tensor before model normalization")
        squeeze = image.ndim == 3
        x = image.unsqueeze(0) if squeeze else image
        if x.ndim != 4 or x.shape[0] < 1 or x.shape[1] not in (1, 3, 4) or min(x.shape[-2:]) < 16:
            raise ValueError("Expected nonempty CHW/BCHW with 1/3/4 channels and H,W >= 16")

        cfg = self.config
        h, w = x.shape[-2:]
        fov_x = cfg.fov_deg
        fov_y = cfg.vertical_fov_deg if cfg.vertical_fov_deg is not None else fov_x*h/w
        nyq = .98 * min(w/(2*fov_x), h/(2*fov_y))
        ceiling = nyq if cfg.band_limit_cpd is None else min(nyq, cfg.band_limit_cpd)
        p = contrast_sensitivity(age)**(1/cfg.speed)
        open_freq = math.sqrt(2)*ceiling/(.98*(1-cfg.taper))
        f = (cfg.birth_frequency_cpd*(1-p) if cfg.invert_frequency
             else (1-p)*cfg.birth_frequency_cpd + p*open_freq)
        diagnostics = dict(age_months=age, contrast_progress=p, band_limit_cpd=ceiling,
                           frequency_boundary_cpd=f, frequency_floor=cfg.invert_frequency,
                           amplitude_ceiling=cfg.invert_amplitude, unfiltered=age >= 300)
        if age >= 300:
            return (image, [dict(diagnostics, amplitude_boundary=None) for _ in x]) if return_diagnostics else image
        if cfg.apply_amplitude and cfg.birth_frequency_cpd >= ceiling:
            raise ValueError(f"Birth frequency must be below the sampled calibration band ({ceiling:g} cpd)")

        dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
        freq, indices = _basis(h, w, fov_x, fov_y, cfg.birth_frequency_cpd, ceiling, x.device, dtype)
        if cfg.apply_amplitude and not indices.numel():
            raise ValueError("The sampled calibration band is empty")
        fine = torch.ones_like(freq)
        if cfg.apply_frequency:
            if cfg.invert_frequency:
                if f > 0:
                    fine = _smooth((freq/f - 1)/cfg.taper)
            else:
                fine = _smooth((1 - freq/f)/cfg.taper)
        rows = []
        y = torch.empty_like(x)
        ctx = torch.autocast(x.device.type, enabled=False) if x.device.type in ("cpu", "cuda") else nullcontext()
        with ctx:
            tiny = torch.finfo(dtype).tiny
            for raw, out in zip(x.split(cfg.chunk_size), y.split(cfg.chunk_size)):
                data = raw[:, :3].to(dtype)
                coeff = torch.fft.rfft2(_reflect(data))
                boundary = birth = None
                if cfg.apply_amplitude:
                    lum = _luminance(coeff).squeeze(1)
                    dc = lum[:, 0, 0].real.clamp_min(tiny*(h*w*4))
                    amplitude = 2*(lum.abs()/dc[:, None, None])
                    amplitude[:, 0, 0] = 0
                    peak = amplitude.amax((-2, -1))
                    birth = _quantile(amplitude.flatten(1).index_select(1, indices), cfg.amplitude_quantile)
                    birth = torch.maximum(birth, peak*torch.finfo(dtype).eps).clamp_min(tiny)
                    if cfg.invert_amplitude:
                        boundary = (1-p)*birth + p*torch.maximum(birth, peak/(1-cfg.taper))
                        gate = _smooth((1-amplitude/boundary[:, None, None])/cfg.taper)
                    else:
                        knee = (cfg.taper*birth).clamp_min(tiny)
                        boundary = birth - p*(birth+knee)
                        gate = _smooth((amplitude-boundary[:, None, None])/knee[:, None, None])
                    gain = fine*gate
                else:
                    gain = fine.expand(len(data), -1, -1).clone()
                gain[:, 0, 0] = 1
                restored = torch.fft.irfft2(coeff.mul_(gain[:, None]), s=(2*h, 2*w))[..., :h, :w]
                if cfg.apply_color and data.shape[1] == 3:
                    alpha = min(1., paper_color_weight(age))
                    restored = (1-alpha)*_luminance(restored) + alpha*restored
                out[:, :3].copy_(restored.clamp_(0, 1))
                if raw.shape[1] == 4:
                    out[:, 3:].copy_(raw[:, 3:])
                if return_diagnostics:
                    bounds = [None]*len(raw) if boundary is None else boundary.tolist()
                    anchors = [None]*len(raw) if birth is None else birth.tolist()
                    rows.extend(dict(diagnostics, amplitude_boundary=b, birth_amplitude=a)
                                for b, a in zip(bounds, anchors))
        y = y[0] if squeeze else y
        return (y, rows) if return_diagnostics else y
