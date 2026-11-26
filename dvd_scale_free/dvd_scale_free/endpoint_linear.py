"""Endpoint Linear DVD reference transform; PyTorch is the only direct dependency.

The default is the V=1 training transform, not the browser's 300-month bypass.
See README.md for the input contract, geometry, and examples.
"""

from __future__ import annotations

from bisect import bisect_left
from contextlib import nullcontext
from dataclasses import dataclass
from functools import lru_cache
import math
from typing import Literal, NamedTuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F


ADULT_AGE_MONTHS = 300.0
ADULT_NYQUIST_FRACTION = 0.98
REFERENCE_ADULT_CPD = ADULT_NYQUIST_FRACTION * 224 / (2 * 15)
AMPLITUDE_FLOOR = 1e-16
SPEED_COMPONENTS = ("acuity", "contrast", "color")
SpeedComponent = Literal["acuity", "contrast", "color"]
_CONTRAST_MIDPOINT = 4.8 * 12.0
_CONTRAST_EXPONENT = 2.1633375920569247


@dataclass(frozen=True)
class EndpointConfig:
    """Angles are in degrees, spatial-frequency anchors in cycles/degree."""

    speed: float = 1.0  # Larger values advance selected curves; smaller values delay them.
    fov_deg: float = 15.0
    vertical_fov_deg: float | None = None
    infant_anchor_cpd: float | Literal["fundamental"] = 0.23
    adult_anchor_cpd: float | Literal["nyquist"] | None = None
    sigmoid_width_log10: float = 0.10
    image_size: int | tuple[int, int] | None = None
    unfiltered_at_300: bool = False
    apply_acuity: bool = True
    apply_contrast: bool = True
    apply_color: bool = True
    clamp_output: bool = True
    low_frequency_policy: Literal["extrapolate", "clamp"] = "extrapolate"
    adult_out_of_band: Literal["clamp", "error"] = "clamp"
    speed_components: tuple[SpeedComponent, ...] = SPEED_COMPONENTS

    def __post_init__(self):
        if not math.isfinite(self.speed) or self.speed <= 0:
            raise ValueError("speed (V) must be finite and positive")
        object.__setattr__(self, "speed_components", _speed_components(self.speed_components))
        for value, name in (
            (self.fov_deg, "fov_deg"), (self.vertical_fov_deg, "vertical_fov_deg"),
            (self.infant_anchor_cpd, "infant_anchor_cpd"),
            (self.adult_anchor_cpd, "adult_anchor_cpd"),
            (self.sigmoid_width_log10, "sigmoid_width_log10"),
        ):
            if (name, value) in (
                ("vertical_fov_deg", None), ("infant_anchor_cpd", "fundamental"),
                ("adult_anchor_cpd", None), ("adult_anchor_cpd", "nyquist"),
            ):
                continue
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number or a documented preset")
        if self.low_frequency_policy not in ("extrapolate", "clamp"):
            raise ValueError("low_frequency_policy must be 'extrapolate' or 'clamp'")
        if self.adult_out_of_band not in ("clamp", "error"):
            raise ValueError("adult_out_of_band must be 'clamp' or 'error'")
        size = self.image_size
        if size is not None:
            dims = (size,) if isinstance(size, int) else size
            if not isinstance(dims, tuple) or len(dims) != (1 if isinstance(size, int) else 2):
                raise ValueError("image_size must be None, a long-edge int, or (height, width)")
            if not all(isinstance(n, int) and n >= 16 for n in dims):
                raise ValueError("image_size dimensions must be >= 16")


def _age(months: float) -> float:
    value = float(months)
    if not math.isfinite(value) or value < 0:
        raise ValueError("months must be finite and nonnegative")
    return value


def _contrast_hill(t):
    if t < 1e-100:
        power = (t / _CONTRAST_MIDPOINT) ** _CONTRAST_EXPONENT
        return power / (1.0 + power)
    return 1.0 / (1.0 + (_CONTRAST_MIDPOINT / t) ** _CONTRAST_EXPONENT)


_ADULT_CONTRAST = _contrast_hill(ADULT_AGE_MONTHS)


def contrast_sensitivity(months: float) -> float:
    """The original DVD Hill fit, normalized and capped at 300 months."""
    return _contrast_hill(min(_age(months), ADULT_AGE_MONTHS)) / _ADULT_CONTRAST


def _speed_components(values):
    if (not isinstance(values, (list, tuple)) or any(not isinstance(v, str) or v not in SPEED_COMPONENTS for v in values)
            or len(set(values)) != len(values)):
        raise ValueError("speed_components must be a list or tuple of distinct acuity, contrast, or color values")
    return tuple(v for v in SPEED_COMPONENTS if v in values)


class _Timing(NamedTuple):
    acuity_months: float
    color_months: float
    frequency_progress: float
    contrast_progress: float


def _stage_timing(months, speed, components):
    base = contrast_sensitivity(months)
    progress = base ** (1 / speed) if speed != 1 and components else base
    acuity_age = color_age = months
    if speed != 1 and 0 < months < ADULT_AGE_MONTHS:
        if "acuity" in components:
            # Scale the odds of elapsed age; never use the contrast fit for Gaussian acuity.
            age_fraction = months / ADULT_AGE_MONTHS
            advanced = speed * age_fraction
            acuity_age = ADULT_AGE_MONTHS * (advanced / (1 - age_fraction + advanced))
        if "color" in components:
            q = progress * _ADULT_CONTRAST
            color_age = min(ADULT_AGE_MONTHS, _CONTRAST_MIDPOINT * (q / (1 - q)) ** (1 / _CONTRAST_EXPONENT))
    # Flipped modes retain their original Hill-driven spectral gate, not the Gaussian acuity curve.
    return _Timing(acuity_age, color_age, progress if "acuity" in components else base,
                   progress if "contrast" in components else base)


def acuity_sigma_degrees(months: float) -> float:
    t = _age(months)
    sigma_224 = (18.035945052640425 * math.exp(-0.7933899743217134 * t)
                 + 1.6012490927401029 * math.exp(-0.027054604551482078 * t))
    return sigma_224 * 15 / 224


def paper_color_weight(months: float) -> float:
    """The raw paper fit, as used by Endpoint Linear training before final clipping."""
    years = _age(months) / 12
    if years == 0:
        return 0.0
    a, b, alpha = 0.008604133954779169, 4.380740053287391e-05, 0.8807610802743646

    def threshold(t):
        return a * t ** (-alpha) + b * t ** alpha

    return threshold((21.2 + 21.1 + 18.8) / 3) / threshold(years)


def _luminance(x: Tensor) -> Tensor:
    return x if x.shape[1] == 1 else .2989 * x[:, :1] + .5870 * x[:, 1:2] + .1140 * x[:, 2:3]


def _blur(x: Tensor, sigma: float, fov_x: float, fov_y: float) -> Tensor:
    if sigma == 0:
        return x

    def axis(y, fov):
        n = y.shape[-1]
        # The even extension implements reflection at the angular pixel edges.
        spectrum = torch.fft.rfft(torch.cat((y, y.flip(-1)), dim=-1), dim=-1)
        k = torch.arange(spectrum.shape[-1], device=y.device, dtype=y.dtype)
        gain = torch.exp(-2 * math.pi**2 * sigma**2 * (k / (2 * fov)).square())
        return torch.fft.irfft(spectrum * gain, n=2 * n, dim=-1)[..., :n]

    return axis(axis(x, fov_x).transpose(-1, -2), fov_y).transpose(-1, -2)


@lru_cache(maxsize=8)
def _grid(h, w, fov_x, fov_y, device):
    def modes(n, fov):
        k = torch.arange(n, dtype=torch.float64, device=device)
        return torch.where(k < (n + 1) // 2, k, k - n) / fov

    fy, fx = torch.meshgrid(modes(h, fov_y), modes(w, fov_x), indexing="ij")
    freq = torch.sqrt(fx.square() + fy.square()).flatten()
    width = math.hypot(1 / fov_x, 1 / fov_y)
    count = math.ceil(float(freq.max()) / width) + 1
    indices = (freq / width).floor().long().clamp_max(count - 1)
    non_dc = freq > 0
    counts = freq.new_zeros(count).scatter_add_(0, indices, non_dc.to(freq.dtype))
    centers = freq.new_zeros(count).scatter_add_(0, indices, freq) / counts.clamp_min(1)
    bands = sorted((f, i) for i, f in enumerate(centers.tolist()) if f > 0)
    center_cpd, band_ids = zip(*bands)
    return indices, non_dc, counts, center_cpd, torch.tensor(band_ids, device=device), width


def _anchor(frequencies, values, frequency, fundamental, low_policy):
    low, high = frequencies[0], frequencies[-1]
    first = max(values[0], AMPLITUDE_FLOOR)
    if frequency < fundamental * (1 - 1e-6):
        if low_policy == "clamp":
            return first, False
        measured = [(f, a) for f, a in zip(frequencies, values) if a > AMPLITUDE_FLOOR]
        fit = [(f, a) for f, a in measured if f <= max(.9, 4 * low)]
        if len(fit) < 3:
            fit = measured[:8]
        alpha = 0.0
        if len(fit) >= 3:
            x, y = zip(*((math.log10(f), math.log10(a)) for f, a in fit))
            xm, ym = sum(x) / len(x), sum(y) / len(y)
            cov = sum((i - xm) * (j - ym) for i, j in zip(x, y))
            var = sum((i - xm)**2 for i in x)
            alpha = min(2.5, max(0., -cov / max(var, 1e-12)))
        return max(10**(math.log10(first) + alpha * math.log10(low / frequency)), AMPLITUDE_FLOOR), True
    if frequency <= low:
        return first, False
    if frequency >= high:
        return max(values[-1], AMPLITUDE_FLOOR), False
    j = bisect_left(frequencies, frequency)
    left, right = map(math.log, frequencies[j-1:j+1])
    z = (math.log(frequency) - left) / (right - left)
    a, b = (math.log(max(x, AMPLITUDE_FLOOR)) for x in values[j-1:j+1])
    return max(math.exp((1-z)*a + z*b), AMPLITUDE_FLOOR), False


class EndpointLinearDVD(nn.Module):
    """Float CHW/BCHW (1/3/4 channels) or a list/tuple of those, before model normalization."""

    def __init__(self, config: EndpointConfig | None = None):
        super().__init__()
        self.config = config if config is not None else EndpointConfig()

    def _geometry(self, source_hw, size):
        cfg = self.config
        h, w = size
        sh, sw = source_hw
        fx, fy = cfg.fov_deg, cfg.vertical_fov_deg
        fy = fx * sh / sw if fy is None else fy
        fundamental = 1 / max(fx, fy)
        band = ADULT_NYQUIST_FRACTION * min(w / (2 * fx), h / (2 * fy))
        requested = cfg.adult_anchor_cpd
        adult_ref = REFERENCE_ADULT_CPD if requested is None else band if requested == "nyquist" else float(requested)
        adult = min(adult_ref, band)
        infant = fundamental if cfg.infant_anchor_cpd == "fundamental" else float(cfg.infant_anchor_cpd)
        if cfg.apply_contrast:
            if adult_ref > band + 1e-12 and cfg.adult_out_of_band == "error":
                raise ValueError(f"Adult anchor {adult_ref:g} cpd exceeds {band:g} cpd on this grid")
            if infant >= adult:
                raise ValueError(f"Infant anchor {infant:g} cpd must be below adult {adult:g} cpd")
        return dict(horizontal_fov_deg=fx, vertical_fov_deg=fy, lowest_sampled_cpd=fundamental,
                    adult_band_limit_cpd=band, requested_adult_anchor_cpd=adult_ref,
                    infant_anchor_cpd=infant, adult_anchor_cpd=adult,
                    adult_anchor_clamped=adult_ref > band + 1e-12, output_size=(h, w))

    def _contrast(self, x, geometry, progress):
        cfg = self.config
        batch, _, h, w = x.shape
        indices, non_dc, counts, centers, valid, width = _grid(
            h, w, geometry["horizontal_fov_deg"], geometry["vertical_fov_deg"], x.device)
        y = _luminance(x)
        mean = y.mean((-2, -1), keepdim=True)
        raw = torch.fft.fft2(y - mean)
        amp = (2*raw.abs() / (h*w*mean.clamp_min(1e-10))).reshape(batch, -1)
        radial = x.new_zeros((batch, counts.numel()))
        radial.scatter_add_(1, indices.expand(batch, -1), amp * non_dc.to(x.dtype))
        radial /= counts.clamp_min(1).to(x.dtype)
        gains = torch.ones_like(radial)
        diagnostics = []
        for i, amplitudes in enumerate(radial.index_select(1, valid).tolist()):
            if max(amplitudes) <= AMPLITUDE_FLOOR:
                diagnostics.append(dict(flat_image=True, contrast_threshold=None))
                continue
            infant, extrapolated = _anchor(centers, amplitudes, geometry["infant_anchor_cpd"],
                                           geometry["lowest_sampled_cpd"], cfg.low_frequency_policy)
            adult, _ = _anchor(centers, amplitudes, geometry["adult_anchor_cpd"],
                               geometry["lowest_sampled_cpd"], cfg.low_frequency_policy)
            threshold = (1-progress)*infant + progress*adult
            log_amplitudes = radial[i].clamp_min(AMPLITUDE_FLOOR).log10()
            gains[i] = ((log_amplitudes - math.log10(threshold)) / cfg.sigmoid_width_log10).sigmoid()
            diagnostics.append(dict(flat_image=False, infant_anchor_extrapolated=extrapolated,
                                    infant_amplitude=infant, adult_amplitude=adult,
                                    contrast_threshold=threshold, endpoint_order_warning=infant < adult,
                                    radial_bin_width_cpd=width))
        mask = gains[:, indices].reshape(batch, 1, h, w)
        mask[:, :, 0, 0] = 1.
        channel_means = x.mean((-2, -1), keepdim=True)
        return torch.fft.ifft2(torch.fft.fft2(x-channel_means) * mask).real + channel_means, diagnostics

    @torch.no_grad()
    def forward(self, image, months: float, *, return_diagnostics: bool = False):
        months = _age(months)
        if isinstance(image, (list, tuple)):
            results = [self(t, months, return_diagnostics=return_diagnostics) for t in image]
            if return_diagnostics:
                return type(image)(v for v, _ in results), type(image)(d for _, d in results)
            return type(image)(results)
        if not isinstance(image, Tensor) or image.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise TypeError("Expected a float PyTorch tensor in [0, 1], before normalization")
        one = image.ndim == 3
        x = image.unsqueeze(0) if one else image
        if x.ndim != 4 or x.shape[1] not in (1, 3, 4) or min(x.shape[-2:]) < 16 or x.shape[0] < 1:
            raise ValueError("Expected nonempty CHW/BCHW, 1/3/4 channels and H,W >= 16")
        ctx = torch.autocast(x.device.type, enabled=False) if x.device.type in ("cpu", "cuda") else nullcontext()
        with ctx:
            output, diagnostics = self._forward_batch(x, months)
        if one:
            output = output[0]
        return (output, diagnostics) if return_diagnostics else output

    def _forward_batch(self, x, months):
        cfg = self.config
        source_hw = tuple(x.shape[-2:])
        size = cfg.image_size
        if isinstance(size, int):
            size = tuple(round(v * size / max(source_hw)) for v in source_hw)
        if size is not None and min(size) < 16:
            raise ValueError("Resizing would make the short edge smaller than 16 pixels")
        output_hw = source_hw if size is None else size
        geometry = self._geometry(source_hw, output_hw)
        timing = _stage_timing(months, cfg.speed, cfg.speed_components)
        sigma = acuity_sigma_degrees(timing.acuity_months)
        has_color = cfg.apply_color and x.shape[1] in (3, 4)
        color_weight = paper_color_weight(timing.color_months) if has_color else 1.0
        meta = dict(geometry, age_months=months, speed=cfg.speed, speed_components=cfg.speed_components,
                    acuity_age_months=timing.acuity_months, color_age_months=timing.color_months,
                    contrast_speed=cfg.speed if "contrast" in cfg.speed_components else 1.0,
                    contrast_progress=timing.contrast_progress, gaussian_sigma_deg=sigma,
                    color_weight=color_weight, unfiltered=False)
        original_dtype = x.dtype
        if output_hw != source_hw:
            x = x.float() if original_dtype in (torch.float16, torch.bfloat16) else x
            x = F.interpolate(x, size=output_hw, mode="bilinear", align_corners=False, antialias=True)
        if cfg.unfiltered_at_300 and months >= ADULT_AGE_MONTHS:
            return x.to(original_dtype), [dict(meta, unfiltered=True, contrast_threshold=None) for _ in x]

        x = x.float() if original_dtype in (torch.float16, torch.bfloat16) else x
        alpha = x[:, 3:] if x.shape[1] == 4 else None
        x = x[:, :3]
        if cfg.apply_acuity:
            x = _blur(x, sigma, geometry["horizontal_fov_deg"], geometry["vertical_fov_deg"])
        if cfg.apply_contrast:
            x, diag = self._contrast(x, geometry, timing.contrast_progress)
        else:
            diag = [dict(contrast_threshold=None) for _ in x]
        if has_color:
            x = color_weight*x + (1-color_weight)*_luminance(x)
        if cfg.clamp_output:
            x = x.clamp(0, 1)
        if alpha is not None:
            x = torch.cat((x, alpha), 1)
        return x.to(original_dtype), [dict(meta, **row) for row in diag]
