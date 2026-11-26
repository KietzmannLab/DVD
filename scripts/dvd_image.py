"""Apply Endpoint Linear DVD to one image and print the resolved filter settings."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps
import torch
from torch.nn.functional import interpolate

from dvd_scale_free.dvd_scale_free.inverse_endpoint_linear import REFERENCE_BAND_CPD, InverseConfig, InverseEndpointDVD
from dvd_scale_free.dvd_scale_free._cli import adult_anchor, infant_anchor
from dvd_scale_free.dvd_scale_free.endpoint_linear import SPEED_COMPONENTS, EndpointConfig, EndpointLinearDVD


def image_size(value):
    if value == "native":
        return None
    if "x" in value.lower():
        return tuple(map(int, value.lower().split("x")))
    return int(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path, help="destination PNG")
    parser.add_argument("--months", type=float, default=48, help="age in months (default: 48)")
    parser.add_argument("--speed", type=float, default=1.,
                        help="developmental speed V: >1 advances selected curves, <1 delays them (default: 1)")
    parser.add_argument("--speed-components", nargs="*", choices=SPEED_COMPONENTS, default=SPEED_COMPONENTS,
                        help="curves affected by V (default: all three; omit values to leave every curve at V=1)")
    parser.add_argument("--fov-deg", type=float, default=15., help="horizontal visual angle")
    parser.add_argument("--vertical-fov-deg", type=float, help="default: infer from source aspect ratio")
    parser.add_argument("--infant-anchor-cpd", type=infant_anchor, default=.23,
                        help="numeric cpd or 'fundamental'")
    parser.add_argument("--adult-anchor-cpd", type=adult_anchor, default=None,
                        help="'auto' (default), numeric cpd, or 'nyquist'")
    parser.add_argument("--sigmoid-width-log10", type=float, default=.10)
    parser.add_argument("--image-size", type=image_size,
                        help="'native' (default), long edge (e.g. 256), or HxW (e.g. 224x256)")
    parser.add_argument("--unfiltered-at-300", action="store_true",
                        help="demo only: return the prepared input at ages >=300 months")
    parser.add_argument("--invert-frequency", action="store_true")
    parser.add_argument("--invert-amplitude", action="store_true")
    parser.add_argument("--inverse-birth-cpd", type=float, default=2.)
    parser.add_argument("--inverse-quantile", type=float, default=.5)
    parser.add_argument("--inverse-taper", type=float, default=.2)
    parser.add_argument("--inverse-native-band", action="store_true")
    parser.add_argument("--dvd-color", action=argparse.BooleanOptionalAction,
                        help="age-dependent color (default: on for forward only)")
    parser.add_argument("--device", default="cpu", help="cpu or cuda[:index]")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".png" or args.output.resolve() == args.input.resolve():
        parser.error("Use a different file with a .png extension for output")
    try:
        with Image.open(args.input) as original:
            im = ImageOps.exif_transpose(original)
            mode = ("RGBA" if "A" in im.getbands() or "transparency" in im.info
                    else "L" if im.mode in ("1", "L") else "RGB")
            im = im.convert(mode)
            x = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
            x = x.reshape(im.height, im.width, len(mode)).permute(2, 0, 1)
        x = x.to(args.device, dtype=torch.float32).div_(255)
        flipped = args.invert_frequency or args.invert_amplitude
        color = not flipped if args.dvd_color is None else args.dvd_color
        common = dict(speed=args.speed, speed_components=args.speed_components,
                      fov_deg=args.fov_deg, vertical_fov_deg=args.vertical_fov_deg, apply_color=color)
        cfg = EndpointConfig(
            **common, infant_anchor_cpd=args.infant_anchor_cpd, adult_anchor_cpd=args.adult_anchor_cpd,
            sigmoid_width_log10=args.sigmoid_width_log10, image_size=args.image_size,
            unfiltered_at_300=args.unfiltered_at_300,
        )
        if flipped:
            if cfg.image_size is not None:
                h, w = x.shape[-2:]
                size = cfg.image_size
                size = tuple(round(v * size / max(h, w)) for v in (h, w)) if isinstance(size, int) else size
                if size != (h, w):
                    x = interpolate(x[None], size=size, mode="bilinear", align_corners=False, antialias=True)[0]
                if common["vertical_fov_deg"] is None:
                    common["vertical_fov_deg"] = args.fov_deg * h / w
            dvd = InverseEndpointDVD(InverseConfig(
                **common, invert_frequency=args.invert_frequency, invert_amplitude=args.invert_amplitude,
                birth_frequency_cpd=args.inverse_birth_cpd, amplitude_quantile=args.inverse_quantile,
                taper=args.inverse_taper, band_limit_cpd=None if args.inverse_native_band else REFERENCE_BAND_CPD))
        else:
            dvd = EndpointLinearDVD(cfg)
        y, meta = dvd(x, args.months, return_diagnostics=True)
    except ValueError as exc:
        parser.error(str(exc))
    y = y.mul(255).round().to(device="cpu", dtype=torch.uint8).permute(1, 2, 0).contiguous()
    Image.frombytes(mode, (y.shape[1], y.shape[0]), y.numpy().tobytes()).save(args.output)
    print(json.dumps(meta[0], indent=2))


if __name__ == "__main__":
    main()
