"""Apply Endpoint Linear DVD to one image and print the resolved filter settings."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageOps
import torch

from dvd_scale_free.dvd_scale_free._cli import adult_anchor, infant_anchor
from dvd_scale_free.dvd_scale_free.endpoint_linear import EndpointConfig, EndpointLinearDVD


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
                        help="contrast speed V: >1 matures sooner, <1 matures later (default: 1)")
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
    parser.add_argument("--device", default="cpu", help="cpu or cuda[:index]")
    args = parser.parse_args()
    if args.output.suffix.lower() != ".png" or args.output.resolve() == args.input.resolve():
        parser.error("Use a different file with a .png extension for output")
    try:
        cfg = EndpointConfig(
            speed=args.speed, fov_deg=args.fov_deg, vertical_fov_deg=args.vertical_fov_deg,
            infant_anchor_cpd=args.infant_anchor_cpd, adult_anchor_cpd=args.adult_anchor_cpd,
            sigmoid_width_log10=args.sigmoid_width_log10, image_size=args.image_size,
            unfiltered_at_300=args.unfiltered_at_300,
        )
        with Image.open(args.input) as original:
            im = ImageOps.exif_transpose(original)
            mode = ("RGBA" if "A" in im.getbands() or "transparency" in im.info
                    else "L" if im.mode in ("1", "L") else "RGB")
            im = im.convert(mode)
            x = torch.frombuffer(bytearray(im.tobytes()), dtype=torch.uint8)
            x = x.reshape(im.height, im.width, len(mode)).permute(2, 0, 1)
        x = x.to(args.device, dtype=torch.float32).div_(255)
        y, meta = EndpointLinearDVD(cfg)(x, args.months, return_diagnostics=True)
    except ValueError as exc:
        parser.error(str(exc))
    y = y.mul(255).round().to(device="cpu", dtype=torch.uint8).permute(1, 2, 0).contiguous()
    Image.frombytes(mode, (y.shape[1], y.shape[0]), y.numpy().tobytes()).save(args.output)
    print(json.dumps(meta[0], indent=2))


if __name__ == "__main__":
    main()
