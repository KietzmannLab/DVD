# DVD scale-free

[![arXiv](https://img.shields.io/badge/arXiv-2507.03168-b31b1b.svg)](https://arxiv.org/abs/2507.03168)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**DVD scale-free** models age-dependent visual acuity, contrast sensitivity, and color sensitivity during network training. For human development, `months_per_epoch` is the only DVD setting you need to choose. Keep the other defaults unless, for example, you need to match a different viewing angle.

[Interactive demo](https://dvd-scale-free.zejinlu.com/) | [Paper](https://doi.org/10.1038/s42256-026-01228-6)

![DVD pipeline overview](assets/DVD_pipeline.png)

## Installation

```sh
git clone --branch scale_free https://github.com/KietzmannLab/DVD.git
cd DVD
python -m pip install -e '.[demo,train]'
```

## Development-Oriented: Model Human Visual Development

Enable DVD with `--development-strategy dvd`. The schedules below are designed to span 300 months (25 years).

| Schedule | Training epochs | Months per epoch |
| :--- | ---: | ---: |
| DVD-B | 150 | 2 |
| DVD-S | 300 | 1 |
| DVD-P | 75 | 4 |

Spatial frequency is measured in cycles per degree, so the curriculum is not tied to one image size. The default horizontal visual angle (`fov_deg`) is 15 degrees; the vertical angle (`vertical_fov_deg`) follows the image aspect ratio. The adult contrast anchor adjusts to the image's usable frequency range.

![Human visual development curves](assets/DVD_trajectories.png)

**Example: DVD-B on ImageNet-1K**

```sh
torchrun --standalone --nproc-per-node=2 -m scripts.main \
  --dataset-name imagenet --imagenet-path /data/imagenet \
  --development-strategy dvd --months_per_epoch 2 --output-dir runs/dvd-b
```

### Preview an Image

```sh
python -m scripts.dvd_image assets/example_stimuli/example_2.jpeg face_48mo.png --months 48
```

In Python, pass a `CHW` or `NCHW` float tensor in `[0, 1]` before model normalization:

```python
from dvd_scale_free import DVDTransformer

aged = DVDTransformer()(images, months=48)
```

## AI-Oriented: Bidirectional Steering of Bias in the Model

To test whether the visual diet shifts reliance on shape or texture, change the developmental speed or try a different spectral starting point.

- **Speed (`V`):** `V = 1` follows the human curves, `V > 1` speeds up maturation, and `0 < V < 1` slows it down. By default this applies to acuity, contrast, and color; use `speed_components` to retime only a subset.
- **Inversion:** The frequency and amplitude switches are independent. Select **Compare all** in the demo to see the four conditions at the same age.

| `invert_frequency` | `invert_amplitude` | Starting point |
| :---: | :---: | :--- |
| Off | Off | Low frequency, high amplitude (forward DVD) |
| On | Off | High frequency, high amplitude |
| Off | On | Low frequency, low amplitude |
| On | On | High frequency, low amplitude (inverse DVD) |

```python
from dvd_scale_free import make_curriculum

dvd = make_curriculum(speed=0.5, invert_frequency=True, invert_amplitude=True)
aged = dvd(images, months=48)
```

Color development is on by default for forward DVD and off for inverted modes; pass `apply_color=True` or `apply_color=False` to `make_curriculum` to override it.

Random resized crop (`--rrc`) is off by default. [Recipe B's large-ratio RRC augmentation](dvd_scale_free/datasets/augmentations.py#L9) keeps 70 to 100% of the image area.

## Datasets

[Ecoset](https://www.kietzmannlab.org/ecoset/) and [ImageNet-1K](https://www.image-net.org/).

## Citation

```bibtex
@article{lu2026developmental,
  title={Adopting a human developmental visual diet yields robust and shape-based AI vision},
  author={Lu, Zejin and Thorat, Sushrut and Cichy, Radoslaw M. and Kietzmann, Tim C.},
  journal={Nature Machine Intelligence},
  volume={8},
  pages={735--748},
  year={2026},
  doi={10.1038/s42256-026-01228-6}
}
```
