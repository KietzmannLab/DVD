# Scale-Free Developmental Visual Diet (DVD)

[![arXiv](https://img.shields.io/badge/arXiv-2507.03168-b31b1b.svg)](https://arxiv.org/abs/2507.03168)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**DVD scale-free v1 (Endpoint Linear version)** models age-dependent visual acuity, contrast sensitivity, and color sensitivity during network training.

## Development-Oriented: Model Human Visual Development

**Enable DVD with `--development-strategy dvd` and choose `--months_per_epoch`.** The defaults follow the human acuity, contrast, and color curves (`V = 1`) from birth onward. Leave the other settings alone unless you need to match a different viewing angle. Without DVD, training uses adult inputs.

### Install and Train

```sh
git clone --branch scale_free https://github.com/KietzmannLab/DVD.git
cd DVD
python -m pip install -e '.[demo,train]'
```

Choose a DVD schedule below to span birth to **25 years (300 months)**. The planned end age is `epochs * months_per_epoch`.

| DVD schedule | Training epochs | Months per epoch | Planned end age |
| :--- | :--- | :--- | :--- |
| DVD-B | 150 | 2 | 300 months (25 years) |
| DVD-S | 300 | 1 | 300 months (25 years) |
| DVD-P | 75 | 4 | 300 months (25 years) |

Training starts at 0 months and stops just before the planned end age. To include 300-month inputs, add an epoch without changing the rate (epoch 151 for DVD-B). The forward contrast filter reaches its adult setting at 300 months and stays on.

The example below uses the DVD-B schedule:

```sh
WANDB_MODE=offline torchrun --standalone --nproc-per-node=2 -m scripts.main /data \
  --dataset-name imagenet --imagenet-path /data/imagenet \
  --development-strategy dvd --epochs 150 --months_per_epoch 2
```

To match the original ResNet-50 training setup, use [Paper Training](#paper-training).

Inspect the resolved settings without loading data or allocating a GPU:

```sh
python -m scripts.main --describe-dvd --development-strategy dvd --epochs 150 --months_per_epoch 2
```

### Preview an Age

![DVD pipeline overview](./assets/DVD_pipeline.png)

For only the Python transform, install with `python -m pip install -e .`.

```python
import torch
from dvd_scale_free import DVDTransformer

dvd = DVDTransformer()
x = torch.rand(1, 3, 256, 256)
y, diagnostics = dvd(x, months=48, return_diagnostics=True)
print(y.shape, diagnostics[0]["adult_anchor_cpd"])
```

Use float tensors in `[0, 1]` before model normalization, in `CHW` or `NCHW` format. Single-channel images skip color development. For RGBA, only RGB is filtered. The transform keeps the input size unless `image_size` is set to a long-edge length or an explicit `(height, width)`. It runs without autograd.

```sh
python -m scripts.dvd_image assets/example_stimuli/example_1.jpeg dvd_48mo.png --months 48
python tests/dvd_images_demo.py --output dvd_gallery.pdf
```

The demo-only `--unfiltered-at-300` returns the prepared image unchanged from 300 months onward. Leave it off to preview training inputs.

### Defaults and Viewing Geometry

![Age-dependent visual development curves](./assets/DVD_trajectories.png)

1. **Acuity:** A reflected Gaussian filter operates in cycles per degree (cpd).
2. **Contrast:** The luminance spectrum after acuity filtering gives two radial amplitude anchors. The threshold interpolates linearly between the anchor amplitudes as contrast sensitivity matures. A log-amplitude sigmoid attenuates components below the threshold while preserving DC.
3. **Color:** The original age-dependent grayscale-to-RGB blend follows the spatial filters.

Keep these defaults for human development unless your stimulus has a different known field of view.

| Setting | Default and meaning |
| :--- | :--- |
| `fov_deg`, `vertical_fov_deg` | The horizontal visual angle is `15` degrees. The default vertical angle follows the source aspect ratio: horizontal angle times image height divided by width. Override these angles only to match a different viewing setup (CLI: `--dvd-fov-deg`, `--dvd-vertical-fov-deg`). |
| `infant_anchor_cpd` | `0.23` cpd, near the spatial frequency at which newborns are most sensitive to contrast ([Brown et al., 2018](https://doi.org/10.1167/tvst.7.3.18)). `"fundamental"` uses the lowest nonzero sampled frequency instead. |
| `adult_anchor_cpd` | `None` (auto). Keep the fixed adult reference frequency when the image can resolve it; otherwise cap it at the current image's usable Nyquist frequency. Pass a number to set the target in cpd, or `"nyquist"` to use the current image's capped Nyquist value directly. |
| `sigmoid_width_log10` | `0.10` log10 amplitude units. This sets the width of the soft contrast threshold. |
| `speed`, `speed_components` | Leave `V = 1` and the default components unchanged for the human fits. See [AI-oriented controls](#ai-oriented-steer-model-bias) to change the rate. |

The automatic adult anchor is calibrated to 224 pixels over 15 degrees: `0.98 * 224 / (2 * 15)`, about 7.3173 cpd. For the current filtered image, the Nyquist frequency of each axis is its pixel count divided by twice its visual angle. Auto uses the smaller of the reference and 98% of the lower axis Nyquist frequency. Raising resolution alone therefore does not raise the reference frequency.

### Training Notes

- Contrast is **on** by default. Leave it on for human development: `--apply_contrast 0` disables DVD contrast here.
- The trainer uses batched real FFTs for square float32 RGB or single-channel batches at least 224 px with default geometry and both spatial stages enabled. Other settings or diagnostic requests can use a slower reference implementation. `--dvd-backend batched` requires the accelerated path; `--dvd-chunk-size` controls its temporary memory.
- Validation uses unfiltered adult inputs. Checkpoints record the filter and schedule; resume with matching settings. Use separate working directories for independent runs.

### Paper Training

Train ResNet-50 on ImageNet-1K with DVD-B timing (150 epochs, 2 months per epoch), the [paper augmentation](dvd_scale_free/datasets/paper_imagenet.py#L87), and no random resized crop. Use the full ImageNet-1K training and validation splits and two CUDA GPUs on one node.

```sh
torchrun --standalone --nproc-per-node=2 -m scripts.main \
  --training-recipe paper \
  --imagenet-path /path/to/imagenet --output-dir runs/dvd-b-paper
```

The paper setup uses 256 px images, 512 images per GPU and 1,251 steps per epoch. It applies augmentation before DVD, then trains with FP16, synchronized batch normalization, class-weighted cross-entropy, and fused Adam (learning rate `1e-4`, weight decay `1e-6`). The best checkpoint is chosen by top-1 accuracy across all 50,000 unfiltered validation images.

## AI-Oriented: Steer Model Bias

Use these controls to test whether changing the training visual diet shifts a model's reliance on shape or texture. Skip this section if you only want to model human development. The paper setup does not use these controls.

### Change Developmental Speed

| Setting | Default and meaning |
| :--- | :--- |
| `speed` (V) | `1.0` follows the original human curves. At the same scheduled age, `V > 1` makes selected stages mature faster and `0 < V < 1` makes them mature more slowly. The scheduled age set by `months_per_epoch` stays the same. |
| `speed_components` | `("acuity", "contrast", "color")` by default. Select any subset, or use `()` to leave all timing unchanged. Unselected stages still develop at the human rate. |
| `apply_acuity`, `apply_contrast`, `apply_color` | Separate switches that disable or enable each forward stage; they do not change its developmental speed. |

Forward acuity uses its own age curve: for `u = t / 300`, it evaluates the original Gaussian width at `300 * V * u / (1 - u + V * u)` months. Contrast uses `p = H(t) ** (1 / V)`, where `H` is the human contrast Hill fit normalized at 300 months; color uses age `inverse_H(p)`. These changes preserve the birth and 300-month endpoints. Changing acuity can still change the spectrum used to set the contrast threshold.

For example, slow contrast development only:

```python
from dvd_scale_free import DVDConfig, DVDTransformer

contrast_only = DVDTransformer(DVDConfig(speed=0.5, speed_components=("contrast",)))
```

Add `--dvd-speed 0.5 --dvd-speed-components contrast` to the general training command, or preview the effect on one image:

```sh
python -m scripts.dvd_image input.jpg slower_contrast.png --months 48 --speed 0.5 --speed-components contrast
```

The browser's slider range is not a training constraint.

### Inverse DVD and Crop Controls

Two independent spectral switches select the starting region:

| Invert frequency | Invert amplitude | Starting region | Preset |
| :---: | :---: | :--- | :--- |
| Off | Off | Coarse frequencies, strong amplitudes | Forward DVD |
| On | Off | Fine frequencies, strong amplitudes | Frequency-inverted control |
| Off | On | Coarse frequencies, weak amplitudes | Amplitude-inverted, exploratory |
| On | On | Fine frequencies, weak amplitudes | Full-color inverse-DVD |

```python
from dvd_scale_free import RRCConfig, make_curriculum, training_rrc

dvd = make_curriculum(invert_frequency=True, invert_amplitude=True, speed=1.0)
crop = training_rrc(256, RRCConfig(enabled=True, scale=(0.08, 1.0)))
aged = dvd(crop(x) if crop is not None else x, months=24)
```

Forward DVD uses Gaussian acuity and a contrast gate on radial means. The flipped controls use soft gates on the reflected Fourier spectrum: `acuity` retimes the frequency gate and `contrast` retimes the amplitude gate. Select `("acuity", "contrast")` to leave color timing unchanged. Both flips at `V = 1` match the full-color inverse filter at 256 px. Flipped modes leave color intact by default, skip random grayscale, and return the input unchanged from 300 months onward.

```sh
python -m scripts.main --describe-dvd --development-strategy dvd \
  --invert-frequency --invert-amplitude --rrc --rrc-scale 0.08 1.0
python -m scripts.dvd_image input.jpg inverse.png --months 12 --invert-frequency --invert-amplitude
```

In the general trainer, `--rrc` adds random resized crop. It is **off by default** and training-only. Use `--rrc-scale MIN MAX` for crop area and `--rrc-ratio MIN MAX` for aspect ratio (default 3:4 to 4:3).

For transfer, keep the visual field and aspect ratio fixed (do not stretch the input). The flipped modes calibrate their birth amplitude quantile in a fixed physical 256 px reference band capped at the actual common Nyquist frequency. `--inverse-native-band` uses the complete sampled band instead. Very low resolutions may not support the requested birth frequency. Matching the physical filter does not guarantee invariant RN50 accuracy or shape bias across input resolutions.

### Recipe B: Large-Ratio RRC

Recipe B centers on random resized crops that retain **70-100% of the image area**, with a 3:4 to 4:3 aspect ratio. The [RGB augmentation function](dvd_scale_free/datasets/augmentations.py#L9) applies the crop, flip, rotation, perspective warp, and blur before DVD. The original training also used CutMix after DVD (probability `0.1`, alpha `0.2`) and label smoothing (`0.1`). The function covers image augmentation only; `--rrc` alone does not reproduce Recipe B.

### Replay a Steering Experiment

For the matched shape/texture steering comparison, use the frozen best-validation RN50 checkpoints rather than retraining from the switches: standard e103, forward e148, full-color inverse e143, and full-color inverse with RRC e147. The inverse runs used the original packed, GPU-augmented pipeline. The matched RRC run used area `(0.08, 1.0)` and ratio `(0.75, 4/3)`, and neither inverse run used random grayscale. The reported scores were measured on unfiltered adult RGB (ImageNet at 256 px; cue conflict at 224 px).

## Version History

- The [last revision shipping the legacy method and the original v1 names](https://github.com/KietzmannLab/DVD/tree/413842b9d90d3a436ce6c32595552e204a2150cd) is `413842b`. Use that revision in a separate environment to replay log-frequency scale-free runs with `--dvd-method legacy` or `cs_logspan_start`. Those settings do not translate to v1.
- The v1 forward API now uses `DVDConfig` and `DVDTransformer` from `dvd_scale_free`. In training commands, drop `--dvd-method endpoint-linear` and change `--endpoint-*` flags to `--dvd-*`. This renames the same algorithm; it does not change the filters or defaults.
- New checkpoints identify the method as `dvd`. Use an explicit `--resume` path for previous v1 runs: the runner accepts the prior method name only if the remaining protocol matches. The numeric protocol schema is separate from the method's v1 name. A checkpoint trained with the earlier Hill-retimed acuity still requires its original revision.

## Tests

```sh
python -m unittest discover -s tests -v
```

On a CUDA node with two GPUs, `PYTHONPATH=. python tests/paper_cuda_smoke.py /path/to/new-smoke-directory` exercises the paper runner on synthetic JPEGs, including optimization, distributed validation, checkpoint save, and a separate-process resume. It does not train the full ImageNet model.

## Datasets

| Dataset   | Description                                                               | Link |
|-----------|---------------------------------------------------------------------------|------|
| **Ecoset** | A natural image dataset introduced in Mehrer et al., 2021                 | [Ecoset Website](https://www.kietzmannlab.org/ecoset/) |
| **ImageNet** | Our models were also trained on the initial release of ImageNet         | [ImageNet Website](https://www.image-net.org/) |

---

## Citation

If you use DVD in your research, please cite our work:

```bibtex
@article{lu2026developmental,
  title={Adopting a human developmental visual diet yields robust and shape-based AI vision},
  author={Lu, Zejin and Thorat, Sushrut and Cichy, Radoslaw M. and Kietzmann, Tim C.},
  journal={Nature Machine Intelligence},
  pages={1-14},
  year={2026}
}
```
