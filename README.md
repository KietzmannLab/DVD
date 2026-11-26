# Scale-Free Developmental Visual Diet (DVD)

[![arXiv](https://img.shields.io/badge/arXiv-2507.03168-b31b1b.svg)](https://arxiv.org/abs/2507.03168)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**DVD scale-free v1 (Endpoint Linear version)** models age-dependent visual acuity, contrast sensitivity, and color sensitivity during network training.

## Development-Oriented: Model Human Visual Development

**Enable DVD with `--development-strategy dvd` and set `--months_per_epoch`.** Once you have chosen the number of training epochs, this is the only DVD parameter you need to adjust. The defaults follow the human fits (`V = 1`) in chronological order, with acuity, contrast, and color enabled on RGB inputs. Override the visual angles only to match a different viewing setup. Without DVD, the general trainer uses adult inputs with no developmental filtering.

### Install and Train

```sh
git clone --branch scale_free https://github.com/KietzmannLab/DVD.git
cd DVD
python -m pip install -e '.[demo,train]'
```

The planned end age in months is `epochs * months_per_epoch`. For a fixed training budget, set `months_per_epoch = 300 / epochs` to span birth to **25 years (300 months)**. If you fix the number of months per epoch instead, divide your target age in months by that value and round up to get the number of epochs.

| Training epochs | Months per epoch | End age |
| :--- | :--- | :--- |
| 150 | 2 | 300 months (25 years) |
| 300 | 1 | 300 months (25 years) |
| 180 | 2 | 360 months (30 years) |

The first batch is at 0 months; the last batch is just before the planned boundary. For example, at 2 months per epoch, run **at least 151 epochs** to train on batches at 300 months or later. At and beyond 300 months, the forward contrast filter stays at its adult maturation level; it does not switch off.

```sh
WANDB_MODE=offline torchrun --standalone --nproc-per-node=2 -m scripts.main /data \
  --dataset-name imagenet --imagenet-path /data/imagenet \
  --development-strategy dvd --epochs 150 --months_per_epoch 2
```

This is the general training command. To match the original RN50 training recipe, use the [P004 Paper Runner](#p004-paper-runner).

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

- Contrast is **on** by default when DVD is selected. The native P004 launcher's `--apply_contrast 0` disabled a superseded filter; in this repository it disables DVD contrast. Do not copy that flag.
- The trainer uses batched real FFTs for square float32 RGB or single-channel batches at least 224 px with default geometry and both spatial stages enabled. Other settings or diagnostic requests can use a slower reference implementation. `--dvd-backend batched` requires the accelerated path; `--dvd-chunk-size` controls its temporary memory.
- Validation uses unfiltered adult inputs. Checkpoints record the filter and schedule; resume with matching settings. Use separate working directories for independent runs.

### P004 Paper Runner

The opt-in recipe ports the P004 runner used for the forward DVD V=1 RN50 experiment. It fixes the training schedule at 150 epochs and 2 months per epoch and requires a full ImageNet-1K ImageFolder tree and two CUDA GPUs on one node:

```sh
python -m scripts.main --training-recipe p004-paper --describe-dvd

torchrun --standalone --nproc-per-node=2 -m scripts.main \
  --training-recipe p004-paper \
  --imagenet-path /path/to/imagenet --output-dir runs/p004-paper-v1
```

| Component | P004 paper recipe |
| :--- | :--- |
| Inputs and augmentation | Ordered JPEGs and common labels from ImageFolder. PIL short-edge resize, then full-frame bilinear 256 px resize (`align_corners=True`, no tensor antialias). Paper Kornia operators draw parameters per image in GPU batches, before FP32 DVD. No RRC or Recipe B. |
| Schedule and optimizer | 150 epochs, 1,251 steps per epoch, 2 months per epoch, 512 images per rank, seed 0 by default. Class-weighted cross-entropy, fused Adam at `1e-4` with `weight_decay=1e-6`, channels-last layout, FP16 model autocast, GradScaler, and the full-batch SyncBN path. |
| Validation and best | Eager inference on 50,000 unfiltered adult images, without augmentation. Both ranks reduce weighted loss and hit counts; best is chosen by whole-set top-1. |
| Inputs and restart verification | Check ImageNet split sizes and class mapping and compare ordered ID hashes and sampled JPEG hashes on both ranks. Save the manifest, library versions, age schedule, optimizer/scaler, and each rank's separate RNG state. Last, best, and periodic checkpoints are written atomically. |

On the lab cluster, omitting the data path reads `/share/klab/datasets/cached_datasets/imagenet_p004` directly. No dataset cache is copied to RAM or local scratch. The paper loader uses bounded batch queues (24 train and 8 validation workers per rank by default); `/dev/shm` is needed for the queues, not for the dataset itself. The runner checks for 80 GiB available RAM and 24 GiB free shared-memory space when the Linux counters are available. For Slurm, the reference allocation is `--mem=105G --cpus-per-task=60` for both ranks together. A known Slurm RAM allocation below 80 GiB is rejected even if the node has ample free RAM.

Continue a new run with the same options and `--resume runs/p004-paper-v1/weights/checkpoint_last.pth`. For a trusted native `linear_v1` checkpoint, the runner checks the saved paper flags, age schedule, and optimizer; continuing training additionally requires both original rank RNG states. Use a new output directory for that continuation and retain the original best checkpoint. To replay ImageNet validation without training or writing checkpoints, use the paper recipe with `--evaluate --resume /path/to/checkpoint_best.pth` and the same ImageNet split.

Use the same ImageNet bytes, PyTorch/CUDA, torchvision, Kornia, and Pillow versions, and comparable GPU topology for quantitative retraining comparisons. The original B200 protocol used the torch 2.7.1 / CUDA 12.8 stack. The data manifest hashes all ordered IDs and labels, but only spot-checks JPEG bytes. A new run is not a bitwise reproduction of the archived weights: GPU FFT/scatter arithmetic and the exact original resume history still matter. Replaying the paper metrics requires the archived checkpoint and evaluation inputs, not a fresh training run.

The general CLI selects the same V=1 **visual transform** as P004 under the matching input geometry, but retains its separate trainer. Its CPU sample augmentations, Adam defaults and rank-local validation are not a replay of the P004 protocol. Use the paper runner above for a retraining comparison and the original checkpoint and evaluation preprocessing for exact analysis replay.

## AI-Oriented: Steer Model Bias

These controls are for experiments on how the **training visual diet steers model biases**, such as reliance on shape or texture. Vary how fast the model sees new information, which frequencies or amplitudes it sees first, or whether it receives random crops. **Skip this section if you only want to model human visual development.** Use the general training CLI for these experiments; `p004-paper` is fixed to the human schedule and paper augmentation.

### Change Developmental Speed

| Setting | Default and meaning |
| :--- | :--- |
| `speed` (V) | `1.0` follows the original human curves. At the same scheduled age, `V > 1` makes selected stages mature faster and `0 < V < 1` makes them mature more slowly. The scheduled age set by `months_per_epoch` stays the same. |
| `speed_components` | `("acuity", "contrast", "color")` by default. Select any subset, or use `()` to leave all timing unchanged. Unselected stages still develop at the human rate. |
| `apply_acuity`, `apply_contrast`, `apply_color` | Separate switches that disable or enable each forward stage; they do not change its developmental speed. |

In forward DVD, acuity uses its own developmental curve. For elapsed age fraction `u = t / 300`, a selected acuity stage evaluates the original double-exponential Gaussian width at `300 * V * u / (1 - u + V * u)` months. Selected contrast uses `p = H(t) ** (1 / V)`, where `H` is the human contrast Hill fit normalized at 300 months. Selected color evaluates the original color fit at `inverse_H(p)`. Retiming changes neither the underlying fits nor the birth and 300-month endpoints. Retiming acuity can still change the spectrum used to compute the contrast threshold, even when the contrast clock stays fixed.

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

At V=1, the original forward transform is unchanged: its frequency line is the Gaussian half-gain and its amplitude gate acts on radial means. The flipped controls instead use calibrated Fourier coefficients on the even-reflected field, with two soft gates. For flips, `acuity` retimes the frequency gate and `contrast` retimes the amplitude gate. To preserve the previous flipped-mode behavior, select `("acuity", "contrast")`. Both flips at V=1 match the full-color 256 px inverse-DVD filter from the steering experiments. Flipped modes are full-color by default and return the input unchanged at 300 months. Random grayscale is also off by default for flipped modes in the training CLI.

```sh
python -m scripts.main --describe-dvd --development-strategy dvd \
  --invert-frequency --invert-amplitude --rrc --rrc-scale 0.08 1.0
python -m scripts.dvd_image input.jpg inverse.png --months 12 --invert-frequency --invert-amplitude
```

Add the inversion and crop flags to the general training command. RRC is **off by default** and is never applied in validation. `--rrc` enables it independently of the DVD mode; `--rrc-scale MIN MAX` changes the area fraction and `--rrc-ratio MIN MAX` changes the aspect ratios (default 3:4 to 4:3). CLI crop draws use Kornia, not the browser's preview seed. The P004 paper recipe does not use RRC.

For transfer, keep the visual field and aspect ratio fixed (do not stretch the input). The flipped modes calibrate their birth amplitude quantile in a fixed physical 256 px reference band capped at the actual common Nyquist frequency. `--inverse-native-band` uses the complete sampled band instead. Very low resolutions may not support the requested birth frequency. Matching the physical filter does not guarantee invariant RN50 accuracy or shape bias across input resolutions.

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
