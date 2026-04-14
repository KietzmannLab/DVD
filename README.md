# Scale-Free Developmental Visual Diet (DVD)

[![arXiv](https://img.shields.io/badge/arXiv-2507.03168-b31b1b.svg)](https://arxiv.org/abs/2507.03168)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

![DVD pipeline overview](./assets/DVD_pipeline.png)

DVD is a **plug-and-play training curriculum** that progressively "ages" input images. By mimicking the maturation of the human visual system, from the blurry, low-contrast world of an infant to the high-fidelity vision of an adult, DVD bridges the gap between biological learning and Artificial Neural Networks (ANNs).

---

## 💡 Why DVD?

Standard ANNs are fed "adult" high-fidelity data from day one. In contrast, human infants learn from a highly constrained sensory diet. DVD models this trajectory through three core lenses:

- **Acuity:** Spatial resolution that sharpens over time.
- **Contrast:** Sensitivity to brightness differences that expands across frequencies.
- **Color:** Chromatic saturation that gradually matures.

The **Scale-Free** version reformulates these transformations relative to image geometry and Nyquist limits, ensuring consistent behavior across different resolutions and viewing conditions.

### 📈 Developmental Trajectories

![Age-dependent visual development curves](./assets/DVD_trajectories.png)

---

## 🚀 Quick Start

### Installation

Setup typically takes 1–3 minutes on a standard machine.

```bash
git clone https://github.com/KietzmannLab/DVD.git
cd DVD
pip install -e .
```

### Minimal Usage

```python
import torch
from dvd_scale_free.dvd_scale_free.development import DVDTransformer, DVDConfig

config = DVDConfig(image_size=256,apply_blur=1,apply_contrast=1,apply_color=1, cs_logspan_start=5e-3)
transformer = DVDTransformer(config)

x = torch.rand(1, 3, 256, 256)
y = transformer(x, months=1)

print(f"Output shape: {y.shape}")
```

### Visualisation Demo Across Developmental Months
To visualise how example RGB images change across developmental ages, run:
```bash
python tests/dvd_images_demo.py
```
This generates a grid of transformed images across multiple months and is useful for quickly inspecting the qualitative effect of DVD.

You can even use the decomposition mode to visualise either the information preserved by the transformation or the complementary information filtered out by it. This makes it possible to inspect both what remains after the cutoff and what is removed across development.

---

## 🛠️ The Pipeline

DVD applies a three-stage differentiable transformation:

1. **Acuity:** Gaussian or frequency-domain filtering to limit spatial detail.
2. **Contrast sensitivity:** Frequency-Amplitude domain filtering that controls how much contrast information is preserved across development.
3. **Color:** Progressive desaturation and restoration of chromatic channels.

### Core Hyperparameters

While highly configurable, the curriculum is primarily governed by two key "knobs":

| Hyperparameter | Description |
| :--- | :--- |
| `months_per_epoch` | Maps training epochs to developmental time (in months), controlling the rate of visual maturation. |
| `cs_logspan_start` | Main contrast-sensitivity hyperparameter. Sets the initial mapping from contrast sensitivity to contrast amplitude in the frequency domain. Lower values yield higher amplitudes mapping and thus lower initial visual fidelity. |

---

## 📊 Training Example

Integrate DVD into your training loop by mapping training steps to "virtual months":

```python
from dvd_scale_free.dvd_scale_free.development import DVDTransformer, DVDConfig, generate_age_months_curve

# Setup developmental age curve.
# Example here: 150 epochs × 2 months/epoch = 300 virtual months total (= 25 years).
# You can adjust `months_per_epoch` depending on how fast you want development to progress across epochs/batches:
age_curve = generate_age_months_curve(
    total_epochs=150,
    len_train_loader=len(train_loader),
    months_per_epoch=2
)

config = DVDConfig(image_size=224,apply_blur=1,apply_contrast=1,apply_color=1, cs_logspan_start=5e-3)
transformer = DVDTransformer(config)

# Inside training loop:
for i, (images, targets) in enumerate(train_loader):
    step_idx = (epoch * len(train_loader)) + i
    current_age = age_curve[step_idx]

    # Age the batch
    images_aged = transformer(images.cuda(), months=current_age)

    # Standard forward/backward pass...
```

---

## 🧪 Experiments & CLI

Run a full training session on **Ecoset** or **ImageNet** using the provided training script:

```bash
python scripts/main.py /share/klab/datasets \
  --arch resnet50 \
  --dataset-name ecoset_square256 \
  --class-weights-json-path /share/klab/datasets/optimized_datasets/lookup_ecoset_json.json \
  --development_strategy dvd \
  --time_order chronological \
  --months_per_epoch 2 \
  --cs_logspan_start 0.005 \
  --epochs 150 \
  --image-size 256 \
  --batch-size-per-gpu 512 \
  --lr 1e-4 \
  --label-smoothing 0 \
  --seed 1
```

### Key CLI Flags

- `--development_strategy`: Use `dvd` for the curriculum or `adult` for standard training.
- `--time_order`: `chronological` (normal aging) or `randomized` (shuffled ages).
- `--blur_mode`: Choose between `gaussian` or `frequency-domain` acuity.

---

## 📚 Datasets

This project makes use of several datasets:

| Dataset   | Description                                                               | Link |
|-----------|---------------------------------------------------------------------------|------|
| **Ecoset** | A natural image dataset introduced in Mehrer et al., 2021                 | [Ecoset Website](https://www.kietzmannlab.org/ecoset/) |
| **ImageNet** | Our models were also trained on the initial release of ImageNet         | [ImageNet Website](https://www.image-net.org/) |

---

## ✍️ Citation

If you use DVD in your research, please cite our work:

```bibtex
@article{lu2025dvd,
  title   = {Adopting a human developmental visual diet yields robust, shape-based AI vision},
  author  = {Lu, Zejin and Thorat, Sushrut and Cichy, Radoslaw M. and Kietzmann, Tim C.},
  journal = {arXiv preprint arXiv:2507.03168},
  year    = {2025},
  doi     = {10.48550/arXiv.2507.03168},
  url     = {https://arxiv.org/abs/2507.03168}
}
```
