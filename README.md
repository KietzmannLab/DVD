![DVD pipeline overview](./assets/DVD_pipeline.png)

# Adopting a human developmental visual diet (DVD) diet yields robust, shape-based AI vision

A *plug-and-play training curriculum* that “ages” each image to mimic the developing optics and neural sensitivities of the human visual system from immature newborn vision to mature adulthood.

---

## 1 Why developmental visual diet (DVD)?

* Human visual systems start from 'immature', while currently AI training is mianly instant with high-fidelity.  
* We fit psychophysical data for **visual acuity**, **contrast sensitivity**, and **chromatic sensitivity** from 0 m-25 y.  
* A differentiable pipeline applies matching visual acuity, contrast sensitivity, and chromatic sensitivity *on-the-fly*.  
* Guiding AI vision through this DVD yields models that closely align with human robust vision across key hallmarks: 1) near-human shape bias, 2) greater corruption robustness, 3) improved adversarial resilience, 4) Abstract shape recognition beyond tested VLMs (e.g. ChatGPT-4o, Gemini 2.0 Flash, and LLaMA-4-Scout)

### 📈 Developmental Visual Trajectories  

![Age-dependent visual development curves](./assets/DVD_trajectories.png)

---

## 2 Installation

```bash
git clone https://github.com/KietzmannLab/DVD.git
cd DVD
pip install -e .
python - <<'PY'
import dvd, torch
print("DVD version:", dvd.__version__, "| CUDA =", torch.cuda.is_available())
PY
```

## 3 Quick demo - aging images

```python
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

# DVD
from dvd.dvd.development import DVDTransformer  

# Configuration
AGES = [1, 4, 16, 64, 256]  # in months
IMG_SIZE = 224              # Resize target size

# Input and output paths
SOURCE_DIR = Path("./assets/example_stimuli/")
IMAGE_PATHS = [
    SOURCE_DIR / "example_1.jpeg",
    SOURCE_DIR / "example_2.jpeg",
]
RESULT_DIR = Path("results/")
OUTPUT_PATH = RESULT_DIR / "dvd_demo_output.pdf"

def load_as_tensor(fp: str | Path) -> torch.Tensor:
    """Load an RGB image as a 4D torch tensor [1, 3, H, W] in [0, 1]."""
    img = Image.open(fp).convert("RGB")
    img.thumbnail((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.asarray(img).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(arr).float().unsqueeze(0)

def grid_demo(image_paths: List[Path], out_file: Path) -> None:
    dvdt = DVDTransformer()
    tensors = [load_as_tensor(p) for p in image_paths]

    rows, cols = len(tensors), len(AGES)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))

    for r, img_t in enumerate(tensors):
        for c, age in enumerate(AGES):
            out = dvdt.apply_fft_transformations(img_t.clone(), age_months=age)
            vis = (out.squeeze(0).permute(1, 2, 0).cpu().numpy()).clip(0, 1)
            axes[r, c].imshow(vis)
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(f"{age} mo", fontsize=12)

    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=200)
    print(f"Demo saved to {out_file.resolve()}")
    
# Run
grid_demo(IMAGE_PATHS, OUTPUT_PATH)
```

## 4 Training with DVD

```bash
torchrun --nproc_per_node=1 scripts/main.py \
  --dataset-name ecoset --data-root /datasets \
  --arch resnet50 --epochs 300 --batch-size 512 --image-size 256 \
  --lr 1e-4 --lr-scheduler fixed \
  --development_strategy dvd --months_per_epoch 2 --time_order chronological \
  --apply_blur --apply_color --apply_contrast \
  --contrast_amplitude_beta 0.1 --contrast_amplitude_lambda 150
```

| Flag                     | Purpose                                             |
|--------------------------|-----------------------------------------------------|
| `--development_strategy` | `dvd` (full curriculum) or `adult` (control).       |
| `--months_per_epoch`     | Virtual ageing per epoch (smaller = finer).         |
| `--time_order`           | `chronological`, `fully_random`.                    |
| `--apply_*`              | Toggle acuity / colour / contrast stages.           |
| `--contrast_amplitude_*` | Fine-tune frequency thresholding.                   |


## 5 Core API

```python
from dvd.dvd.development import DVDTransformer, generate_age_months_curve

# Initialize transformer and generate age mapping curve
dvdt = DVDTransformer()
age_curve = generate_age_months_curve(
    epochs=args.epochs,
    steps_per_epoch=len(train_loader),
    months_per_epoch=args.months_per_epoch,
)

# Map current batch index to virtual age in months
step_idx = (epoch * len(train_loader)) + i
age_months = age_curve[step_idx]

# Apply age-based visual transformations
images_aged = dvdt.apply_fft_transformations(
    images,  # Tensor [B, 3, H, W] in [0, 1]
    age_months=age_months,
    apply_blur=True,
    apply_color=True,
    apply_contrast=True,
    contrast_amplitude_beta=0.1,
    contrast_amplitude_lambda=150,
    image_size=224,
)
```

## 6 Citation

```bash
@article{li2025dvd,
  title   = {Adopting a human developmental visual diet yields robust, shape-based AI vision models},
  author  = {Zejin Lu, Sushrut Thorat, Radoslaw M Cichy & Tim C Kietzmann},
  journal = {Placeholders – will be filled once DOI is live.)},
  year    = {2025}
} -->
```
