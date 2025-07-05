![DVD pipeline overview](./assets/DVD_pipeline.png)

# Adopting a Human Developmental Visual Diet (DVD) diet yields robust, shape-based AI vision

A **plug-and-play training curriculum* that “ages” each image to mimic the developing optics and neural sensitivities of the human visual system from immature newborn vision to mature adulthood. 
---

## 🔎 Summary  

Despite years of research and scaling, a stark misalignment between human and AI vision remains. Unlike humans, AI systems over-rely on texture rather than shape, are fragile to distortions, vulnerable to adversarial attacks, and struggle to recognize abstract shapes in context. We trace this gap to a key difference in upbringing: humans are born with vision that is blurry, low in contrast sensitivity, and desaturated—qualities that gradually mature over time. By quantifying these developmental trajectories and training networks on them as a **Developmental Visual Diet (DVD)**, we achieve shape-based recognition performance that surpasses large VLMs trained on orders of magnitude more data. Our findings show that robust AI vision depends not just on how much a model learns, but how it learns.
---

## 📜 Main ideas 

* Human visual systems start from 'immature', while currently AI training is mianly instant with high-fidelity.  
* We fit psychophysical data for **visual acuity**, **contrast sensitivity**, and **chromatic sensitivity** from 0 m-25 y.  
* A differentiable pipeline applies matching visual acuity, contrast sensitivity, and chromatic sensitivity *on-the-fly*.  
* Guiding AI vision through this DVD yields models that closely align with human robust vision across key hallmarks: 1) near-human shape bias, 2) greater corruption robustness, 3) improved adversarial resilience, 4) Abstract shape recognition beyond tested VLMs (e.g. ChatGPT-4o, Gemini 2.0 Flash, and LLaMA-4-Scout)

---

## 📈 Developmental Visual Trajectories  

![Age-dependent visual development curves](./assets/DVD_trajectories.png)

---

## 1 Why developmental visual diet (DVD)?

* **FFT-based contrast gating** — fast & differentiable.  
* **Curriculum scheduling** — chronological, reverse, or fully random.  
* **Zero-friction API** — a single `DVDTransformer` in any dataset.

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

⸻

## 3 Quick demo — ageing images

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
SOURCE_DIR = Path("./data/example_stimuli/")
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


⸻

## 4 Training with DVD

torchrun --nproc_per_node=1 scripts/main.py \
  --dataset-name ecoset --data-root /datasets \
  --arch resnet50 --epochs 300 --batch-size 512 --image-size 256 \
  --lr 1e-4 --lr-scheduler fixed \
  --development_strategy dvd --months_per_epoch 2 --time_order chronological \
  --apply_blur --apply_color --apply_contrast \
  --contrast_amplitude_beta 0.1 --contrast_amplitude_lambda 150

Flag	Purpose
--development_strategy	dvd (full curriculum) or adult (control).
--months_per_epoch	Virtual ageing per epoch (smaller = finer).
--time_order	chronological, fully_random.
--apply_*	Toggle acuity / colour / contrast stages.
--contrast_amplitude_*	Fine-tune frequency thresholding.


⸻

5 Core API

from dvd.dvd.development import DVDTransformer
dvdt = DVDTransformer()

# Generate age months curve to map batches to age months for DVD
age_months_curve = dvd.dvd.development.generate_age_months_curve(
    args.epochs,
    len(train_loader),
    args.months_per_epoch,
)
age_months = age_months_curve[(epoch -0) * len(train_loader) + i]

images_aged = dvdt.apply_fft_transformations(
    images,                 # Tensor [B,3,H,W] ∈ [0,1]
    age_months=age_months,
    apply_blur=True,
    apply_color=True,
    apply_contrast=True,
    contrast_amplitude_beta=0.1,
    contrast_amplitude_lambda=150,
    image_size=224,
    fully_random=False,
)

⸻

8 Citation

@article{li2025dvd,
  title   = {Adopting a human developmental visual diet yields robust, shape-based AI vision models},
  author  = {Zejin Lu, Sushrut Thorat, Radoslaw M Cichy & Tim C Kietzmann},
  journal = {Placeholders – will be filled once DOI is live.)},
  year    = {2025}
}





# <!-- # Adopting a human developmental visual diet (DVD) yields robust, shape-based AI vision models

# This repository provides code and utilities for implmenting developmental visual diet (DVD) in deep learning models training. We reviewed the existing psychophysical evidence on how human vision develops from newborn to 25 years of age and synthesised the developmental trajectories of three core dimensions of visual maturation (visual acuity, chromatic sensitivity, and contrast sensitivity) into a preprocessing pipeline for AI vision. 

# ## Installation

# 1. **Clone the repository**:
#    ```bash
#    git clone https://github.com/KietzmannLab/DVD.git
#    cd DVD
#     ```
# 2.	Install in editable mode (so local changes are immediately reflected):
#     ```
#     pip install -e .
#     ```

# 3.	Verify installation:
#     ```
#     python -c "import dvd"
#     ```

# 4. Example of running a script:

#     ```
#     torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=10021 scripts/main.py --arch resnet50 --lr-scheduler fixed --lr 5e-5 --label-smoothing 0.1 --dataset-name texture2shape_miniecoset --development-strategy dvd --time-order normal --months-per-epoch 2 --contrast_threshold 0.1 --decrease_contrast_threshold_spd 150 --epochs 300 --image-size 256 --batch-size 512 '/share/klab/datasets'

#     ```
#     In this example:
    
# 	- --development-strategy evd enables early visual development transformations.

# 	- --months-per-epoch 2 sets how many “virtual months” of development each epoch simulates.

# 	- --time-order normal applies development changes in a typical forward chronological order.

# 	- Other parameters control model architecture, data settings, etc.

#     - 💡 If --development-strategy is set to anything other than 'evd', the parameters --months-per-epoch, --time-order, --contrast_threshold, and --decrease_contrast_threshold_spd are not required and will be ignored. -->
