# Early Visual Experience Modeling 

This repository contains code for modeling early visual experience. The package provides scripts and utilities to facilitate research and experimentation in this domain.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/KietzmannLab/blurring4texture2shape1file.git
   cd blurring4texture2shape1file
    ```
2.	Install in editable mode (so local changes are immediately reflected):
    ```
    pip install -e .
    ```

3.	Verify installation:
    ```
    python -c "import evd; print(evd.__version__)"
    ```

4. Example of running a script:

    ```
    torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=10021 scripts/main.py --arch resnet50 --lr-scheduler fixed --lr 5e-5 --label-smoothing 0.1 --dataset-name texture2shape_miniecoset --development-strategy evd --time-order normal --months-per-epoch 2 --contrast_threshold 0.2 --decrease_contrast_threshold_spd 100 --epochs 300 --image-size 256 --batch-size 512 '/share/klab/datasets'

    ```
---
