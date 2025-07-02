# Adopting a human developmental visual diet (DVD) yields robust, shape-based AI vision models

This repository provides code and utilities for simulating early visual development (EVD) in deep learning models. By modifying images in a manner consistent with how vision develops in early life, we aim to investigate how early experience may influence model performance and representation.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/KietzmannLab/DVD.git
   cd DVD
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
    torchrun --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="localhost" --master_port=10021 scripts/main.py --arch resnet50 --lr-scheduler fixed --lr 5e-5 --label-smoothing 0.1 --dataset-name texture2shape_miniecoset --development-strategy evd --time-order normal --months-per-epoch 2 --contrast_threshold 0.1 --decrease_contrast_threshold_spd 150 --epochs 300 --image-size 256 --batch-size 512 '/share/klab/datasets'

    ```
    In this example:
    
	- --development-strategy evd enables early visual development transformations.

	- --months-per-epoch 2 sets how many “virtual months” of development each epoch simulates.

	- --time-order normal applies development changes in a typical forward chronological order.

	- Other parameters control model architecture, data settings, etc.

    - 💡 If --development-strategy is set to anything other than 'evd', the parameters --months-per-epoch, --time-order, --contrast_threshold, and --decrease_contrast_threshold_spd are not required and will be ignored.
