# Early Visual Experience Modeling

This repository provides code and utilities for simulating early visual development (EVD) in deep learning models. By modifying images in a manner consistent with how vision develops in early life, we aim to investigate how early experience may influence model performance and representation.

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
    In this example:
	•	--development-strategy evd enables early visual development transformations.
	•	--months-per-epoch 2 sets how many “virtual months” of development each epoch simulates.
	•	--time-order normal applies development changes in a typical forward chronological order.
	•	Other parameters control model architecture, data settings, etc.
---

## Training your own model with early visual experience

After installation, you can import evd.evd.development to generate the developmental transformations. In your training loop, you will:
	1.	Generate an age-months curve that maps each training batch to a specific “month” in the simulated development.
	2.	Apply an EVD-based transformation (e.g., blurring, color changes, contrast adjustments) based on the current month.

A minimal training function might look like this:
    ```
    import math
    import evd.evd.development

    def train(
        train_loader,
        model,
        criterion,
        optimizer,
        scaler,
        wandb_run,
        logger,
        epoch,
        args,
    ):
        # Generate age-months curve for EVD
        age_months_curve = evd.evd.development.generate_age_months_curve(
            args.epochs,
            len(train_loader),
            args.months_per_epoch,
            mid_phase=(args.time_order == "mid_phase"),
            shuffle=(args.time_order == "random"),
            seed=args.seed,
        )
        
        model.train()
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            # Determine the 'age_months' for this batch
            age_months = age_months_curve[epoch][batch_idx]
            
            # Experience across visual development
            if args.development_strategy == "evd":
                # Dynamically adjust contrast threshold
                contrast_control_coeff = max(
                    math.floor(age_months / args.decrease_contrast_threshold_spd) * 2, 
                    1
                )
                images = evd.evd.development.EarlyVisualDevelopmentTransformer().apply_fft_transformations(
                    images,
                    age_months,
                    apply_blur=args.apply_blur, 
                    apply_color=args.apply_color, 
                    apply_contrast=args.apply_contrast,
                    contrast_threshold=args.contrast_threshold / contrast_control_coeff,
                    image_size=args.image_size,
                    verbose=False,
                )
            elif args.development_strategy == "adult":
                pass  # No transformation applied
            else:
                raise NotImplementedError(
                    f"Development strategy {args.development_strategy} not implemented"
                )
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            # Backward pass & optimization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            if wandb_run:
                wandb_run.log({"loss": loss.item(), "epoch": epoch})
            
            # Optionally log or print progress
            logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
    ```

Data Augmentation Pipelines

You can either:
	•	Use the default supervised pipeline transform from: evd.datasets.dataset_loader.SupervisedLearningDataset.get_supervised_pipeline_transform
	•	Or write your own data augmentation transforms that best suit your experimental setup.

⸻

Citation

If you use this repository or find it helpful in your research, please cite it accordingly : XXX.


⸻

License

This project is licensed under the MIT License. Please see the LICENSE file for details.


---