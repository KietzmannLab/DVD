
import subprocess
import os
from time import time
import random
import socket
from itertools import product

def find_free_port():
    """Find an available port dynamically."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))  # Bind to any available port
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


# Submission settings
submit_script = "scripts/submit_task_from_array.sh"  # The SLURM array script
cfg = f"./logs/slurm_logs/tmp-cfg-{int(time())}.txt"           # Temporary file to hold commands
nparallel = 8                               # Max number of concurrent tasks in the array

# Example of parameter sweeps (simplify or expand as you need)
seeds       = [1, 2, 3, 4, 5][:1]        # e.g., keep only the first for demonstration
learning_rs = [ 1e-4, 1e-3,5e-4][:1]
batch_sizes = [ 512, 1024][:1]
architectures = ["resnet50", "alexnet"][:1]

lr_schedulers    = ['', "cosine"][:1]

# Additional “fixed” arguments from your example
label_smoothing = 0 # No label smoothing
dataset_names    = ["texture2shape_miniecoset", "ecoset_square256", "imagenet"][:1]
class_weights_json_path_dict = {'texture2shape_miniecoset': '', 'ecoset_square256': '/share/klab/datasets/optimized_datasets/lookup_ecoset_json.json'}
data_dir        = "/share/klab/datasets" # Directory with your data

# Fixed Development Strategy
img_size        = 256
dev_strategies    = ["evd", "adult"][:1]
time_order_val    = ["normal", "random", 'mid_phase'][0:1]
n_months_list = [300][:1]

#* Sweepable parameters for EVD
months_per_epochs = [2, 1, 4, 3, 0.5, 8, 0.25][-4:]
contrast_thresh = [0.2, 0.1, 0.4][:3]
contrast_spd    = [100, 50, 150][:3]

# Ablation flags
apply_blur_flags = [1][:1]
apply_color_flags = [1][:1]
apply_contrast_flags = [1][:1]

print(f"Generating job configurations in '{cfg}'...")
with open(cfg, "w") as f:
    for seed, lr, lr_scheduler, bs, arch, dataset_name, dev_strategy, time_order_val, n_months, months_per_epoch, contrast_thresh, contrast_spd, apply_blur, apply_color, apply_contrast in product(seeds, learning_rs, lr_schedulers, batch_sizes, architectures, dataset_names, dev_strategies, time_order_val, n_months_list, months_per_epochs, contrast_thresh, contrast_spd, apply_blur_flags, apply_color_flags, apply_contrast_flags):
    # in product(seeds, learning_rs, batch_sizes, architectures):

        # Each job needs its own free port to avoid collision:
        host_port = find_free_port()

        epochs = max(int(n_months / months_per_epoch), 150) # at least 150 ep

        # Build the entire torchrun command for each set of params:
        cmd = [
            "torchrun",
            "--nproc_per_node=1",          # 2 processes per node (2 GPUs)
            "--nnodes=1",                  # single node
            "--node_rank=0",               # node rank
            f"--master_addr=localhost",
            f"--master_port={host_port}",
            "scripts/main.py",             # the actual training script

            f"--arch {arch}",
            f"--lr {lr}",
            # f"--lr-scheduler {lr_scheduler}", #* Not using 
            f"--label-smoothing {label_smoothing}",
            f"--dataset-name {dataset_name}",
            f"--development-strategy {dev_strategy}",
            f"--time-order {time_order_val}",
            f"--months-per-epoch {months_per_epoch}",
            f"--contrast_threshold {contrast_thresh}",
            f"--decrease_contrast_threshold_spd {contrast_spd}",
            f"--epochs {epochs}",
            f"--image-size {img_size}",
            f"--batch-size {bs}",
            f"--seed {seed}",             # example: set seed
            data_dir                       # last positional argument (/share/klab/datasets)
        ]

        # Write the entire command as one line
        f.write(" ".join(cmd) + "\n")

# Count how many lines (jobs) we wrote
job_count = sum(1 for _ in open(cfg))
print(f"Total jobs: {job_count}")

# Build the sbatch command with --array
sbatch_cmd = [
    "sbatch",
    f"--array=1-{job_count}%{nparallel}",
    submit_script,
    cfg
]

print("Submitting SLURM array job...")
print(" ".join(sbatch_cmd))

res = subprocess.run(sbatch_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if res.returncode == 0:
    print(f"Submission successful: {res.stdout.decode()}")
else:
    print(f"Error during submission: {res.stderr.decode()}")
