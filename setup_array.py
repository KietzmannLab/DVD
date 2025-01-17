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

host_port = find_free_port()


# Submission settings
submit_script = "submit_task_from_array.sh"  # SLURM script that each array task will run

# Parameters to sweep (change these as needed)
seeds = [1, 2, 3, 4, 5][:1]
batch_sizes = [512, 1024, 2048][1:2] #* 512 170ms while 1024 190 ms (256 160ms slightly faster so 512 is fine), though speed difference 2 time mainly comes from H100 vs A100 | due to losts of transformation larger
architectures = ["resnet50", "alexnet"][:1]

n_months_list = [300][:1]
time_orders = ["", "random", 'mid_phase'][:1]
learning_rates = [ 0.05, 0.1,  1e-3, 5e-3,  1e-5, 1e-4,][-1:]

dataset_types = ['texture2shape_miniecoset', 'ecoset_square256'][:1]#!
class_weights_json_path_dict = {'texture2shape_miniecoset': '', 'ecoset_square256': '/share/klab/datasets/optimized_datasets/lookup_ecoset_json.json'}

development_modes = ['retina_fft_development'][:1]
#! control these now
months_per_epoch_list = [1, 2, 0.5, 4][-1:]
cs_alphas = [ 0.05, 0.025, 0.1][:]
cs_drop_n_monthly_list = [36, 18, 72, 9, 4.5, ][:3]
cs_drop_speeds = [2][:1]

apply_blur_flags = [1][:1]
apply_color_flags = [1][:1]
apply_contrast_flags = [1][:1]


# Control array job execution
nparallel = 2  # Number of parallel jobs
cfg = f"tmp-cfg-{int(time())}.txt"  # Temporary config file for jobs


# Generate job configurations
print("Generating job configurations...")
with open(cfg, "w") as f:
    for seed, batch, arch, n_months, time_order, lr, dataset_type, dev_mode, \
        months_per_epoch, cs_drop_n_monthly, cs_spd, cs_alpha, \
        apply_blur, apply_color, apply_contrast in product(
            seeds, batch_sizes, architectures, n_months_list, time_orders, learning_rates,
            dataset_types, development_modes, months_per_epoch_list, cs_drop_n_monthly_list,
            cs_drop_speeds, cs_alphas, apply_blur_flags, apply_color_flags, apply_contrast_flags
        ):

        # Find a unique free port
        host_port = find_free_port()

        # Calculate n_epochs as an integer
        n_epochs = int(n_months / months_per_epoch)
        class_weights_json_path = class_weights_json_path_dict[dataset_type]
        
        # Create options
        #* --seed 1 --learning_rate 1e-4 --model_name resnet50 --months_per_epoch 1 --contrast_drop_speed_factor 0.05 --using_mixed_grayscale 1 --decrease_contrast_drop_speed_factor_every_n_month 36 --decrease_speed_of_contrast_drop_speed_factor_every_n_month 2 --using_mixed_grayscale 1 --development_strategy retina_fft_development  --dataset texture2shape_miniecoset   --norm_flag 1 --blur_norm_order blur_first --batch_size 512 --n_epochs 600 --end_epoch 600 --id 1
        options = [
            f"--seed {seed}",
            f"--learning_rate {lr}",
            f"--batch_size {batch}",
            f"--model_name {arch}",

            # f"--time_order ''",
            f"--n_epochs {n_epochs}",

            f"--development_strategy {dev_mode}",
            f"--using_mixed_grayscale 1",
            f"--months_per_epoch {months_per_epoch}",
            f"--decrease_contrast_drop_speed_factor_every_n_month {cs_drop_n_monthly}",
            f"--decrease_speed_of_contrast_drop_speed_factor_every_n_month {cs_spd}",
            f"--contrast_drop_speed_factor {cs_alpha}",

            # f"--apply_blur {apply_blur}",
            # f"--apply_color {apply_color}",
            # f"--apply_contrast {apply_contrast}",

            f"--dataset {dataset_type}",
            # f"--class_weights_json_path {class_weights_json_path}", #* class weights json path
            # f"--class_weights_json_path '' ", #* class weights json path


            # f"--dist-url tcp://localhost:{host_port}",
            # "--multiprocessing-distributed",
            # "--world-size 1",
            # "--rank 0",

            # "--lars",
            # "--start-epoch 0",
            # "--resume ''",
            # "/share/klab/datasets/"
        ]
        f.write(" ".join(options) + "\n")

print(f"Configurations saved to {cfg}")

# Submit the SLURM array job
job_count = sum(1 for _ in open(cfg))
cmd = [
    "sbatch",
    f"--array=1-{job_count}%{nparallel}",
    submit_script,
    cfg
]

print("Submitting SLURM array job...")
print("Command:", " ".join(cmd))

res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
if res.returncode == 0:
    print(f"Submission successful: {res.stdout.decode()}")
else:
    print(f"Error during submission: {res.stderr.decode()}")

#! test_more6aug_halfgray_resnet50_linear_no_beta_dn36.0_2.0_mixgray1.0_mpe1.0_alpha0.05beta1.0_id_1_lr_0.0001_none_texture2shape_miniecoset_dev_retina_fft_development_leReLU0.01_colorspd1_egray0_cs_age50_57.6_T__rm1000.0_seed1
#* test_more6aug_halfgray_resnet50_linear_no_beta_dn36.0_2.0_mixgray1.0_mpe1.0_alpha0.05beta1.0_id_1_lr_0.0001_none_texture2shape_miniecoset_dev_retina_fft_development_leReLU0.01_colorspd1_egray0_cs_age50_57.6_T_normal_rm1000.0_seed1