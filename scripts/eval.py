import os
import argparse

import pandas as pd
import torch
import torch.nn as nn

import dvd.utils
import dvd.models.loader
from dvd.models.eval import validate
from dvd.datasets.dataset_loader import get_test_loaders


################################################################################
#                                 CONFIG
################################################################################
# Kept for compatibility; this script only evaluates accuracy.

# Dataset / model
dataset_id = 1  # 0=texture2shape_miniecoset, 1=ecoset, 2=imagenet, 3=imagenet_16, 4=facescrub
DATASET_TRAINED_ON = ["texture2shape_miniecoset", "ecoset_square256","imagenet",][dataset_id]
IMAGE_SIZE = 256  
MODEL_TYPE = "resnet50"
EPOCH = "best"

# Paths
CONFIG_FILE = "./dvd/models/config/config.yaml"

SHARED_WEIGHTS_ROOT = "./logs/open_source_weights/"

MODEL_NAME2PATH = {
    "resnet50_baseline": os.path.join(SHARED_WEIGHTS_ROOT, "resnet50_ecoset_baseline", "weights", f"checkpoint_{EPOCH}.pth"),
    "resnet50_DVD-B": os.path.join(SHARED_WEIGHTS_ROOT, "resnet50_ecoset_DVD_B", "weights", f"checkpoint_{EPOCH}.pth"),
}


################################################################################
#                               SETUP
################################################################################
# Load config and data
config_dict = dvd.utils.load_config(CONFIG_FILE)
args = argparse.Namespace(**config_dict)
args.dataset_name = DATASET_TRAINED_ON
test_loader = get_test_loaders(args)


################################################################################
#                              EVALUATION
################################################################################

for model_name, ckpt_path in MODEL_NAME2PATH.items():
    print(f"Evaluating {model_name} starting from epoch {EPOCH}")

    # Build model and load weights
    args.arch = MODEL_TYPE
    model, _ = dvd.models.loader.create_model(args)

    if os.path.isfile(ckpt_path):
        dvd.models.loader.load_checkpoint(
            model=model, model_path=ckpt_path, log_dir=None, args=None
        )
    else:
        print(f"[WARN] Checkpoint not found: {ckpt_path}")

    # Accuracy
    criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
    top1, top5 = validate(test_loader, model, criterion, EPOCH, image_size=IMAGE_SIZE)

    print(f"{model_name} | epoch {EPOCH} -> top1 {top1:.2f}, top5 {top5:.2f}")