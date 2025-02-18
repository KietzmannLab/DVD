import os
import re
import cv2
import json
import h5py
import time
import math
import subprocess
import wandb
import argparse
import random
import numpy as np
import scipy.stats as stats
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageChops

import os
import torch
import torchvision
import timm
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from collections import defaultdict
from torchvision import transforms
import torchvision.transforms.functional as F
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset

import torch
import torch.nn as nn
import kornia
import kornia.augmentation as K
import kornia.filters as KF
from torch.utils.data import Subset, DataLoader, Dataset
import random

from evd.datasets.loader import get_transform, Ecoset

class MixedEcosetDataset(Dataset):
    """
    A dataset that mixes two underlying datasets (foveated and original) based on
    a dynamically adjustable foveation probability.
    """
    def __init__(self, foveated_dataset, original_dataset, initial_foveation_prob=1.0):
        super().__init__()
        self.foveated_dataset = foveated_dataset
        self.original_dataset = original_dataset
        self.foveation_prob = initial_foveation_prob

        # Ensure both datasets are fully loaded and of the same length
        self.length = min(len(self.foveated_dataset), len(self.original_dataset))

    def set_foveation_prob(self, prob):
        """Update the probability of sampling from the foveated dataset."""
        self.foveation_prob = prob

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Sample from the foveated or original dataset based on the foveation probability.
        """
        if random.random() < self.foveation_prob:
            return self.foveated_dataset[idx]
        else:
            return self.original_dataset[idx]


def get_mixed_dataset_loaders(
    hyp,
    splits,
    in_memory=True,
    compute_stats=False,
    current_month=0,
    total_months=300,
):
    """
    Load both the original and foveated datasets fully and create a MixedEcosetDataset
    for the training data. Dynamically control the foveation proportion using 
    the contrast sensitivity model.
    """
    # -----------------------------------------
    # Load Original Ecoset Dataset
    # -----------------------------------------
    if 'ecoset_square256' in hyp['dataset']['name']:
        original_dataset_path = f"{hyp['dataset']['dataset_path']}ecoset_square{hyp['dataset']['image_size']}_proper_chunks.h5"
        print(f"Loading ORIGINAL dataset from: {original_dataset_path}")

        train_transform = get_transform(hyp['dataset']['augment'], hyp)
        val_test_transform = get_transform(hyp['dataset']['val_test_augment'], hyp)

        if 'train' in splits:
            original_train_dataset = Ecoset('train', original_dataset_path, train_transform, in_memory=in_memory)
        if 'val' in splits:
            original_val_dataset = Ecoset('val', original_dataset_path, val_test_transform, in_memory=in_memory)
        if 'test' in splits:
            original_test_dataset = Ecoset('test', original_dataset_path, val_test_transform, in_memory=in_memory)

        hyp['dataset']['num_classes'] = 565
    else:
        original_train_dataset = None
        original_val_dataset = None
        original_test_dataset = None

    # -----------------------------------------
    # Load Foveated Ecoset Dataset
    # -----------------------------------------
    if 'ecoset_square256_patches' in hyp['dataset']['name']:
        # patch_dataset_path = f"{hyp['dataset']['dataset_path']}optimized_datasets/megacoset.h5"
        patch_dataset_path = f"{hyp['dataset']['dataset_path']}optimized_datasets/coset.h5"
        print(f"Loading FOVEATED dataset from: {patch_dataset_path}")

        train_transform_patch = get_transform(hyp['dataset']['augment'], hyp)
        val_test_transform_patch = get_transform(hyp['dataset']['val_test_augment'], hyp)

        if 'train' in splits:
            patch_train_dataset = Ecoset('train', patch_dataset_path, train_transform_patch, in_memory=in_memory)
        if 'val' in splits:
            patch_val_dataset = Ecoset('val', patch_dataset_path, val_test_transform_patch, in_memory=in_memory)
        if 'test' in splits:
            patch_test_dataset = Ecoset('test', patch_dataset_path, val_test_transform_patch, in_memory=in_memory)
        
        hyp['dataset']['num_classes'] = 565
    else:
        patch_train_dataset = None
        patch_val_dataset = None
        patch_test_dataset = None

    # -----------------------------------------
    # Combine Datasets for Training
    # -----------------------------------------
    # Infants perfer to fixate and zoom at highest-contrast position initially
    foveation_prob = max(0.0, 1.0 - contrast_sensitivity_development(current_month, age50=4.8 * 12, n=2.1633375920569247))
    print(f"[Current Month: {current_month}] Foveation Prob: {foveation_prob:.3f}")

    train_dataset = None
    if 'train' in splits:
        if original_train_dataset is not None and patch_train_dataset is not None:
            train_dataset = MixedEcosetDataset(patch_train_dataset, original_train_dataset, foveation_prob)
        elif original_train_dataset is not None:
            train_dataset = original_train_dataset
        elif patch_train_dataset is not None:
            train_dataset = patch_train_dataset

    val_dataset = original_val_dataset if 'val' in splits else None
    test_dataset = original_test_dataset if 'test' in splits else None

    # -----------------------------------------
    # Build DataLoaders
    # -----------------------------------------
    batch_size = hyp['optimizer'].get('batch_size', 64)
    num_workers = hyp['dataset'].get('num_workers', 4)

    train_loader = None
    val_loader = None
    test_loader = None

    if train_dataset is not None:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,pin_memory=True, persistent_workers=True )

    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True)

    return train_loader, val_loader, test_loader, hyp


def contrast_sensitivity_development(age_months, age50=4.8 * 12, n=2.1633375920569247):
    """Models the development of contrast sensitivity over age."""
    y_max = (300 ** n) / (300 ** n + age50 ** n)  # Adult-level reference
    return (age_months ** n) / (age_months ** n + age50 ** n) / y_max  # Normalize to [0, 1]