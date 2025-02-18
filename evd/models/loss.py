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

import torch
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
import torch.nn.utils.prune as prune
from torchvision.datasets import ImageFolder


###########################################################################################################
## Training and Evaluation
###########################################################################################################

def get_loss_function(hyp, device):
    """
    Set up the loss function (criterion) based on the dataset and hyperparameters.

    Args:
        hyp (dict): Hyperparameters, including dataset details and class weights.
        device (torch.device): Device to which tensors should be moved.

    Returns:
        torch.nn.Module: The loss function configured based on the dataset and hyperparameters.
    """
    dataset_name = hyp['dataset']['name']
    class_weights_json_path = hyp['dataset'].get('class_weights_json_path')
    is_self_supervised = hyp['network'].get('self_supervised', False)

    def create_criterion_with_weights(weights):
        """Helper to create a loss function with class weights."""
        return torch.nn.CosineSimilarity(dim=1).cuda() if is_self_supervised else torch.nn.CrossEntropyLoss(weight=weights)

    # Handle specific datasets with class weights
    if class_weights_json_path and dataset_name in ['ecoset_square256_patches', 'ecoset_square256']:
        print(f"Loading class weights from {class_weights_json_path}")
        class_weights_dict = class_weights_from_json(class_weights_json_path, normalize='max')
        num_classes = max(class_weights_dict.keys()) + 1
        class_weights_tensor = torch.tensor(
            [class_weights_dict[i] for i in range(num_classes)],
            dtype=torch.float32
        ).to(device)
        return create_criterion_with_weights(class_weights_tensor)

    if dataset_name == 'imagenet':
        imagenet_path = "/share/klab/datasets/imagenet/train"  # Adjust as needed
        print(f"Calculating class weights from {imagenet_path} for ImageNet")
        imagenet_train_dataset = ImageFolder(root=imagenet_path)
        class_weights = calculate_class_weights_from_imagefolder(imagenet_train_dataset).to(device)
        return create_criterion_with_weights(class_weights)

    # Default criterion for other datasets
    return create_criterion_with_weights(None)

############################################################################################################
## utils for data loading and processing
############################################################################################################

def class_weights_from_json(json_path, normalize=None):
    with open(json_path) as file:
        data = json.load(file)
    category_image_count = defaultdict(int)
        
    for k, v in data.items():
        
        zero_indexed_category_label = int(k) - 1
        images_in_category = data[k]["images"]
        category_image_count[zero_indexed_category_label] = int(images_in_category)
        
    max_key = max(category_image_count, key=category_image_count.get)
    min_key = min(category_image_count, key=category_image_count.get)    
    print(f'Max images per class {max(category_image_count.values())} for class label {max_key}, Min images per class {min(category_image_count.values())} for class label {min_key}')
    total_classes = len(data.keys())    
    total_images = sum(category_image_count.values())  
    class_weights = {
        category_label: total_images / (image_count * total_classes)
        for category_label, image_count in category_image_count.items()
    }  

    
    if normalize == 'max':
        max_weight = max(class_weights.values())
        class_weights = {
            category_label: weight / max_weight
            for category_label, weight in class_weights.items()
        }
    elif normalize == 'sum':
        weight_sum = sum(class_weights.values())
        class_weights = {
            category_label: weight / weight_sum
            for category_label, weight in class_weights.items()
        }
    
    return class_weights

def calculate_class_weights_from_imagefolder(dataset):
    """
    Calculate class weights for CrossEntropyLoss based on the dataset loaded with ImageFolder.

    Args:
        dataset (torchvision.datasets.ImageFolder): Dataset loaded using ImageFolder.

    Returns:
        torch.Tensor: Tensor of class weights to use with CrossEntropyLoss.
    """
    # Get the list of labels for all samples in the dataset
    labels = [sample[1] for sample in dataset.samples]

    # Count occurrences of each class
    class_counts = Counter(labels)

    # Get total number of samples
    total_samples = sum(class_counts.values())

    # Calculate class weights: inverse proportional to class frequency
    num_classes = len(class_counts)
    class_weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]

    # Print min and max counts
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    print(f"Minimum count per class: {min_count}")
    print(f"Maximum count per class: {max_count}")

    # Normalize weights (optional, depends on preference)
    class_weights = np.array(class_weights) / sum(class_weights)

    # Convert to a tensor for use in PyTorch
    return torch.tensor(class_weights, dtype=torch.float)

