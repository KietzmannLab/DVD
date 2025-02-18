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

from evd.models.blt import BLT_VS

##################################
## Importing the network
##################################
def get_network_model(hyp):
    """Create a network model based on provided hyperparameters."""


    # Hyperparameter extraction
    model_type = hyp['network']['model']
    dataset_name = hyp['dataset']['name']
    identifier = hyp['network']['identifier']
    using_mixed_grayscale = hyp['network']['using_mixed_grayscale']
    time_order = hyp["network"]["time_order"]
    
    development_strategy = hyp["network"]["development_strategy"]
    contrast_threshold_alpha = hyp['network']['contrast_threshold_alpha'] # DropSpeed_{contrast_threshold_alpha}
    decrease_n_months = hyp['network']['decrease_contrast_threshold_alpha_every_n_month']
    decrease_n_months_speed = hyp['network']['decrease_speed_of_contrast_threshold_alpha_every_n_month']
    contrast_spd_beta = hyp['network']['contrast_spd_beta']

    apply_blur = hyp['network']['apply_blur']
    apply_color = hyp['network']['apply_color']
    apply_contrast = hyp['network']['apply_contrast']

    lr_scheduler = hyp['optimizer']['lr_scheduler']
    num_classes = hyp['dataset']['num_classes']
    seed = hyp['optimizer']['seed']

    # Additional parameters for SimSiam if needed
    self_supervised = hyp['dataset'].get('self_supervised', False)
    simsiam_dim = hyp['network'].get('simsiam_dim', 2048)
    simsiam_pred_dim = hyp['network'].get('simsiam_pred_dim', 512)
    
    net_name = f'Feb_{model_type}_mpe{hyp["network"]["months_per_epoch"]}_alpha{contrast_threshold_alpha}_dn{decrease_n_months}_{decrease_n_months_speed}_{dataset_name}_mgray{using_mixed_grayscale}_{identifier}_{lr_scheduler}_dev_{development_strategy}_b{apply_blur}c{apply_color}cs{apply_contrast}_T_{time_order}_seed_{seed}'
    #* Further adjust the net_name for control conditions
    net_name = hyp['dataset']['control_aug_str']+ net_name if hyp['dataset']['control_aug_str'] else net_name # if more aug
    net_name = "cp_" + net_name if hyp['dataset']['center_crop'] else net_name
    net_name = net_name.replace('ecoset_square256+ecoset_square256_patches', 'ecoset+patches') if hyp['dataset']['name'] == 'ecoset_square256+ecoset_square256_patches' else net_name

    # Map model types to their respective initialization logic
    model_map = {
        'resnet50': lambda: models.resnet50(weights=None),
        'alexnet': lambda: models.alexnet(weights=None),
        'vgg16': lambda: models.vgg16(pretrained=False),
        'vgg19': lambda: models.vgg19(pretrained=False),
        'resnet18': lambda: models.resnet18(pretrained=False),
        'vit_small': lambda: timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=num_classes),
        'vit_base': lambda: timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes),
        'vit_large': lambda: timm.create_model('vit_large_patch16_224', pretrained=False, num_classes=num_classes),
        'blt_vs': lambda: BLT_VS(num_classes=num_classes)#, add_feats=50)
    }
    def raise_unknown_model_error(model_type):
        raise ValueError(f"Unknown model: {model_type}")

    # Initialize the model
    model = model_map.get(model_type, lambda: raise_unknown_model_error(model_type))()
    # Customize the final layer if necessary
    if 'resnet' in model_type:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif 'vgg' in model_type or model_type == 'alexnet':
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif 'vit' in model_type:
        model.head = nn.Linear(model.head.in_features, num_classes)

    # Print trainable parameter count
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nThe network has {num_trainable_params} trainable parameters\n")

    return model, net_name



##################################
## Initializing the network and optimizer and scheduler
##################################
def initialize_weights(module):
    """Xavier weight initialization for given module."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)

def get_optimizer(hyp, model):
    """Return the optimizer based on hyperparameters."""
    if hyp['optimizer']['type'] == 'adam':
        return optim.Adam(model.parameters(), lr=hyp['optimizer']['lr'])
    else:
        raise ValueError(f"Unknown optimizer: {hyp['optimizer']['type']}")

def get_lr_scheduler(optimizer, scheduler_name, hyp):
    if scheduler_name == 'cosine':
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=hyp['optimizer']['n_epochs'])
    elif scheduler_name == 'step':
        return lr_scheduler.StepLR(optimizer, step_size=hyp['optimizer']['step_size'], gamma=hyp['optimizer']['gamma'])
    elif scheduler_name == 'warmup_cosine':
        return WarmupCosineAnnealingLRScheduler(optimizer, months_per_epoch= hyp['network']['months_per_epoch'], peak_age= 500 / 30, n_epochs= hyp['optimizer']['n_epochs'])
    elif scheduler_name == 'visual_acuity_lr_scheduler':
        return VisualAcuityLRScheduler(optimizer, months_per_epoch= hyp['network']['months_per_epoch'] )
    elif scheduler_name == 'synaptic_density_lr_scheduler':
        return SynapticDensityLRScheduler(optimizer, months_per_epoch= hyp['network']['months_per_epoch'] )
    else:
        return None


class WarmupCosineAnnealingLRScheduler(_LRScheduler):
    def __init__(self, optimizer, months_per_epoch, peak_age, n_epochs, last_epoch: int = 0):
        self.months_per_epoch = months_per_epoch
        self.peak_age = peak_age
        self.n_epochs = n_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        age_months = torch.tensor((self.last_epoch - 1) * self.months_per_epoch, dtype=torch.float32) # Convert epoch to months (epochs start from 1, so -1)
        
        if age_months < self.peak_age:
            # Linear warm-up phase before peak_age
            slope = 1.0 / self.peak_age
            learning_rate = slope * age_months
        else:
            # Cosine annealing after peak_age
            progress = (age_months - self.peak_age) / max(1, (self.n_epochs * self.months_per_epoch) - self.peak_age)
            learning_rate = 0.5 * (1 + torch.cos(torch.pi * progress))
        
        return [base_lr * learning_rate for base_lr in self.base_lrs]



##############################
## Logging functions
##############################
def ensure_directory_exists(directory_name):
    """Ensure the directory exists; if not, create it."""
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
        print(f'{directory_name} is created!')

def setup_logging_directories(net_name):
    """Set up log directories and return their paths."""

    print('Setting up log folders...')

    base_log_dir = 'logs/perf_logs'
    base_net_dir = 'logs/net_params'
    
    ensure_directory_exists(base_log_dir)
    ensure_directory_exists(base_net_dir)

    log_directory = os.path.join(base_log_dir, net_name)
    net_directory = os.path.join(base_net_dir, net_name)

    ensure_directory_exists(log_directory)
    ensure_directory_exists(net_directory)

    return log_directory, net_directory 


##########################
## Loading the checkpoint and initializing or resuming  the logs
##########################
def load_checkpoint(net, net_name, net_path, log_path, hyp):
    """Load the latest checkpoint if available. If a checkpoint file with 'last'
    in its name exists, load that checkpoint; otherwise load the checkpoint
    with the highest epoch number in the filename."""
    import os
    import numpy as np
    import torch

    # First check for a checkpoint with "last" in its name
    last_files = [f for f in os.listdir(net_path) if f.endswith(".pth") and "last" in f]
    if last_files and len(os.listdir(net_path)) > 2: # more than best and last
        latest_model_file = last_files[0]
        print(f"Loading 'last' checkpoint for {net_name}: {latest_model_file}")
    else:
        # Get the list of model files excluding any with 'best' or 'last'
        model_files = [f for f in os.listdir(net_path) if f.endswith(".pth") and 'best' not in f and 'last' not in f]
        if not model_files:
            print(f"No checkpoints found for {net_name}!") 
            return net, None, 1

        # Sort the model files by epoch number and select the latest one.
        model_files.sort(key=lambda f: int(f.split("_")[-1].split(".")[0]))
        latest_model_file = model_files[-1]
        print(f"Loading checkpoint for {net_name} from epoch: {latest_model_file}")

    # Load log data
    data = np.load(os.path.join(log_path, 'loss_' + net_name + '.npz'))

    logs = {
        key: (list(data[key]) if np.ndim(data[key]) > 0 else [data[key]]) +
             [default_val] * (hyp['optimizer']['n_epochs'] - len(data[key]) + 1)
        for key, default_val in [
            ("train_losses", 0), 
            ("val_losses", 0), 
            ("train_accuracies", 0), 
            ("val_accuracies", 0), 
            ("lrs", hyp['optimizer']['lr'])
        ]
    }
    data.close()

    # Load the checkpoint
    checkpoint_path = os.path.join(net_path, latest_model_file)
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint.get('model_state_dict', checkpoint))

    # Load the optimizer state if it exists
    optimizer = get_optimizer(hyp, net)
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        move_optimizer_to_device(optimizer, hyp['optimizer']['device'])  # Ensure the optimizer is on the correct device

    # Determine the starting epoch
    if "last" in latest_model_file:
        # Use the epoch saved in the checkpoint if available
        start_epoch = checkpoint.get('epoch', 1)
    else:
        start_epoch = int(latest_model_file.split("_")[-1].split(".")[0])

    return net, logs, start_epoch

def initialize_or_resume_training(net, net_name, hyp):
    """
    Initialize or resume training based on the presence of a checkpoint.
    
    Args:
    - net (torch.nn.Module): The network model.
    - net_name (str): The name of the network.
    - hyp (dict): Hyperparameters.
    
    Returns:
    - logs (dict): Logs dictionary.
    - start_epoch (int): The epoch to start/resume training from.
    """

    # Create folders for logging
    log_path, net_path = setup_logging_directories(net_name)
    print(f'Log_folders: {log_path} -- {net_path}')

    # Attempt to load a checkpoint if available
    net, logs, start_epoch = load_checkpoint(net, net_name, net_path, log_path, hyp)
    
    if not logs:
        logs = initialize_logs(hyp)
        start_epoch = 1
        net.apply(initialize_weights)
        print('\nTraining begins here from scratch!\n')
    else:
        print(f'\nTraining resumes from epoch {start_epoch}.\n')


    return net, logs, start_epoch, log_path, net_path


def initialize_logs(hyp):
    """
    Initialize logs for a fresh training.
    """
    epochs = hyp['optimizer']['n_epochs']
    
    return {
        "train_losses": [0 for _ in range(epochs+1)],
        "val_losses": [0 for _ in range(epochs+1)],
        "train_accuracies": [0 for _ in range(epochs+1)],
        "val_accuracies": [0 for _ in range(epochs+1)],
        "lrs": [hyp['optimizer']['lr'] for _ in range(epochs+1)],
        # "overall_shape_texture_acc": [0 for _ in range(epochs+1)],
        # "overall_stimuli_shape_bais": [0 for _ in range(epochs+1)],
        # "median_shape_bias": [0 for _ in range(epochs+1)],
        # "avg_val_blurring_acc": [0 for _ in range(epochs+1)],
        # **{f'{cat_name}_shape_bias': [0 for _ in range(epochs+1)] for cat_name in hyp['categories']['imagenet16']}
    }

def log_and_save_metrics(epoch,logs, net, optimizer, log_path, net_path, net_name, hyp, is_best=False):
    """
    Log the training and validation metrics, and save the model and logs periodically.
    
    Args:
    - epoch (int): The current epoch.
    - train_loss (float): Training loss for the current epoch.
    - train_acc (float): Training accuracy for the current epoch.
    - val_loss (float): Validation loss for the current epoch.
    - val_acc (float): Validation accuracy for the current epoch.
    - net (torch.nn.Module): The network model.
    - optimizer (torch.optim.Optimizer): The optimizer.
    - log_path (str): Path to the logging directory.
    - net_name (str): The name of the network.
    - hyp (dict): Hyperparameters.
    """

    train_loss = logs["train_losses"][epoch]
    val_loss = logs["val_losses"][epoch]
    train_acc = logs["train_accuracies"][epoch]
    val_acc = logs["val_accuracies"][epoch]
    lrs = logs["lrs"][epoch]

    # Print metrics
    print(f'Train loss: {train_loss:.2f}; acc: {train_acc:.2f}%')
    print(f'Val loss: {val_loss:.2f}; acc: {val_acc:.2f}%\n')

    # Log metrics to wandb
    wandb.log({
        "train_loss": train_loss, 
        "val_loss": val_loss,
        "train_acc": train_acc,
        "val_acc": val_acc,
        "lr": optimizer.param_groups[0]['lr'],

    })
    # Save logs
    if (epoch + 1) % hyp['misc']['save_logs'] == 0:
        # np.savez(log_path + f'/loss_ep{hyp["optimizer"]["n_epochs"]}_' + net_name + '.npz', 
        np.savez(log_path + f'/loss_' + net_name + '.npz',
                 train_losses =logs["train_losses"], 
                 val_losses =logs["val_losses"], 
                 train_accuracies=logs["train_accuracies"], 
                 val_accuracies=logs["val_accuracies"], 
                 lrs=logs["lrs"],
                 )

    # Save specific epoch model
    if (epoch + 1) % hyp['misc']['save_net'] == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_history': train_loss,
            'val_loss_history': val_loss
        }, f'{net_path}/{net_name}_epoch_{epoch}.pth') 
    
    # Save last epoch model
    torch.save({
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss_history': train_loss,
        'val_loss_history': val_loss
    }, f'{net_path}/{net_name}_last.pth')
    
    # Save the best model
    if is_best:
        print(f'Saving the best model with val_acc {val_acc:.2f} at epoch {epoch}...')
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_history': train_loss,
            'val_loss_history': val_loss
        }, f'{net_path}/{net_name}_best.pth')

##################################
# move optimizer to cuda    (do we really need this?)
##################################
def move_optimizer_to_device(optim, device):
    """Move optimizer state to a device."""
    for state in optim.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)