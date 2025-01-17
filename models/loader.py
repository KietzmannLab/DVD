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


##################################
## Importing the network
##################################
def get_network_model(hyp):
    """Create a network model based on provided hyperparameters."""


    # Hyperparameter extraction
    model_type = hyp['network']['model']
    dataset_name = hyp['dataset']['name']
    identifier = hyp['network']['identifier']
    blurring_strategy = hyp['network']['blurring_strategy']
    # blurring_decay_steepness = hyp['network']['blurring_decay_steepness']
    blurring_start_epoch = hyp['network']['blurring_start_epoch']
    blurring_lasting_epochs = hyp['network']['blurring_lasting_epochs']
    mixed_chance_blur = hyp['network']['mixed_chance_blur']
    mixed_chance_blur_start_epoch = hyp['network']['mixed_chance_blur_start_epoch']

    color_development_strategy = hyp['network']['color_development_strategy']
    color_development_steepness = hyp['network']['color_development_steepness']
    color_develop_start_epoch = hyp['network']['color_develop_start_epoch']
    color_develop_lasting_epochs = hyp['network']['color_develop_lasting_epochs']
    using_mixed_grayscale = hyp['network']['using_mixed_grayscale']
    
    byte_flag = hyp['network']['byte_flag']
    byte_start_epoch = hyp['network']['byte_start_epoch']
    byte_lasting_epochs = hyp['network']['byte_lasting_epochs']
    byte_steepness = hyp['network']['byte_steepness']
    using_mixed_byte = hyp['network']['using_mixed_byte']
    epoch_to_remove_byte = hyp['network']['epoch_to_remove_byte']
    contrast_drop_speed_factor = hyp['network']['contrast_drop_speed_factor'] # DropSpeed_{contrast_drop_speed_factor}
    mean_freq_factor=hyp['network']['mean_freq_factor']
    std_freq_factor=hyp['network']['std_freq_factor']

    decrease_n_months = hyp['network']['decrease_contrast_drop_speed_factor_every_n_month']
    decrease_n_months_speed = hyp['network']['decrease_speed_of_contrast_drop_speed_factor_every_n_month']
    contrast_thres_map_mode = hyp['network']['contrast_thres_map_mode']
    contrast_spd_beta = hyp['network']['contrast_spd_beta']

    num_classes = hyp['dataset']['num_classes']
    seed = hyp['optimizer']['seed']
    lr_scheduler = hyp['optimizer']['lr_scheduler']

    # Additional parameters for SimSiam if needed
    self_supervised = hyp['dataset'].get('self_supervised', False)
    simsiam_dim = hyp['network'].get('simsiam_dim', 2048)
    simsiam_pred_dim = hyp['network'].get('simsiam_pred_dim', 512)

    # Generate a concise, structured model name
    # model_name = (
    #     f"test_1stDec_halfgray_{model_type}_{contrast_thres_map_mode}_"
    #     f"dn{decrease_n_months}_{decrease_n_months_speed}_"
    #     f"mixgray{using_mixed_grayscale}_"
    #     f"mpe{hyp['months_per_epoch']}_"
    #     f"_alpha{contrast_drop_speed_factor}beta{contrast_spd_beta}_"
    #     f"id{identifier}_{dataset_name}_"
    #     f"dev_{hyp['development_strategy']}_"
    #     f"leReLU{hyp['csf_leaky_relu_slope']}_"
    #     f"colorspd{hyp['color_speed_factor']}_"
    #     f"egray{hyp['extra_grayscale_end_months']}_"
    #     f"cs_age50_{hyp['cs_age50']:.1f}_"
    #     f"T_{hyp['time_order']}_"
    #     f"rm{epoch_to_remove_byte}_"
    #     f"seed{seed}"
    # )
    # model_name = f'test_more7aug_halfgray_{model_type}_{contrast_thres_map_mode}_dn{decrease_n_months}_{decrease_n_months_speed}_mixgray{using_mixed_grayscale}_mpe{hyp["network"]["months_per_epoch"]}_alpha{contrast_drop_speed_factor}beta{contrast_spd_beta}_{identifier}_{lr_scheduler}_{dataset_name}_dev_{hyp["network"]["development_strategy"]}_leReLU{hyp["network"]["csf_leaky_relu_slope"]}_colorspd{hyp["network"]["color_speed_factor"]}_egray{hyp["network"]["extra_grayscale_end_months"]}_cs_age50_{hyp["network"]["cs_age50"]:.1f}_T_{hyp["network"]["time_order"]}_rm{epoch_to_remove_byte}_seed{seed}'
    if self_supervised:
        model_name = f'ssl1_{model_type}_{contrast_thres_map_mode}_dn{decrease_n_months}_{decrease_n_months_speed}_mixgray{using_mixed_grayscale}_mpe{hyp["network"]["months_per_epoch"]}_alpha{contrast_drop_speed_factor}beta{contrast_spd_beta}_{identifier}_{lr_scheduler}_{dataset_name}_dev_{hyp["network"]["development_strategy"]}_leReLU{hyp["network"]["csf_leaky_relu_slope"]}_colorspd{hyp["network"]["color_speed_factor"]}_egray{hyp["network"]["extra_grayscale_end_months"]}_cs_age50_{hyp["network"]["cs_age50"]:.1f}_T_{hyp["network"]["time_order"]}_rm{epoch_to_remove_byte}_seed{seed}'
    else:
        model_name = f'test_more6aug_halfgray_{model_type}_{contrast_thres_map_mode}_dn{decrease_n_months}_{decrease_n_months_speed}_mixgray{using_mixed_grayscale}_mpe{hyp["network"]["months_per_epoch"]}_alpha{contrast_drop_speed_factor}beta{contrast_spd_beta}_{identifier}_{lr_scheduler}_{dataset_name}_dev_{hyp["network"]["development_strategy"]}_leReLU{hyp["network"]["csf_leaky_relu_slope"]}_colorspd{hyp["network"]["color_speed_factor"]}_egray{hyp["network"]["extra_grayscale_end_months"]}_cs_age50_{hyp["network"]["cs_age50"]:.1f}_T_{hyp["network"]["time_order"]}_rm{epoch_to_remove_byte}_seed{seed}'
    

    # Define model initialization logic
    def configure_resnet50():
        model = models.resnet50(weights=None)
        # if hyp['network']['pretrained']:
        #     pretrained_path = hyp['network']['pretrained']
        #     model.fc = nn.Linear(2048, hyp['dataset']['num_classes'])
        #     model.load_state_dict(torch.load(pretrained_path)['model_state_dict'])
        if hyp['network']['byte_flag'] in ['on_off_center', 'byte_plus_imgs', '6_channel']:
            model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif hyp['network']['byte_flag'] == 'byte_byte_plus_imgs':
            model.conv1 = nn.Conv2d(9, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(2048, num_classes)
        return model

    def raise_unknown_model_error(model_type):
        raise ValueError(f"Unknown model: {model_type}")

    # Map model types to their respective initialization logic
    model_map = {
        'resnet50': configure_resnet50,
        'alexnet': lambda: models.alexnet(weights=None),
        'vgg16': lambda: models.vgg16(pretrained=False),
        'vgg19': lambda: models.vgg19(pretrained=False),
        'resnet18': lambda: models.resnet18(pretrained=False),
        'vit_small': lambda: timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=num_classes),
        'vit_base': lambda: timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes),
        'vit_large': lambda: timm.create_model('vit_large_patch16_224', pretrained=False, num_classes=num_classes),
    }

    # Initialize the model
    if self_supervised:
        # Build a SimSiam model
        # We will assume for simplicity that you use a ResNet-50 as the base encoder for SimSiam
        # If you need other encoders, you can add conditions here
        base_encoder = configure_resnet50_base_encoder
        model = simsiam.builder.SimSiam(base_encoder, dim=simsiam_dim, pred_dim=simsiam_pred_dim)
        print("Initialized SimSiam model for self-supervised learning.")
    else:
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

    return model, model_name



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
    """Load the latest checkpoint if available."""
    # Get the list of model files in the folder if available
    model_files = [f for f in os.listdir(net_path) if f.endswith(".pth")]
    if not model_files:
        #TODO fix the log & net conflict problem
        model_files = [f for f in os.listdir(net_path) if f.endswith(".pt")]
        if not model_files:
            print(f"No checkpoints found for {net_name}!")
            return net, None, 1

    # Sort the model files by epoch number and Get the latest model file
    model_files.sort(key=lambda f: int(f.split("_")[-1].split(".")[0]))
    latest_model_file = model_files[-1]
    
    # Load log data
    try:
        log_prefix = f'/loss_ep{hyp["optimizer"]["n_epochs"]}_'
        data = np.load(log_path + log_prefix + net_name + '.npz')
    except:
        log_prefix = f'/loss_'
        data = np.load(log_path + log_prefix + net_name + '.npz')
    
    
    logs = {
        key: (list(data[key]) if np.ndim(data[key]) > 0 else [data[key]]) + [default_val] * (hyp['optimizer']['n_epochs'] - len(data[key]) + 1)
        for key, default_val in [
            ("train_losses", 0),  #! else train loss no plural
            ("val_losses", 0), #! else  no plural
            ("train_accuracies", 0), 
            ("val_accuracies", 0), 
            ("lrs", hyp['optimizer']['lr'])
        ]
    }
    data.close()


    # Load the model and optimizer and Load the optimizer state dict if available
    checkpoint = torch.load(os.path.join(net_path, latest_model_file))
    net.load_state_dict(checkpoint.get('model_state_dict', checkpoint))

    #TODO double check if optimizer = get_optimizer(hyp, net) correctly
    optimizer = get_optimizer(hyp, net)
    # import pdb; pdb.set_trace()
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    move_optimizer_to_device(optimizer, hyp['optimizer']['device'])  # Assuming this function is defined elsewhere


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
        "overall_shape_texture_acc": [0 for _ in range(epochs+1)],
        "overall_stimuli_shape_bais": [0 for _ in range(epochs+1)],
        "median_shape_bias": [0 for _ in range(epochs+1)],
        "avg_val_blurring_acc": [0 for _ in range(epochs+1)],
        **{f'{cat_name}_shape_bias': [0 for _ in range(epochs+1)] for cat_name in hyp['categories']['imagenet16']}
    }

def log_and_save_metrics(epoch,logs, net, optimizer, log_path, net_path, net_name, hyp):
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

    # Save model
    if (epoch + 1) % hyp['misc']['save_net'] == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_history': train_loss,
            'val_loss_history': val_loss
        }, f'{net_path}/{net_name}_epoch_{epoch}.pth') #! net_path or log_path? depends on 

##################################
# move optimizer to cuda    (do we really need this?)
##################################
def move_optimizer_to_device(optim, device):
    """Move optimizer state to a device."""
    for state in optim.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)