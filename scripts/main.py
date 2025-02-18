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

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp

import torch
import torch.nn as nn
import kornia.augmentation as K
import kornia.filters as KF

import evd.datasets.loader, evd.datasets.mix_loader
import evd.models.evaluate, evd.models.loader, evd.models.loss
import evd.evd.development

torch.backends.cudnn.benchmark = True

##############################
## Hyperparameters
##############################
def get_args():
    """Get command-line arguments."""
    parser = argparse.ArgumentParser(description='Hyperparameters for blurring project')

    parser.add_argument('--model_name', type=str, default='resnet50')

    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--step_size', type=int, default=20) 
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--lr_scheduler', type=str, default='none', help='Learning rate scheduler to use')
    parser.add_argument('--id', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1)
    
    # time_order
    parser.add_argument('--time_order', type=str, default= '')
    
    # age that get contrast sensitivity 50% of adult 
    parser.add_argument('--cs_age50', type=float, default=4.8*12) #* 4.8*12 for 4.8 years

    parser.add_argument('--contrast_threshold_alpha', type=float, default=1.0) #*
    parser.add_argument('--contrast_spd_beta', type=float, default=1.0) #* contrast_spd_beta for contrast increasing speed control
    parser.add_argument('--decrease_contrast_threshold_alpha_every_n_month', type=float, default=1) #*
    parser.add_argument('--decrease_speed_of_contrast_threshold_alpha_every_n_month', type=float, default=0) #* Default 0 means no decrease

    parser.add_argument('--color_development_strategy', type=int, default=0) #*
    parser.add_argument('--color_development_steepness', type=float, default=5) #* # color_development_steepness 
    parser.add_argument('--color_develop_start_epoch', type=int, default=0) #*
    parser.add_argument('--color_develop_lasting_epochs', type=int, default=20) #* # color_develop_lasting_epochs

    # parser.add_argument('--early_gray_scale_flag', type=int, default=0) #* previous default is using blurring_strage
    parser.add_argument('--blurring_strategy', type=str, default= ['sharp','first_few_epochs_exponentially_decreasing'][0])
    
    parser.add_argument('--blurring_start_epoch', type=int, default=0) #*
    parser.add_argument('--blurring_lasting_epochs', type=int, default=20) #*

    parser.add_argument('--blur_norm_order', type=str, default= ['blur_first','norm_first'][0])

    # development_strategy, months_per_epoch
    parser.add_argument('--development_strategy', type=str, default= 'adult')
    parser.add_argument('--months_per_epoch', type=float, default=1.0)

    parser.add_argument('--end_epoch', type=int, default=float('inf')) # IF not remove or assign specific, just inf

    #* apply  blur, color, contrast
    parser.add_argument('--apply_blur', type=int, default=1, help='Flag to apply blur to images')
    parser.add_argument('--apply_color', type=int, default=1, help='Flag to apply color changes')
    parser.add_argument('--apply_contrast', type=int, default=1, help='Flag to apply contrast adjustments')

    parser.add_argument('--dataset', type=str, default= ['ecoset_square256'][0]) # 'texture2shape_miniecoset',
    parser.add_argument('--class_weights_json_path', type=str, default= None) #'/share/klab/datasets/optimized_datasets/lookup_ecoset_json.json')
    parser.add_argument('--pretrained', type=int, default=0)

    parser.add_argument('--grayscale_flag', type=int, default=0) 
    parser.add_argument('--using_mixed_grayscale', type=float, default=1)

    # Contrast interval mode & Light Intnesity Sensitivity
    parser.add_argument('--contrast_interval_mode', type=str, default='fixed')
    parser.add_argument('--starting_mode', type=str, default='linear')


    parser.add_argument('--show_progress_bar', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--batch_size_val_test', type=int, default=256)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--n_epochs', type=int, default=150)

    # control aug str
    parser.add_argument('--control_aug_str', type=str, default='')
    # center crop
    parser.add_argument('--center_crop', type=int, default=0)
    # resize
    parser.add_argument('--resize', type=int, default=0)
    
    return parser.parse_args()

def get_hyp(args):
    """Return hyperparameters as a dictionary."""

    return {
        'dataset': {
            'name': args.dataset,
            'image_size': args.image_size,
            'dataset_path': '/share/klab/datasets/', #'/home/hpczeji1/hpc-work/Codebase/Datasets/',
            'class_weights_json_path': args.class_weights_json_path,
            'augment': {'randomflip', 'grayscale',  'randomrotation', 'equalize', 'perspective', 'globalbrightness', 'globalcontrast', 'blur', 'sharpness', }, 
            'val_test_augment': ['normalize'],
            # 'num_classes': [112,565, 565, 16,1000,118][['texture2shape_miniecoset','ecoset_square256','ecoset_square256_patches','imagenet16','imagenet','facescrub'].index(args.dataset)],
            'grayscale_flag': args.grayscale_flag,
            'control_aug_str': args.control_aug_str,
            'center_crop': args.center_crop,
            'resize': args.resize,
        },
        'network': {
            'model': args.model_name,
            'identifier': f'id_{args.id}_lr_{args.learning_rate}',
            'time_order': args.time_order,

            # development_strategy
            'development_strategy': args.development_strategy,
            'months_per_epoch': args.months_per_epoch,
            'using_mixed_grayscale': args.using_mixed_grayscale,

            #* apply blur, color, contrast
            'apply_blur': args.apply_blur,
            'apply_color': args.apply_color,
            'apply_contrast': args.apply_contrast,

            #* Details of configuration of contrast part
            'cs_age50': args.cs_age50,
            'contrast_interval_mode': args.contrast_interval_mode,
            'contrast_threshold_alpha': args.contrast_threshold_alpha,
            'decrease_contrast_threshold_alpha_every_n_month': args.decrease_contrast_threshold_alpha_every_n_month,
            'decrease_speed_of_contrast_threshold_alpha_every_n_month': args.decrease_speed_of_contrast_threshold_alpha_every_n_month,
            'contrast_spd_beta': args.contrast_spd_beta,
            
            'blur_norm_order': args.blur_norm_order,
            'pretrained': args.pretrained,
            'end_epoch': args.end_epoch,

        },
        'optimizer': {
            'type': 'adam',
            'lr': args.learning_rate,
            'lr_scheduler': args.lr_scheduler,
            'batch_size': args.batch_size,
            'n_epochs': args.n_epochs,
            'device': 'cuda',
            'dataloader': {
                'num_workers_train': 10, # number of cpu workers processing the batches 
                'prefetch_factor_train': 4, # number of batches kept in memory by each worker (providing quick access for the gpu)
                'num_workers_val_test': 3, # do not need lots of workers for val/test
                'prefetch_factor_val_test': 4 
                },
            'show_progress_bar': args.show_progress_bar,
            'seed': args.seed,
            'step_size': args.step_size,
            'gamma': args.gamma,
        },
        'misc': {
            'use_amp': True,
            'batch_size_val_test': args.batch_size_val_test,
            'save_logs': 5,
            'save_net': 5,
            'project_name': "Dec_4thYear_texture2shape_project", #* Saving name for wandb projectlogging

        },
    }



def train_epoch(epoch, net, train_loader, optimizer, criterion, scaler, hyp):
    """
    Train the network for one epoch.

    Args:
        epoch (int): Current epoch number.
        net (torch.nn.Module): Neural network model.
        train_loader (DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (torch.nn.Module): Loss function.
        scaler (torch.amp.GradScaler): Gradient scaler for mixed precision training.
        hyp (dict): Hyperparameters.

    Returns:
        tuple: Average training loss and accuracy for the epoch.
    """

    
    before_start = time.time()
    net.train()
    device = hyp['optimizer']['device']
    net.to(device).float()
    train_loss, train_acc = 0.0, 0.0

    # Generate age months curve to map batches to age months
    age_months_curve = evd.evd.development.generate_age_months_curve(hyp['optimizer']['n_epochs'], len(train_loader), hyp['network']['months_per_epoch'], 
                                                 mid_phase=hyp['network']['time_order'] == 'mid_phase', shuffle=hyp['network']['time_order'] == 'random', seed=hyp['optimizer']['seed'])

    start = time.time()
    # print("time start", time.time()-start)
    for batch_id, (imgs, lbls) in enumerate(train_loader):
        # start = time.time()
        # print("time start", time.time()-start)
        imgs, lbls = imgs.to(device), lbls.to(device)
        imgs = imgs.squeeze(1)  #*  [512, 1, 3, 256, 256] -> [512, 3, 256, 256]

        optimizer.zero_grad()

        # Apply transformations based on development strategy
        age_months = age_months_curve[(epoch - 1) * len(train_loader) + batch_id]

        # Blur, Color, Contrast
        apply_blur = hyp['network']['apply_blur']
        apply_color = hyp['network']['apply_color']
        apply_contrast = hyp['network']['apply_contrast']
        cs_age50=hyp['network']['cs_age50']
        contrast_threshold_alpha = hyp['network']['contrast_threshold_alpha']
        decrease_contrast_threshold_alpha_every_n_month = hyp['network']['decrease_contrast_threshold_alpha_every_n_month'] 
        decrease_speed_of_contrast_threshold_alpha_every_n_month = hyp['network']['decrease_speed_of_contrast_threshold_alpha_every_n_month']

        # Apply transformations across time in development
        if hyp["network"]["development_strategy"] == 'evd':
            contrast_spd_control_coeff = max(math.floor(age_months / decrease_contrast_threshold_alpha_every_n_month)* decrease_speed_of_contrast_threshold_alpha_every_n_month, 1) # need to larger than 1
            imgs = evd.evd.development.EarlyVisualDevelopmentTransformer().apply_fft_transformations(imgs, apply_blur, apply_color, apply_contrast, age_months, contrast_threshold_alpha = contrast_threshold_alpha/contrast_spd_control_coeff, cs_age50=cs_age50,  image_size=hyp['dataset']['image_size'], verbose=False)
        elif hyp["network"]["development_strategy"] == 'adult':
            pass                              
        else:
            raise ValueError(f"Unknown development strategy: {hyp['network']['development_strategy']}")

        # Compute the forward pass and loss.
        if hyp['optimizer']['device'] == 'cuda':
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=hyp['misc']['use_amp']):
                outputs = net(imgs)
                if isinstance(outputs, list):
                    loss = criterion(outputs[0], lbls.long())
                    if len(outputs) > 1:
                        for t in range(len(outputs)-1):
                            loss = loss + criterion(outputs[t+1], lbls.long())
                    loss = loss/len(outputs)
                else:
                    loss = criterion(outputs, lbls.long())
        else:
            raise ValueError("Invalid device")
        
        # Backward pass and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update metrics
        train_loss += loss.item()
        if isinstance(outputs, list):
            train_acc += np.mean(evd.models.evaluate.compute_accuracy(outputs, lbls))
        else:
            train_acc += evd.models.evaluate.compute_accuracy(outputs, lbls)
 
        if hyp['optimizer']['show_progress_bar']:
            print(f'Training Epoch {epoch}: Batch {batch_id} of {len(train_loader)}', end="\r")
        

    avg_train_loss = train_loss / len(train_loader)
    avg_train_acc = train_acc / len(train_loader)
    
    print(f'\nEpoch {epoch} completed in {time.time() - start:.2f} seconds')
    return avg_train_loss, avg_train_acc




if __name__ == '__main__':

    # Get the hyperparameters
    args = get_args()
    hyp = get_hyp(args)

    # Ensure reproducibility
    torch.manual_seed(hyp['optimizer']['seed'])
    np.random.seed(hyp['optimizer']['seed'])
    random.seed(hyp['optimizer']['seed'])

    # Load datasets
    if hyp['dataset']['name'] == 'ecoset_square256+ecoset_square256_patches':
        print('Loading mixed dataset...')
        train_loader, val_loader, test_loader, hyp = evd.datasets.mix_loader.get_mixed_dataset_loaders(
                                                                                                hyp,
                                                                                                ['train', 'val', 'test'],
                                                                                                in_memory=True,
                                                                                                compute_stats=False,
                                                                                                current_month=0,
                                                                                                total_months=300,
                                                                                            )
    else:
        train_loader, val_loader, test_loader, hyp = evd.datasets.loader.get_dataset_loaders(hyp, ['train', 'val', 'test'])

    # Initialize network and optimizer
    net, net_name = evd.models.loader.get_network_model(hyp)
        
    optimizer =  evd.models.loader.get_optimizer(hyp, net)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Set up the scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=hyp['misc'].get('use_amp', False))

    # Set up the loss function --> differnt loss function for different dataset | class weights
    criterion = evd.models.loss.get_loss_function(hyp, device)

    # Save initial learning rates in optimizer param groups
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = param_group['lr']
    # Set up the learning rate scheduler
    lr_scheduler =  evd.models.loader.get_lr_scheduler(optimizer, hyp['optimizer'].get('lr_scheduler'), hyp)

    # Initialize Weights & Biases (WandB) for experiment tracking
    wandb.init(
        project=hyp['misc'].get('project_name', "Dec_4thYear_Blurring_project"),
        name=net_name,
        config=hyp
    )

    # Initialize or resume training
    net, logs, start_epoch, log_path, net_path = evd.models.loader.initialize_or_resume_training(net, net_name, hyp)
    net = net.float().to(device)

    # Training loop
    n_epochs = hyp['optimizer']['n_epochs']
    end_epoch = hyp['network'].get('end_epoch', float('inf'))
    best_val_acc = 0.0

    for epoch in range(start_epoch, n_epochs + 1):
        # Early stopping condition
        if epoch > end_epoch:
            break

        # Update the current epoch in hyperparameters
        hyp["epoch"] = epoch

        # Training phase
        train_loss, train_acc = train_epoch(
            epoch, net, train_loader, optimizer, criterion, scaler, hyp
        )

        # Validation phase
        val_loss, val_acc =  evd.models.evaluate.validate_epoch(net, val_loader, criterion, hyp)
        if val_acc > best_val_acc:
            best_val_acc = val_acc

        # Step the learning rate scheduler
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Log metrics
        logs['train_losses'][epoch] = train_loss
        logs['val_losses'][epoch] = val_loss
        logs['train_accuracies'][epoch] = train_acc
        logs['val_accuracies'][epoch] = val_acc

        # Save metrics and model checkpoints
        evd.models.loader.log_and_save_metrics(
            epoch, logs, net, optimizer, log_path, net_path, net_name, hyp, is_best=(val_acc == best_val_acc)
        )

    print('\nTraining completed! Evaluating on the test set...\n')

    # Save the final model checkpoint if not already saved
    final_model_path = f'{net_path}/{net_name}_epoch_{epoch}.pth'
    if not os.path.exists(final_model_path):
        torch.save(net.state_dict(), final_model_path)

    # Evaluate the model on the test set
    evd.models.evaluate.evaluate_on_test_set(net, test_loader, criterion, hyp)

    print('\nTraining and evaluation completed successfully!\n')








