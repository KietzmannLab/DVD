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
## evaluation
##################################
def compute_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    total_samples = labels.size(0)
    correct_predictions = (predicted == labels).sum().item()
    return 100.0 * correct_predictions / total_samples

def evaluate_model(data_loader, model, criterion, hyp, transform=None, log_polar_transform=False, self_supervised= False):
    total_loss = 0.0
    total_accuracy = 0.0 
    device = hyp['optimizer']['device']
    months_per_epoch = 300 / hyp['optimizer']['n_epochs']

    with torch.no_grad():
        for imgs, labels in data_loader:
            
            model = model.to(device)
            if not hyp['dataset'].get('self_supervised', False):
                imgs, labels = imgs.to(device), labels.to(device)
            else:
                # For self-supervised learning, data might be different
                # For example, data could be (img1, img2)
                imgs[0] = imgs[0].to(device)
                imgs[1] = imgs[1].to(device)

            if log_polar_transform:
                #! Doing log polar transform, Precompute the log-polar grid for the final_size image
                print('Applying log polar transform during evaluation')
                imgs = apply_logpolar_transform(imgs, final_size=hyp['dataset']['image_size'], device=device)

            if transform:
                imgs = transform(imgs)
            
            if device == 'cuda':
                with torch.cuda.amp.autocast():
                    if self_supervised:
                        # Compute self-supervised loss (e.g., contrastive loss)
                        # import pdb; pdb.set_trace()
                        p1, p2, z1, z2 = net(x1=imgs[0], x2=imgs[1])

                        # Normalize to unit vectors
                        p1 = torch.nn.functional.normalize(p1, dim=1)  # Normalize along the feature dimension
                        p2 = torch.nn.functional.normalize(p2, dim=1)
                        z1 = torch.nn.functional.normalize(z1, dim=1)
                        z2 = torch.nn.functional.normalize(z2, dim=1)
                        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
                    else:
                        outputs = model(imgs)
                        loss = criterion(outputs, labels.long())
            else:
                raise ValueError('Device not implemented')
                # outputs = model(imgs)
                # loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            if not self_supervised:
                total_accuracy += compute_accuracy(outputs, labels)
    
    if not self_supervised:
        return total_loss, total_accuracy
    else:
        return total_loss, 0.0


def evaluate_model_shape_bias( model, criterion, batch_size, transform=None, hdf5_file_path= '/share/klab/datasets/cue_conflicts_testset_224px_geirhos.h5', cat_num_samples=75, verbose=True):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    # Use the MiniEcoset dataset
    test_dataset = ShapeTextureDataset('test', hdf5_file_path, transform=transform)
    data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    shape_categories = test_dataset.shape_categories
    texture_categories = test_dataset.texture_categories
    category_labels = test_dataset.categories #* Same as hyp['categories']['imagenet16'] 

    total_shape_acc = []
    total_texture_acc = []
    total_stimuli_shape_bias = []
    category_level_shape_acc = []
    category_level_texture_acc = []
    category_level_shape_bias = []
    num_categories = len(category_labels)
    category_shape_acc = {category: 0 for category in category_labels}
    category_texture_acc = {category: 0 for category in category_labels}
    num_samples = len(test_dataset)
    

    for imgs, shape_labels, texture_labels in data_loader:
        shape_labels = shape_labels.to(device).long() # Convert shape_labels to torch.int64 (torch.long)
        texture_labels = texture_labels.to(device).long()
    
        imgs = imgs.to(device).float()

        if device == 'cuda':
            with torch.cuda.amp.autocast():
                outputs = model(imgs)
                # loss = criterion(outputs, shape_labels)  # Loss based on shape labels
        else:
            outputs = model(imgs)
            # loss = criterion(outputs, shape_labels)  # Loss based on shape labels
        
        _, predicted = torch.max(outputs, 1)

        shape_correct_list = (predicted == shape_labels).tolist()
        texture_correct_list = (predicted == texture_labels).tolist()
        shape_correct_count = sum(shape_correct_list)
        texture_correct_count = sum(texture_correct_list)
        overal_shape_texture_acc = (shape_correct_count + texture_correct_count) / len(predicted)
        overall_stimuli_shape_bais = shape_correct_count / (shape_correct_count + texture_correct_count) if (shape_correct_count + texture_correct_count) > 0 else float('nan')
        total_stimuli_shape_bias.append(overall_stimuli_shape_bais)
        # print(f"Overall Shape Bias: {overall_shape_bais}")

        # category level accuracy & bias (each category has 75 samples in default)
        # import pdb; pdb.set_trace()
        # print(f"categories: {category_labels}")
        for i, category in enumerate(category_labels):
            shape_acc = sum(shape_correct_list[i*cat_num_samples:(i+1)*cat_num_samples])/ cat_num_samples
            texture_acc = sum(texture_correct_list[i*cat_num_samples:(i+1)*cat_num_samples]) / cat_num_samples
            shape_bias = (shape_acc / (shape_acc + texture_acc)) if (shape_acc + texture_acc) > 0 else 0.5 # or float('nan')
            
            category_level_shape_acc.append(shape_acc)
            category_level_texture_acc.append(texture_acc)
            category_level_shape_bias.append(shape_bias)


    median_shape_bias = np.nanmedian(category_level_shape_bias)
    if verbose:
        print(f"Overall Shape Texture Accuracy: {overal_shape_texture_acc}")
        print(f"Overall Stimuli Shape Bias: {overall_stimuli_shape_bais}")
        print(f"Category Level Shape Accuracy: {category_level_shape_bias}")
        print(f"Median Shape Bias: {median_shape_bias} with 95% CI: {np.nanpercentile(category_level_shape_bias, [2.5, 97.5])}")
    # import pdb; pdb.set_trace()
    return overal_shape_texture_acc, overall_stimuli_shape_bais, category_level_shape_bias, median_shape_bias
    

###########################################################################################################
## Training and Evaluation
###########################################################################################################

def evaluate_on_test_set(net, test_loader, criterion, hyp):
    """
    Evaluate the network on the test set and print the accuracy.
    
    """
    net.eval()
    _, test_acc_running = evaluate_model(test_loader, net, criterion, hyp)

    test_acc = test_acc_running / len(test_loader)
    print(f'Test accuracy: {test_acc:.2f}%')

def validate_epoch(model, val_loader, criterion, hyp, log_polar_transform=False, self_supervised=False):
    """
    Validate the network on the validation dataset.
    
    Args:
    - model (torch.nn.Module): Neural network model.
    - val_loader (DataLoader): Validation data loader.
    - criterion (torch.nn.Module): Loss function.
    
    Returns:
    - tuple: Average validation loss and accuracy for the epoch.
    """
    model.eval()
    val_loss, val_acc = evaluate_model(val_loader, model, criterion, hyp, log_polar_transform=log_polar_transform, self_supervised=self_supervised)
     
    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = np.mean(val_acc) / len(val_loader)

    return avg_val_loss, avg_val_acc


