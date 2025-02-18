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
    
    # if BLT
    if isinstance(outputs, list):
        timesteps = len(outputs)
        accuracies = [0 for t in range(timesteps)]
        
        for t in range(timesteps):
            _, predicted = torch.max(outputs[t].data, 1)
            total = labels.shape[0]
            correct = (predicted == labels).sum().item()
            accuracies[t] = correct*100./total
        
        return accuracies 
    else:
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
                # For self-supervised learning, data might be different; For example, data could be (img1, img2)
                imgs[0] = imgs[0].to(device)
                imgs[1] = imgs[1].to(device)

            if transform:
                imgs = transform(imgs)
            
            #* squeeze imgs to ensure in shape (batch_size, channels, height, width)
            imgs = imgs.squeeze()
            if device == 'cuda':
                with torch.cuda.amp.autocast():
                    if self_supervised:
                        # Compute self-supervised loss (e.g., contrastive loss)
                        p1, p2, z1, z2 = model(x1=imgs[0], x2=imgs[1])
                        # Normalize to unit vectors
                        p1 = torch.nn.functional.normalize(p1, dim=1)  # Normalize along the feature dimension
                        p2 = torch.nn.functional.normalize(p2, dim=1)
                        z1 = torch.nn.functional.normalize(z1, dim=1)
                        z2 = torch.nn.functional.normalize(z2, dim=1)
                        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
                    else:
                        outputs = model(imgs)
                        
                        # For models have multiple timesteps
                        if isinstance(outputs, list):
                            loss = criterion(outputs[0], labels.long())
                            if len(outputs) > 1:
                                for t in range(len(outputs)-1):
                                    loss = loss + criterion(outputs[t+1], labels.long())
                            loss = loss/len(outputs)
                        else:
                            loss = criterion(outputs, labels.long())
            else:
                raise ValueError('Device not implemented')
            
            total_loss += loss.item()
            if not self_supervised:
                if isinstance(outputs, list):
                    total_accuracy += np.mean(compute_accuracy(outputs, labels))
                else:
                    total_accuracy += compute_accuracy(outputs, labels)
    
    if not self_supervised:
        return total_loss, total_accuracy
    else:
        return total_loss, 0.0
    

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


