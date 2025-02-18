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

import torch
import torch.nn as nn
import kornia
import kornia.augmentation as K
import kornia.filters as KF

# Loading the dataset loaders
def get_dataset_loaders(hyp, splits, in_memory =True, compute_stats=False, self_supervised=False):
    """Return train, validation, and test dataloaders based on given hyperparameters."""
    if hyp['dataset']['name'] == 'texture2shape_miniecoset':
        dataset_path = f"{hyp['dataset']['dataset_path']}texture2shape_miniecoset_{hyp['dataset']['image_size']}px.h5"
        print(f"Loading dataset from: {dataset_path}")

        train_transform = get_transform(hyp['dataset']['augment'], hyp)
        val_test_transform = get_transform(hyp['dataset']['val_test_augment'], hyp)                                    
        if self_supervised:
            train_transform = TwoCropsTransform(train_transform)
            val_test_transform = TwoCropsTransform(val_test_transform)

        train_dataset = Ecoset('train', dataset_path, train_transform, in_memory=in_memory) if 'train' in splits else None
        val_dataset = Ecoset('val', dataset_path, val_test_transform, in_memory=in_memory) if 'val' in splits else None
        test_dataset = Ecoset('test', dataset_path, val_test_transform, in_memory=in_memory) if 'test' in splits else None
 
        hyp['dataset']['num_classes'] = 112

    elif hyp['dataset']['name'] == 'ecoset_square256':

        dataset_path = f"{hyp['dataset']['dataset_path']}ecoset_square{hyp['dataset']['image_size']}_proper_chunks.h5"
        print(f"Loading dataset from: {dataset_path}")

        train_transform = get_transform(hyp['dataset']['augment'], hyp)
        val_test_transform = get_transform(hyp['dataset']['val_test_augment'], hyp)                                    
        if self_supervised:
            train_transform = TwoCropsTransform(train_transform)
            val_test_transform = TwoCropsTransform(val_test_transform)

        train_dataset = Ecoset('train', dataset_path, train_transform, in_memory=in_memory) if 'train' in splits else None
        val_dataset = Ecoset('val', dataset_path, val_test_transform, in_memory=in_memory) if 'val' in splits else None
        test_dataset = Ecoset('test', dataset_path, val_test_transform, in_memory=in_memory) if 'test' in splits else None
        
        hyp['dataset']['num_classes'] = 565

    elif hyp['dataset']['name'] == 'ecoset_square256_patches':
        
        # dataset_path = f"{hyp['dataset']['dataset_path']}optimized_datasets/megacoset.h5"
        dataset_path = f"{hyp['dataset']['dataset_path']}optimized_datasets/coset.h5"
        print(f"Loading dataset from: {dataset_path}")

        train_transform = get_transform(hyp['dataset']['augment'], hyp)
        val_test_transform = get_transform(hyp['dataset']['val_test_augment'], hyp)                                    
        if self_supervised:
            train_transform = TwoCropsTransform(train_transform)
            val_test_transform = TwoCropsTransform(val_test_transform)
            
        train_dataset = Ecoset('train', dataset_path, train_transform, in_memory=in_memory) if 'train' in splits else None
        val_dataset = Ecoset('val', dataset_path, val_test_transform, in_memory=in_memory) if 'val' in splits else None
        test_dataset = Ecoset('test', dataset_path, val_test_transform, in_memory=in_memory) if 'test' in splits else None
        
        hyp['dataset']['num_classes'] = 565
        

    elif hyp['dataset']['name'] == 'imagenet':
        from evd.datasets.imagenet.imagenet import load_imagenet
        imagenet_path= "/share/klab/datasets/imagenet/"

        train_transform = get_transform(hyp['dataset']['augment'], hyp)
        val_test_transform = get_transform(hyp['dataset']['val_test_augment'], hyp)   
        if self_supervised:
            train_transform = TwoCropsTransform(train_transform)
            val_test_transform = TwoCropsTransform(val_test_transform)
            
        train_loader, train_sampler, test_loader = load_imagenet(imagenet_path=imagenet_path,
                                                    batch_size=hyp['optimizer']['batch_size'],
                                                    distributed = False,
                                                    workers = hyp['optimizer']['dataloader']['num_workers_train'],
                                                    train_transforms = train_transform,
                                                    test_transforms= val_test_transform,
                                                    # normalization = False, # Not setting norm here
        )
        
        hyp['dataset']['num_classes'] = 1000
        with open("imagenet/imagenet_classnames.json", "r") as f:
            hyp['dataset']['class_names'] = json.load(f) # Load imagenet classnames
        
        return train_loader, test_loader, test_loader, hyp
    

    elif hyp['dataset']['name'] == 'facescrub':
        dataset_file = '/share/klab/datasets/texture2shape_projects/generate_facescrub_dataset/facescrub_256px.h5'
        print(f"Loading dataset from: {dataset_file}")
        with h5py.File(dataset_file, "r") as f:
            hyp['dataset']['num_classes'] = np.array(f['categories']).shape[0]
        
        train_transform = get_transform(hyp['dataset']['augment'], hyp)
        val_test_transform = get_transform(hyp['dataset']['val_test_augment'], hyp)  
        if self_supervised:
            train_transform = TwoCropsTransform(train_transform)
            val_test_transform = TwoCropsTransform(val_test_transform)
            
        train_dataset = MiniEcoset('train', dataset_file, train_transform)
        val_dataset = MiniEcoset('val', dataset_file, val_test_transform)
        test_dataset = MiniEcoset('test', dataset_file, val_test_transform)

    else:
        raise ValueError(f"Unknown dataset: {hyp['dataset']['name']}")

    # Create Dataloaders for the splits
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hyp['optimizer']['batch_size'], shuffle=True,
                                                num_workers=hyp['optimizer']['dataloader']['num_workers_train'],
                                                prefetch_factor=hyp['optimizer']['dataloader']['prefetch_factor_train'],
                                                pin_memory=True,                # <--- recommended True for CUDA
                                                persistent_workers=True         # <--- recommended
                                                ) if 'train' in splits else None
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=hyp['misc']['batch_size_val_test'],
                                                num_workers=hyp['optimizer']['dataloader']['num_workers_val_test'],
                                                prefetch_factor=hyp['optimizer']['dataloader']['prefetch_factor_val_test'],
                                                pin_memory=True,                # <--- recommended True for CUDA
                                                persistent_workers=True         # <--- recommended
                                                ) if 'val' in splits else None
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hyp['misc']['batch_size_val_test'],
                                                num_workers=hyp['optimizer']['dataloader']['num_workers_val_test'],
                                                prefetch_factor=hyp['optimizer']['dataloader']['prefetch_factor_val_test'],
                                                pin_memory=True,                # <--- recommended True for CUDA
                                                persistent_workers=True         # <--- recommended
                                                ) if 'test' in splits else None


    
    if compute_stats:
        mean, std = compute_mean_std(train_loader)
        print(f"train loader mean & std: {mean}, {std}")
        mean, std = compute_mean_std(val_loader)
        print(f"val loader mean & std: {mean}, {std}")
        mean, std = compute_mean_std(test_loader)
        print(f"test loader mean & std: {mean}, {std}")  
        
    return train_loader, val_loader, test_loader, hyp


def get_transform(aug_str, hyp=None, normalize_type='0-1'):

    """
    Build a Kornia augmentation pipeline as an nn.Module (on GPU).
    """
    aug_list = []
    aug_list.append(torchvision.transforms.ConvertImageDtype(torch.float))

    if hyp['dataset']['resize']:
        print('resizing to 224x224')
        aug_list.append(K.Resize((224, 224), align_corners=True))

    if 'grayscale' in aug_str:
        aug_list.append(K.RandomGrayscale(p=0.5))
    if 'randomflip' in aug_str:
        aug_list.append(K.RandomHorizontalFlip(p=0.5))
    if 'randomrotation' in aug_str:
        aug_list.append(K.RandomRotation(degrees=45.0, p=0.5))
    if 'globalbrightness' in aug_str:
        aug_list.append(K.RandomBrightness(brightness=(0.8, 1.2), p=0.5))
    if 'equalize' in aug_str:
        aug_list.append(K.RandomEqualize(p=0.5))
    if 'perspective' in aug_str:
        aug_list.append(K.RandomPerspective(distortion_scale=0.5, p=0.5))

    if 'sharpness' in hyp['dataset']['control_aug_str']:
        if 'sharpness' in aug_str:
            print('using sharpness')
            aug_list.append(K.RandomSharpness(p=0.5))
    if 'blur' in hyp['dataset']['control_aug_str']:
        if 'blur' in aug_str:
            print('using blur')
            aug_list.append(K.RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0), p=0.5))

    augmentations = nn.Sequential(*aug_list)

    class KorniaTransform(nn.Module):
        def __init__(self, aug_pipe, norm_type, parent_hyp):
            super().__init__()
            self.aug_pipe = aug_pipe
            self.normalize_type = norm_type
            self.parent_hyp = parent_hyp

        def forward(self, x):
            if not torch.is_tensor(x):
                x = torchvision.transforms.functional.to_tensor(x)

            x = self.aug_pipe(x)

            if 'normalize' in aug_str:
                if self.normalize_type == '0-1':
                    pass
                else:
                    raise ValueError(f"Unknown normalize type: {self.normalize_type}")
            return x

    return KorniaTransform(augmentations, normalize_type, hyp)


class Ecoset(torch.utils.data.Dataset):
    #Import Ecoset as a Dataset splitwise
    def __init__(self, split, dataset_path, transform=None, in_memory=True):
        """
        Args:
            dataset_path (string): Path to the .h5 file
            transform (callable, optional): Optional transforms to be applied
                on a sample.
            in_memory: Should we pre-load the dataset?
        """
        self.root_dir = dataset_path
        self.transform = transform
        self.split = split
        self.in_memory = in_memory

        if self.in_memory:
            with h5py.File(dataset_path, "r") as f:
                self.images = torch.from_numpy(f[split]['data'][()]).permute((0, 3, 1, 2)) # to match the CHW expectation of pytorch
                self.labels = torch.from_numpy(f[split]['labels'][()].astype(np.int64)).long()
        else:
            self.split_data = h5py.File(dataset_path, "r")[split]
            self.images = self.split_data['data']
            self.labels = self.split_data['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx): # accepts ids and returns the images and labels transformed to the Dataloader
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.in_memory:
            imgs = self.images[idx]
            labels = self.labels[idx]
        else:
            with h5py.File(self.root_dir, "r") as f:
                imgs = torch.from_numpy(np.asarray(self.images[idx])).permute((2,0,1))    
                labels = torch.from_numpy(np.asarray(self.labels[idx].astype(np.int64))).long()

        if self.transform:
            imgs = self.transform(imgs)

        return imgs, labels


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]

def compute_mean_std(loader):
    # Variables for the sum and square sum of all pixels and the number of batches
    mean = 0.0
    mean_square = 0.0
    samples = 0

    for data in loader:
        # Assuming the data loader returns a tuple of (images, labels)
        images, _ = data
        # If images is a list (self-supervised setting), combine the two views
        if isinstance(images, list):
            images = torch.cat(images, dim=0)  # Concatenate along the batch dimension
            
        # Flatten the channels
        images = images.view(images.size(0), images.size(1), -1)
        mean += images.mean(2).sum(0)
        mean_square += (images ** 2).mean(2).sum(0)
        samples += images.size(0)
    
    # Final calculation of mean and std
    mean /= samples
    mean_square /= samples
    std = (mean_square - mean ** 2) ** 0.5

    return mean, std