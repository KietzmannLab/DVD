import os
from PIL import Image
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
from torchvision import transforms, datasets
from dvd.datasets.view_generator import ContrastiveLearningViewGenerator
from torch.utils.data.distributed import DistributedSampler

from typing import Optional, Dict

import kornia
import kornia.augmentation as K
import kornia.filters as KF
import dvd.utils

##############################
## Testing datasets loader
##############################

def get_test_loaders(args):
    """
    Retrieves the requested datasets and wraps them into DataLoaders with distributed sampling.
    Only returns (loader, sampler) pairs for splits specified in the 'splits' list.
    """
    dataset_name = args.dataset_name
    dataset = SupervisedLearningDataset(args.data, args)  # Load dataset from path (args.data) & also parse all
    dataset = dataset.get_dataset(dataset_name)

    test_sampler = None
    test_loader = torch.utils.data.DataLoader(
        dataset["test"],
        batch_size=args.batch_size_per_gpu,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        # sampler=test_sampler,
        drop_last=True,
    )
    return test_loader

def get_blur_loader(sigma: int, args) -> DataLoader:
    """
    DataLoader for the sigma‐blurred test set.
    Expects args['dataset']['name'] in {'texture2shape_miniecoset','facescrub'}
    and HDF5 files named accordingly under your robustness_datasets folder.
    """
    image_size = 256
    ds_name = args.dataset_name
    if ds_name == 'texture2shape_miniecoset':
        base = "/home/student/l/lzejin/codebase/P001_dvd_gpus/data/robustness_datasets/degradation_images_generator/data/miniecoset/blurred/"
        split = 'test'
        fname = f"texture2shape_miniecoset_{image_size}px_sigma_{sigma}_blurred.h5"
    elif ds_name == 'ecoset_square256':
        base = "/home/student/l/lzejin/codebase/P001_dvd_gpus/data/robustness_datasets/degradation_images_generator/data/objects_blurred_datasets/"
        split = 'test'
        fname = f"ecoset_{image_size}px_sigma_{sigma}_blurred.h5"
    elif ds_name == 'facescrub':
        base = "/home/student/l/lzejin/codebase/P001_dvd_gpus/data/robustness_datasets/degradation_images_generator/data/faces_blurred_datasets/"
        split = 'test'
        fname = f"facescrub_{image_size}px_sigma_{sigma}_blurred.h5"
    else:
        raise ValueError(f"Unsupported dataset for blur: {ds_name}")

    path = os.path.join(base, fname)
    print(f"[blur_loader_fn] Loading: {path}")

    # simple torchvision test‐transform
    test_tf = transforms.Compose([
        transforms.ToPILImage(),  # handles Tensor or ndarray → PIL
        transforms.ToTensor(),
        # transforms.Normalize(mean=mean.tolist(), std=std.tolist()),
    ])

    # wrap your Ecoset HDF5 reader
    ds = Ecoset(split=split, dataset_path=path, transform=test_tf, in_memory=False)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.workers,
        pin_memory=True,
    )
    return loader


def get_distortion_loader(method: str, severity: int, args) -> DataLoader:
    """
    DataLoader for an ImageNet-C style corruption:
      `method` (e.g. "Gaussian Noise") at `severity` (1–5).
    HDF5 files live under your distorted_datasets folder.
    """
    image_size = 256
    ds_name = args.dataset_name
    if ds_name == 'ecoset_square256':
        # base = "/share/klab/datasets/robustness_datasets/degradation_images_generator/data/ecoset_old/full_ecoset_distorted_datasets_v1/"
        base = "/share/klab/datasets/robustness_datasets/degradation_images_generator/data/ecoset_degradtions/"
    else:
        # base = "/share/klab/datasets/robustness_datasets/distorted_datasets/"
        raise ValueError(f"Unsupported dataset for distortion: {ds_name}")
    # fname = f"{method.replace(' ', '_')}_severity_{severity}_{image_size}px.h5"
    fname =  f"{method.replace(' ','_')}_sev{severity}_{image_size}px.h5"
    path = os.path.join(base, fname)
    print(f"[distortion_loader_fn] Loading: {path}")

    test_tf = transforms.Compose([
        transforms.ToPILImage(), 
        transforms.ToTensor(),
    ])
        # transforms.Normalize(mean=mean.tolist(), std=std.tolist()),

    ds = Ecoset(split='test', dataset_path=path, transform=test_tf, in_memory=True)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.workers,
        pin_memory=True,
    )
    return loader


class KorniaTransform(nn.Module):
    def __init__(self, aug_pipe, normalize_type, args):
        super().__init__()
        self.aug_pipe = aug_pipe
        self.normalize_type = normalize_type
        # expose a 'transforms' list for compatibility checks
        self.transforms = getattr(aug_pipe, 'transforms', [])

        if args is not None and 'train_img_mean_channels' in args:
            self.mean = torch.tensor(args['train_img_mean_channels']) / 255.0
            self.std = torch.tensor(args['train_img_std_channels']) / 255.0
        else:
            self.mean = torch.tensor([0.485, 0.456, 0.406])
            self.std = torch.tensor([0.229, 0.224, 0.225])

    def forward(self, x):
        # if not a tensor, convert it to tensor
        if not torch.is_tensor(x):
            x = torchvision.transforms.functional.to_tensor(x)
        # x should be a Tensor (C,H,W) or (B,C,H,W) on GPU/CPU
        x = self.aug_pipe(x)
        x = x.squeeze()  # remove extra batch dim if present
        # optional normalization
        # if 'normalize' in aug_str:
        if self.normalize_type == '0-1':
            pass  # data is already [0,1]
        elif self.normalize_type == 'mean-std':
            x = K.Normalize(mean=self.mean, std=self.std)(x)
        elif self.normalize_type == '-1-1':
            half = torch.tensor([0.5, 0.5, 0.5], device=x.device).view(1, -1, 1, 1)
            x = (x - half) / half
        else:
            raise ValueError(f"Unknown normalize type: {self.normalize_type}")
        return x

class KorniaTransform(nn.Module):
    def __init__(self, aug_pipe, normalize_type, args):
        super().__init__()
        self.aug_pipe = aug_pipe
        self.normalize_type = normalize_type
        self.transforms = getattr(aug_pipe, 'transforms', []) # expose a 'transforms' list for compatibility checks


    def forward(self, x):
        if not torch.is_tensor(x):
            x = torchvision.transforms.functional.to_tensor(x) # if not a tensor, convert it to tensor
        x = self.aug_pipe(x) # x should be a Tensor (C,H,W) or (B,C,H,W) on GPU/CPU
        x = x.squeeze()  # remove extra batch dim if present
        return x

class SupervisedLearningDataset:
    """
    A rewrite of the original 'ContrastiveLearningDataset' to handle
    standard supervised train/val/test splits with typical augmentations.
    """

    def __init__(self, 
                 root_folder: str, 
                 args= None):
        """
        :param root_folder: Base path to your datasets
        :param args: A dictionary of argserparameters/configs
        """
        self.root_folder = root_folder
        self.args = args if args is not None else {}

    def get_supervised_pipeline_transform(self, 
                                          train: bool = True, 
                                          normalize_type: str = '0-1') -> KorniaTransform:
        """
        Build a Kornia (or torchvision) augmentation pipeline as an nn.Module
        suitable for supervised training. The pipeline differs for
        train vs. val/test.
        """
        aug_list = []

        # 1) Convert images to float [0,1]
        aug_list.append(torchvision.transforms.ConvertImageDtype(torch.float))
        # 2) Resize to 224x224 or 256x256 if specified in args
        aug_list.append(K.Resize((self.args.image_size, self.args.image_size)))
    
        if train:
            # Typical augmentations for training
            aug_list.append(K.RandomHorizontalFlip(p=0.25))
            aug_list.append(K.RandomRotation(degrees=15.0, p=0.25))
            aug_list.append(K.RandomGrayscale(p=0.5))
            aug_list.append(K.RandomBrightness(brightness=(0.8, 1.2), p=0.5))
            aug_list.append(K.RandomEqualize(p=0.5))
            aug_list.append(K.RandomPerspective(distortion_scale=0.5, p=0.5))
            aug_list.append(K.RandomSharpness(p=0.5))
            if getattr(self.args, "blur_aug", False):
                aug_list.append(K.RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0), p=0.5))
        else:
            # Typical "inference" transforms (e.g. CenterCrop, Resize)
            # Adjust as per your dataset dimension preferences:
            # For example:
            # aug_list.append(K.CenterCrop((224, 224)))
            pass

        # Build the final pipeline
        augmentations = nn.Sequential(*aug_list)
        return KorniaTransform(augmentations, normalize_type, self.args)

    def get_dataset(self, dataset_name: str) -> Dict[str, torch.utils.data.Dataset]:
        """
        Creates and returns a dict of { 'train': ..., 'val': ..., 'test': ... }
        for the requested dataset_name.
        """
        valid_datasets = {
            'texture2shape_miniecoset': self._get_texture2shape_miniecoset,
            'ecoset_square256': self._get_ecoset_square256,
            'ecoset_square256_patches': self._get_ecoset_square256_patches,
            'imagenet': self._get_imagenet,
            'facescrub': self._get_facescrub,
            'stl10': self._get_stl10,
            'coco-split515': self._get_coco_split515,
            'skeleton_ayzenberg': self._get_skeleton_ayzenberg,
        }

        # import pdb;pdb.set_trace()
        
        if dataset_name not in valid_datasets:
            raise InvalidDatasetSelection(f"Dataset {dataset_name} not supported.")

        return valid_datasets[dataset_name]()

    # ------------------------------------------------
    # Below are example helper methods for each dataset
    # Each returns {'train': ..., 'val': ..., 'test': ...}
    # ------------------------------------------------

    def _get_texture2shape_miniecoset(self) -> Dict[str, torch.utils.data.Dataset]:
        """
        Example: texture2shape_miniecoset dataset using an .h5 file
        with train/val/test splits.
        """
        image_size = 256 #self.args.get('dataset', {}).get('image_size', 256)
        dataset_path = f"{self.root_folder}/texture2shape_miniecoset_{image_size}px.h5" 
        print(f"[INFO] Loading texture2shape_miniecoset dataset from {dataset_path}")

        # Get transforms
        train_transform = self.get_supervised_pipeline_transform(train=True)
        val_transform   = self.get_supervised_pipeline_transform(train=False)
        test_transform  = self.get_supervised_pipeline_transform(train=False)

        # Example: Ecoset-like usage with splits 'train', 'val', 'test'
        train_dataset = Ecoset(
            split='train',
            dataset_path=dataset_path,
            transform=train_transform,
            in_memory=True
        )
        val_dataset = Ecoset(
            split='val',
            dataset_path=dataset_path,
            transform=val_transform,
            in_memory=True
        )
        test_dataset = Ecoset(
            split='test',
            dataset_path=dataset_path,
            transform=test_transform,
            in_memory=True
        )

        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }

    def _get_ecoset_square256(self) -> Dict[str, torch.utils.data.Dataset]:
        """
        Example: Ecoset with 256x256 images
        """
        image_size = 256 #self.args.get('dataset', {}).get('image_size', 256)
        dataset_path = f"{self.root_folder}/ecoset_square{image_size}_proper_chunks.h5"
        print(f"[INFO] Loading ecoset_square256 dataset from {dataset_path}")

        train_transform = self.get_supervised_pipeline_transform(train=True)
        val_transform   = self.get_supervised_pipeline_transform(train=False)
        test_transform  = self.get_supervised_pipeline_transform(train=False)

        train_dataset = Ecoset(
            split='train',
            dataset_path=dataset_path,
            transform=train_transform,
            in_memory=False
        )
        val_dataset = Ecoset(
            split='val',
            dataset_path=dataset_path,
            transform=val_transform,
            in_memory=False
        )
        test_dataset = Ecoset(
            split='test',
            dataset_path=dataset_path,
            transform=test_transform,
            in_memory=False
        )

        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }

    def _get_ecoset_square256_patches(self) -> Dict[str, torch.utils.data.Dataset]:
        """
        Example: Ecoset patches
        """
        image_size = 256 #self.args.get('dataset', {}).get('image_size', 256)
        dataset_path = f"{self.root_folder}/optimized_datasets/megacoset.h5"
        print(f"[INFO] Loading ecoset_square256_patches dataset from {dataset_path}")

        train_transform = self.get_supervised_pipeline_transform(train=True)
        val_transform   = self.get_supervised_pipeline_transform(train=False)
        test_transform  = self.get_supervised_pipeline_transform(train=False)

        train_dataset = Ecoset(
            split='train',
            dataset_path=dataset_path,
            transform=train_transform,
            in_memory=False
        )
        val_dataset = Ecoset(
            split='val',
            dataset_path=dataset_path,
            transform=val_transform,
            in_memory=False
        )
        test_dataset = Ecoset(
            split='test',
            dataset_path=dataset_path,
            transform=test_transform,
            in_memory=False
        )

        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }

    def _get_imagenet(self) -> Dict[str, torch.utils.data.Dataset]:
        """
        Example: ImageNet. For demonstration, we use torchvision's
        datasets.FakeData as a placeholder.
        """
        image_size = 256 #self.args.get('dataset', {}).get('image_size', 224)
        imagenet_path= "/share/klab/datasets/imagenet/"
        print(f"[INFO] Loading ImageNet dataset ({imagenet_path}).")

        train_transform = self.get_supervised_pipeline_transform(train=True)
        val_test_transform   = self.get_supervised_pipeline_transform(train=False)

        from dvd.datasets.imagenet.imagenet import load_imagenet
        imagenet_path= "/share/klab/datasets/imagenet/"

        train_dataset, train_sampler, val_dataset = load_imagenet(imagenet_path=imagenet_path,
                                                    batch_size= self.args.batch_size_per_gpu,
                                                    distributed = True,
                                                    workers = self.args.workers,
                                                    train_transforms = train_transform,
                                                    test_transforms= val_test_transform,
                                                    # normalization = False, # Not setting norm here
        )

        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': val_dataset,
        }
    
    def _get_skeleton_ayzenberg(self) -> Dict[str, torch.utils.data.Dataset]:
        image_size = 256
        data_path = "/home/student/l/lzejin/codebase/P001_dvd_gpus/data/skeleton_ayzenberg/original/"  # change as appropriate
        print(f"[INFO] Loading Skeleton Ayzenberg dataset from ({data_path}).")

        transform = self.get_supervised_pipeline_transform(train=False)
        test_dataset = Dataset_from_Dir(root_dir=data_path, transform=transform)

        return {
            'train': None,
            'val': test_dataset,
            'test': test_dataset,
    }

    def _get_coco_split515(self) -> Dict[str, torch.utils.data.Dataset]:
        """
        Example: coco-split515 dataset using an .h5 file
        with train/val/test splits.
        """
        image_size = 256  # fixed for this split
        dataset_path = f"{self.root_folder}/gaze_stitching_dataset/coco-split515.h5"
        print(f"[INFO] Loading coco-split515 dataset from {dataset_path}")

        # Get your usual train/val/test transforms
        train_transform = self.get_supervised_pipeline_transform(train=True)
        val_transform   = self.get_supervised_pipeline_transform(train=False)
        test_transform  = self.get_supervised_pipeline_transform(train=False)

        # Instantiate one HDF5-backed dataset per split.
        train_dataset = COCOSplitH5Dataset(
            split='train',
            dataset_path=dataset_path,
            transform=train_transform,
            in_memory=True
        )
        val_dataset = COCOSplitH5Dataset(
            split='val',
            dataset_path=dataset_path,
            transform=val_transform,
            in_memory=True
        )
        test_dataset = COCOSplitH5Dataset(
            split='test',
            dataset_path=dataset_path,
            transform=test_transform,
            in_memory=True
        )

        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }

    def _get_facescrub(self) -> Dict[str, torch.utils.data.Dataset]:
        """
        Example: FaceScrub dataset usage from an HDF5 file with 
        train/val/test splits.
        """
        image_size = 256 #self.args.get('dataset', {}).get('image_size', 256)
        # dataset_file = f"{self.root_folder}/facescrub_{image_size}px.h5"
        dataset_file =  f'{self.root_folder}/texture2shape_projects/generate_facescrub_dataset/facescrub_256px.h5'
        print(f"[INFO] Loading facescrub dataset from {dataset_file}")

        train_transform = self.get_supervised_pipeline_transform(train=True)
        val_transform   = self.get_supervised_pipeline_transform(train=False)
        test_transform  = self.get_supervised_pipeline_transform(train=False)

        train_dataset = Ecoset(
            split='train',
            dataset_path=dataset_file,
            transform=train_transform,
            in_memory=True
        )
        val_dataset = Ecoset(
            split='val',
            dataset_path=dataset_file,
            transform=val_transform,
            in_memory=True
        )
        test_dataset = Ecoset(
            split='test',
            dataset_path=dataset_file,
            transform=test_transform,
            in_memory=True
        )

        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }

    def _get_stl10(self) -> Dict[str, torch.utils.data.Dataset]:
        """
        Example: STL10. Typically, 'train' is labeled, 'unlabeled' is for SSL, 
        but for supervised we can still define train/val/test. 
        We'll create a simple demonstration here.
        """
        print("[INFO] Loading STL10 dataset (split='train', 'test').")
        train_transform = self.get_supervised_pipeline_transform(train=True)
        val_transform   = self.get_supervised_pipeline_transform(train=False)
        test_transform  = self.get_supervised_pipeline_transform(train=False)

        # STL10 does not provide a separate 'val' by default, you might 
        # do your own split or treat part of 'train' as val. For illustration:
        train_dataset = datasets.STL10(
            root=self.root_folder,
            split='train',
            download=True,
            transform=train_transform
        )
        # There's no official 'val', let's just clone 'train' as a placeholder:
        val_dataset = datasets.STL10(
            root=self.root_folder,
            split='train',
            download=True,
            transform=val_transform
        )
        test_dataset = datasets.STL10(
            root=self.root_folder,
            split='test',
            download=True,
            transform=test_transform
        )

        return {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }

class BaseException(Exception):
    """Base exception"""

class InvalidDatasetSelection(BaseException):
    """Raised when the choice of dataset is invalid."""

class ContrastiveLearningDataset:
    """
    A rewrite of your original dataset selection logic in the style of
    the ContrastiveLearningDataset used in SimCLR-like frameworks.
    """

    def __init__(self, root_folder, n_views, args=None):
        """
        :param root_folder: Base path to your datasets
        :param args: A dictionary of argserparameters/configs
        :param n_views: Number of augmented views to generate (default=2 for standard contrastive learning)
        """
        self.root_folder = root_folder
        self.args = args if args is not None else {}
        self.n_views = n_views
        self.image_size = getattr(self.args, 'image_size', 256)
    
    def get_simclr_pipeline_transform(self, train=True, args=None, normalize_type='0-1'):
        """
        Build a Kornia augmentation pipeline as an nn.Module.
        All transforms will run on GPU if the input tensor is on GPU.
        """
        aug_list = []

        # ADD to float, need to be done before Kornia transforms
        aug_list.append(torchvision.transforms.ConvertImageDtype(torch.float))
        aug_list.append(K.Resize((self.image_size , self.image_size)))

        if train:
            #* Now set the same as supervised learning
            aug_list.append(K.RandomHorizontalFlip(p=0.25))
            aug_list.append(K.RandomRotation(degrees=15.0, p=0.25))
            # if self.args.grayscale_aug:
            aug_list.append(K.RandomGrayscale(p=0.5))
            aug_list.append(K.RandomBrightness(brightness=(0.8, 1.2), p=0.5))
            aug_list.append(K.RandomEqualize(p=0.5))
            aug_list.append(K.RandomPerspective(distortion_scale=0.5, p=0.5))
            aug_list.append(K.RandomSharpness(p=0.5))
            if getattr(self.args, "blur_aug", False):
                aug_list.append(K.RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0), p=0.5))

        # Build the final pipeline
        augmentations = nn.Sequential(*aug_list)

        return KorniaTransform(augmentations, normalize_type, self.args)

    def get_dataset(self, dataset_name):
        """
        Creates and returns the dataset object corresponding to `dataset_name`.
        This uses ContrastiveLearningViewGenerator to replace TwoCropsTransform.
        You can add logic for train/val/test splits, etc., as needed.
        """
        # A dictionary to map dataset names to the corresponding creation logic:
        valid_datasets = {
            'texture2shape_miniecoset': self._get_texture2shape_miniecoset,
            'ecoset_square256': self._get_ecoset_square256,
            'ecoset_square256_patches': self._get_ecoset_square256_patches,
            'imagenet': self._get_imagenet,
            'facescrub': self._get_facescrub,
            'stl10': self._get_stl10
        }

        if dataset_name not in valid_datasets:
            raise InvalidDatasetSelection(f"Dataset {dataset_name} not supported.")

        return valid_datasets[dataset_name]()

    # ------------------------------------------------
    # Below are example helper methods for each dataset
    # ------------------------------------------------

    def _get_texture2shape_miniecoset(self):
        """
        Example: texture2shape_miniecoset dataset, referencing the .h5 path as in your original code.
        """
        image_size =  256 #self.args.get('dataset', {}).get('image_size', 256)
        dataset_path = f"{self.root_folder}/texture2shape_miniecoset_{image_size}px.h5" 
        print(f"[INFO] Loading texture2shape_miniecoset dataset from {dataset_path}")

        # If doing self-supervised, we replace TwoCropsTransform with ContrastiveLearningViewGenerator:
        simclr_transform = self.get_simclr_pipeline_transform()
        contrastive_transform = ContrastiveLearningViewGenerator(simclr_transform, self.n_views)

        # For simplicity, just return the 'train' portion. 
        # Extend this logic for 'val'/'test' as needed.
        dataset = Ecoset(
            split='train',
            dataset_path=dataset_path,
            transform=contrastive_transform,
            in_memory=True  # or False, as you prefer
        )
        return dataset

    def _get_ecoset_square256(self):
        """
        Example: Ecoset with 256x256 images
        """
        image_size = 256 # self.args.get('dataset', {}).get('image_size', 256)
        dataset_path = f"{self.root_folder}/ecoset_square{image_size}_proper_chunks.h5"
        print(f"[INFO] Loading ecoset_square256 dataset from {dataset_path}")

        simclr_transform = self.get_simclr_pipeline_transform()
        contrastive_transform = ContrastiveLearningViewGenerator(simclr_transform, self.n_views)

        dataset = Ecoset(
            split='train',
            dataset_path=dataset_path,
            transform=contrastive_transform,
            in_memory=True
        )
        return dataset

    def _get_ecoset_square256_patches(self):
        """
        Example: Ecoset patches
        """
        image_size = 256 #self.args.get('dataset', {}).get('image_size', 256)
        dataset_path = f"{self.root_folder}/optimized_datasets/megacoset.h5"
        print(f"[INFO] Loading ecoset_square256_patches dataset from {dataset_path}")

        simclr_transform = self.get_simclr_pipeline_transform()
        contrastive_transform = ContrastiveLearningViewGenerator(simclr_transform, self.n_views)

        dataset = Ecoset(
            split='train',
            dataset_path=dataset_path,
            transform=contrastive_transform,
            in_memory=True
        )
        return dataset

    def _get_imagenet(self):
        """
        Load ImageNet-1k for self-supervised contrastive learning.

        It expects the canonical folder layout::
            <imagenet_root>/
                train/
                    n01440764/xxx.JPEG
                    ...
                val/
                    n01440764/xxx.JPEG
                    ...

        Returns
        -------
        dict
            {
            'train'        : torchvision.datasets.ImageFolder,
            'val'          : torchvision.datasets.ImageFolder,
            'test'         : torchvision.datasets.ImageFolder,   # alias to val
            'train_sampler': torch.utils.data.DistributedSampler | None,
            'val_sampler'  : torch.utils.data.DistributedSampler | None
            }
        """

        # ---------- paths ----------
        imagenet_root = getattr(self.args, "imagenet_path",
                                "/share/klab/datasets/imagenet")
        train_dir = os.path.join(imagenet_root, "train")
        val_dir   = os.path.join(imagenet_root, "val")
        if not (os.path.isdir(train_dir) and os.path.isdir(val_dir)):
            raise FileNotFoundError(
                f"Expected ImageNet folders at {train_dir} and {val_dir}"
            )

        # ---------- transforms ----------
        simclr_train = self.get_simclr_pipeline_transform(train=True)
        train_transform = ContrastiveLearningViewGenerator(simclr_train,
                                                        self.n_views)

        simclr_eval = self.get_simclr_pipeline_transform(train=False)
        eval_transform = ContrastiveLearningViewGenerator(simclr_eval,
                                                        self.n_views)

        # ---------- datasets ----------
        train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
        val_dataset   = datasets.ImageFolder(val_dir,   transform=eval_transform)

        # ---------- distributed samplers (optional) ----------
        train_sampler = val_sampler = None
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            val_sampler   = DistributedSampler(val_dataset,   shuffle=False)

        return {
            'train'        : train_dataset,
            'val'          : val_dataset,
            'test'         : val_dataset,   # reuse val split
            'train_sampler': train_sampler,
            'val_sampler'  : val_sampler,
        }
    def _get_facescrub(self):
        """
        Example: FaceScrub dataset usage.
        In your original code, you have an HDF5 with 'train', 'val', 'test' splits.
        We show a simplified single-split usage here.
        """
        image_size = 256 #self.args.get('dataset', {}).get('image_size', 256)
        dataset_file = f"{self.root_folder}/facescrub_256px.h5"
        print(f"[INFO] Loading facescrub dataset from {dataset_file}")

        simclr_transform = self.get_simclr_pipeline_transform()
        contrastive_transform = ContrastiveLearningViewGenerator(simclr_transform, self.n_views)

        # If you created a custom FaceScrub class similar to Ecoset:
        return Ecoset(
            split='train',
            dataset_path=dataset_file,
            transform=contrastive_transform,
            in_memory=True
        )

    def _get_stl10(self):
        """
        Example: STL10 in an unlabeled split, as is common in self-supervised learning.
        """
        print("[INFO] Loading STL10 dataset (unlabeled split).")
        simclr_transform = self.get_simclr_pipeline_transform()
        contrastive_transform = ContrastiveLearningViewGenerator(simclr_transform, self.n_views)

        dataset = datasets.STL10(
            root=self.root_folder,
            split='train',  # often used for SSL
            download=True,
            transform=contrastive_transform
        )
        return dataset


class Ecoset(Dataset):
    """
    Example dataset class for Ecoset, similar to your original code.
    Note: This class is simplified and only demonstrates reading data.
    """

    def __init__(self, split, dataset_path, transform=None, in_memory=False):
        """
        Args:
            dataset_path (string): Path to the .h5 file
            transform (callable, optional): Optional transforms to be applied on a sample.
            in_memory (bool): Whether to preload the entire dataset into memory.
        """
        self.dataset_path = dataset_path
        self.split = split
        self.transform = transform
        self.in_memory = in_memory

        if self.in_memory:
            # Load entire dataset into memory
            with h5py.File(self.dataset_path, "r") as f:
                self.images = torch.from_numpy(f[split]['data'][()]).permute(0, 3, 1, 2)
                self.labels = torch.from_numpy(f[split]['labels'][()].astype(np.int64))
        else:
            # Lazy load
            self.h5_ref = h5py.File(self.dataset_path, "r")[split]
            self.images = self.h5_ref['data']
            self.labels = self.h5_ref['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if self.in_memory:
            img = self.images[idx]
            label = self.labels[idx]
        else:
            # On-the-fly access
            img = torch.from_numpy(np.asarray(self.images[idx])).permute((2, 0, 1))
            # label = torch.from_numpy(np.asarray(self.labels[idx])).long()
            label = torch.from_numpy(np.asarray(self.labels[idx]).astype(np.int64))

        if self.transform:
            img = self.transform(img)

        return img, label

class COCOSplitH5Dataset(Dataset):
    """
    Dataset class for coco-split515.h5, with train/val/test splits.
    Each split group contains:
      - images           (N, 256, 256, 3)   uint8
      - dg3_cover_mask   (N, 256, 256)      bool
      - dg3_cover_probs  (N, 256, 256)      float16 or float32
      - embeddings       (N, 768)           float32
    """
    def __init__(self, split, dataset_path, transform=None, in_memory=False):
        """
        Args:
            split (str):      'train', 'val' or 'test'
            dataset_path (str): Path to coco-split515.h5
            transform (callable, optional): applied to images only
            in_memory (bool): preload everything into RAM
        """
        self.split        = split
        self.dataset_path = dataset_path
        self.transform    = transform
        self.in_memory    = in_memory

        if self.in_memory:
            # load entire split into RAM
            with h5py.File(self.dataset_path, 'r') as f:
                grp = f[self.split]
                # (N,256,256,3) → (N,3,256,256)
                self.images     = torch.from_numpy(grp['images'][()]) \
                                          .permute(0, 3, 1, 2)
                self.masks      = torch.from_numpy(grp['dg3_cover_mask'][()])
                self.probs      = torch.from_numpy(grp['dg3_cover_probs'][()])
                self.embeddings = torch.from_numpy(grp['embeddings'][()])
        else:
            # keep file open and index on-the-fly
            self.h5_file      = h5py.File(self.dataset_path, 'r')
            grp                = self.h5_file[self.split]
            self.images_ref   = grp['images']
            self.masks_ref    = grp['dg3_cover_mask']
            self.probs_ref    = grp['dg3_cover_probs']
            self.embeddings_ref = grp['embeddings']

    def __len__(self):
        return (
            self.images.shape[0]
            if self.in_memory
            else self.images_ref.shape[0]
        )

    def __getitem__(self, idx):
        if self.in_memory:
            img       = self.images[idx]
            mask      = self.masks[idx]
            probs     = self.probs[idx]
            embedding = self.embeddings[idx]
        else:
            # load single sample
            img_np       = np.asarray(self.images_ref[idx])
            mask_np      = np.asarray(self.masks_ref[idx])
            probs_np     = np.asarray(self.probs_ref[idx])
            embed_np     = np.asarray(self.embeddings_ref[idx])

            img       = torch.from_numpy(img_np).permute(2, 0, 1)
            mask      = torch.from_numpy(mask_np)
            probs     = torch.from_numpy(probs_np)
            embedding = torch.from_numpy(embed_np)

        if self.transform:
            img = self.transform(img)

        # returns (image, mask, cover_probs, embedding)
        return img, mask, probs, embedding

    def __del__(self):
        # clean up file handle if used
        if not self.in_memory and hasattr(self, 'h5_file'):
            self.h5_file.close()

class Dataset_from_Dir(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        for fname in sorted(os.listdir(root_dir)):
            if fname.endswith('.jpg'):
                label = fname.split('_')[0]  # e.g., 'boat' from 'boat_1_30.jpg'
                self.samples.append((os.path.join(root_dir, fname), label))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(sorted(set(lbl for _, lbl in self.samples)))}
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_str = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.class_to_idx[label_str]
        if self.transform:
            image = self.transform(image)
        return image, label

# ----------------------------------------------------------------
# Example usage:
# ----------------------------------------------------------------
if __name__ == "__main__":
    # Suppose we want the 'ecoset_square256' dataset
    args = {
        'dataset': {
            'name': 'stl10',
            'image_size': 96
        }
    }
    data_obj = ContrastiveLearningDataset(root_folder="/share/klab/datasets/", args=args, n_views=2)
    dataset = data_obj.get_dataset(args['dataset']['name'])  # This returns the train split, for example

    # Next steps: iterate, train, etc.
    for images, labels in dataset:
        # images is a list of two augmented views [B, C, H, W]
        # labels is the ground truth label (or dummy labels if unlabeled)
        view1, view2 = images
        print(view1.shape, view2.shape, labels)
        