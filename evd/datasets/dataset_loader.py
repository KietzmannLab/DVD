import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

import torch.nn as nn
import torchvision
from torchvision.transforms import transforms
from torchvision import transforms, datasets
from evd.datasets.view_generator import ContrastiveLearningViewGenerator

from typing import Optional, Dict

import kornia
import kornia.augmentation as K
import kornia.filters as KF
import evd.utils

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
            label = torch.from_numpy(np.asarray(self.labels[idx])).long()

        if self.transform:
            img = self.transform(img)

        return img, label


class KorniaTransform(nn.Module):
    def __init__(self, aug_pipe, normalize_type, hyp):
        super().__init__()
        self.aug_pipe = aug_pipe
        self.normalize_type = normalize_type

        if hyp is not None and 'train_img_mean_channels' in hyp:
            self.mean = torch.tensor(hyp['train_img_mean_channels']) / 255.0
            self.std = torch.tensor(hyp['train_img_std_channels']) / 255.0
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


class SupervisedLearningDataset:
    """
    A rewrite of the original 'ContrastiveLearningDataset' to handle
    standard supervised train/val/test splits with typical augmentations.
    """

    def __init__(self, 
                 root_folder: str, 
                 hyp= None):
        """
        :param root_folder: Base path to your datasets
        :param hyp: A dictionary of hyperparameters/configs
        """
        self.root_folder = root_folder
        self.hyp = hyp if hyp is not None else {}

    def get_supervised_pipeline_transform(self, 
                                          train: bool = True, 
                                          hyp = None,
                                          normalize_type: str = '0-1') -> KorniaTransform:
        """
        Build a Kornia (or torchvision) augmentation pipeline as an nn.Module
        suitable for supervised training. The pipeline differs for
        train vs. val/test.
        """
        aug_list = []

        # 1) Convert images to float [0,1]
        aug_list.append(torchvision.transforms.ConvertImageDtype(torch.float))

        try:
            if self.hyp.resize_to_224:
                print("Resize to 224x224")
                aug_list.append(K.Resize((224, 224)))
        except:
            print("Fail to resize to 224x224")
            pass

        if train:
            # Typical augmentations for training
            aug_list.append(K.RandomHorizontalFlip(p=0.25))
            aug_list.append(K.RandomRotation(degrees=15.0, p=0.25))
            if self.hyp.grayscale_aug:
                aug_list.append(K.RandomGrayscale(p=0.5))
            aug_list.append(K.RandomBrightness(brightness=(0.8, 1.2), p=0.5))
            aug_list.append(K.RandomEqualize(p=0.5))
            aug_list.append(K.RandomPerspective(distortion_scale=0.5, p=0.5))
            aug_list.append(K.RandomSharpness(p=0.5))
            aug_list.append(K.RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0), p=0.5))
        else:
            # Typical "inference" transforms (e.g. CenterCrop, Resize)
            # Adjust as per your dataset dimension preferences:
            # For example:
            # aug_list.append(K.CenterCrop((224, 224)))
            pass

        # Build the final pipeline
        augmentations = nn.Sequential(*aug_list)
        return KorniaTransform(augmentations, normalize_type, hyp)

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
            'stl10': self._get_stl10
        }

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
        image_size = 256 #self.hyp.get('dataset', {}).get('image_size', 256)
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
        image_size = 256 #self.hyp.get('dataset', {}).get('image_size', 256)
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
        image_size = 256 #self.hyp.get('dataset', {}).get('image_size', 256)
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
        image_size = 256 #self.hyp.get('dataset', {}).get('image_size', 224)
        print("[INFO] Loading ImageNet dataset (placeholder).")

        train_transform = self.get_supervised_pipeline_transform(train=True)
        val_transform   = self.get_supervised_pipeline_transform(train=False)
        test_transform  = self.get_supervised_pipeline_transform(train=False)

        # In real usage, do:
        # train_dataset = datasets.ImageNet(root=self.root_folder, split='train', transform=train_transform)
        # val_dataset   = datasets.ImageNet(root=self.root_folder, split='val', transform=val_transform)
        # test_dataset  = ... # Possibly the same as val or a separate test set

        # For demonstration with FakeData:
        train_dataset = datasets.FakeData(
            size=5000,
            image_size=(3, image_size, image_size),
            num_classes=1000,
            transform=train_transform
        )
        val_dataset = datasets.FakeData(
            size=1000,
            image_size=(3, image_size, image_size),
            num_classes=1000,
            transform=val_transform
        )
        test_dataset = datasets.FakeData(
            size=1000,
            image_size=(3, image_size, image_size),
            num_classes=1000,
            transform=test_transform
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
        image_size = 256 #self.hyp.get('dataset', {}).get('image_size', 256)
        dataset_file = f"{self.root_folder}/facescrub_{image_size}px.h5"
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

    def __init__(self, root_folder, n_views, hyp=None):
        """
        :param root_folder: Base path to your datasets
        :param hyp: A dictionary of hyperparameters/configs
        :param n_views: Number of augmented views to generate (default=2 for standard contrastive learning)
        """
        self.root_folder = root_folder
        self.hyp = hyp if hyp is not None else {}
        self.n_views = n_views
    
    def get_simclr_pipeline_transform(train=True, hyp=None, normalize_type='0-1'):
        """
        Build a Kornia augmentation pipeline as an nn.Module.
        All transforms will run on GPU if the input tensor is on GPU.
        """
        aug_list = []

        # ADD to float, need to be done before Kornia transforms
        aug_list.append(torchvision.transforms.ConvertImageDtype(torch.float))

        if train:
            #* Now set the same as supervised learning
            aug_list.append(K.RandomHorizontalFlip(p=0.25))
            aug_list.append(K.RandomRotation(degrees=15.0, p=0.25))
            # if args.grayscale_aug:
            aug_list.append(K.RandomGrayscale(p=0.5))
            aug_list.append(K.RandomBrightness(brightness=(0.8, 1.2), p=0.5))
            aug_list.append(K.RandomEqualize(p=0.5))
            aug_list.append(K.RandomPerspective(distortion_scale=0.5, p=0.5))
            aug_list.append(K.RandomSharpness(p=0.5))
            aug_list.append(K.RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0), p=0.5))

        # Build the final pipeline
        augmentations = nn.Sequential(*aug_list)

        return KorniaTransform(augmentations, normalize_type, hyp)

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
        image_size =  256 #self.hyp.get('dataset', {}).get('image_size', 256)
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
        image_size = 256 # self.hyp.get('dataset', {}).get('image_size', 256)
        dataset_path = f"{self.root_folder}/ecoset_square{image_size}_proper_chunks.h5"
        print(f"[INFO] Loading ecoset_square256 dataset from {dataset_path}")

        simclr_transform = self.get_simclr_pipeline_transform()
        contrastive_transform = ContrastiveLearningViewGenerator(simclr_transform, self.n_views)

        dataset = Ecoset(
            split='train',
            dataset_path=dataset_path,
            transform=contrastive_transform,
            in_memory=False
        )
        return dataset

    def _get_ecoset_square256_patches(self):
        """
        Example: Ecoset patches
        """
        image_size = 256 #self.hyp.get('dataset', {}).get('image_size', 256)
        dataset_path = f"{self.root_folder}/optimized_datasets/megacoset.h5"
        print(f"[INFO] Loading ecoset_square256_patches dataset from {dataset_path}")

        simclr_transform = self.get_simclr_pipeline_transform()
        contrastive_transform = ContrastiveLearningViewGenerator(simclr_transform, self.n_views)

        dataset = Ecoset(
            split='train',
            dataset_path=dataset_path,
            transform=contrastive_transform,
            in_memory=False
        )
        return dataset

    def _get_imagenet(self):
        """
        Example: ImageNet. 
        Here, we demonstrate a simple approach: rely on torchvision's ImageNet if available,
        or a custom approach. For demonstration, we use a placeholder.
        """
        image_size = 256 #self.hyp.get('dataset', {}).get('image_size', 224)
        print("[INFO] Loading ImageNet dataset (placeholder).")

        simclr_transform = self.get_simclr_pipeline_transform()
        contrastive_transform = ContrastiveLearningViewGenerator(simclr_transform, self.n_views)

        # For real usage with ImageNet, you might do:
        # return datasets.ImageNet(self.root_folder, split='train', transform=contrastive_transform)
        # or use a custom data loader.

        return datasets.FakeData(
            size=1000,  # Example
            image_size=(3, image_size, image_size),
            num_classes=1000,
            transform=contrastive_transform
        )

    def _get_facescrub(self):
        """
        Example: FaceScrub dataset usage.
        In your original code, you have an HDF5 with 'train', 'val', 'test' splits.
        We show a simplified single-split usage here.
        """
        image_size = 256 #self.hyp.get('dataset', {}).get('image_size', 256)
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

# ----------------------------------------------------------------
# Example usage:
# ----------------------------------------------------------------
if __name__ == "__main__":
    # Suppose we want the 'ecoset_square256' dataset
    hyp = {
        'dataset': {
            'name': 'stl10',
            'image_size': 96
        }
    }
    data_obj = ContrastiveLearningDataset(root_folder="/share/klab/datasets/", hyp=hyp, n_views=2)
    dataset = data_obj.get_dataset(hyp['dataset']['name'])  # This returns the train split, for example

    # Next steps: iterate, train, etc.
    for images, labels in dataset:
        # images is a list of two augmented views [B, C, H, W]
        # labels is the ground truth label (or dummy labels if unlabeled)
        view1, view2 = images
        print(view1.shape, view2.shape, labels)
        