import os
from typing import Tuple, Any

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import os
import torch
from torchvision import datasets, transforms
from typing import Any, Tuple

import torch.distributed as dist


def load_imagenet(
    imagenet_path: str = "/share/klab/datasets/imagenet/",
    batch_size: int = 1024,
    distributed: bool = False,
    workers: int = 4,
    train_transforms: Any = None,
    test_transforms: Any = None,
    image_size =  256,
) -> Tuple[datasets.ImageFolder, Any, datasets.ImageFolder]:
    """
    Return raw ImageFolder datasets + optional sampler (train_dataset, train_sampler, val_dataset).
    Dataset __getitem__ returns (img, label) like Ecoset.
    """
    traindir = os.path.join(imagenet_path, "train")
    valdir   = os.path.join(imagenet_path, "val")

    # if transforms are Compose, insert Resize at index 1
    def insert_resize(ts):
        if isinstance(ts, transforms.Compose):
            ops = list(ts.transforms)
            ops.insert(1, transforms.Resize(image_size))
            return transforms.Compose(ops)
        else:
            # if you passed in something else, just wrap it
            return transforms.Compose([transforms.Resize(image_size), ts])

    if train_transforms is not None:
        print(f'resizing to {image_size} in train...')
        train_transforms = insert_resize(train_transforms)
    if test_transforms is not None:
        print(f'resizing to {image_size} in val/test...')
        test_transforms = insert_resize(test_transforms)
    # import pdb; pdb.set_trace()

    train_dataset = datasets.ImageFolder(traindir, transform=train_transforms)
    val_dataset   = datasets.ImageFolder(valdir,   transform=test_transforms)

    if dist.is_available() and dist.is_initialized():
        train_sampler =  torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = None

    return train_dataset, train_sampler, val_dataset

