import os
from typing import Tuple, Any

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import os
import torch
from torchvision import datasets, transforms
from typing import Any, Tuple

def load_imagenet(
    imagenet_path: str = "/share/klab/datasets/imagenet/",
    batch_size: int = 1024,
    distributed: bool = False,
    workers: int = 4,
    train_transforms: Any = None,
    test_transforms: Any = None
) -> Tuple[Any, Any, Any]:
    traindir = os.path.join(imagenet_path, "train")
    valdir = os.path.join(imagenet_path, "val")

    # Define custom dataset
    custom_dataset = datasets.ImageFolder(traindir)

    # Initialize transform_train with default resizing to (224, 224)
    custom_dataset.transform_train = transforms.Compose([
        transforms.Resize((224, 224)),  # Default resizing
    ])

    # Convert images to half-precision (float16)
    custom_dataset.transform_train.transforms.append(transforms.ConvertImageDtype(torch.float16))

    # Add user-defined train transforms, if available
    if train_transforms and len(train_transforms.transforms) >= 1:
        custom_dataset.transform_train.transforms = [transforms.Resize((224, 224))]  # Replace with user transforms
        for train_transform in train_transforms.transforms:
            custom_dataset.transform_train.transforms.append(train_transform)

    # Test dataset custom transformations
    custom_dataset.transform_test = transforms.Compose([
        transforms.Resize((224, 224)),  # Default resizing for test
    ])

    # Add user-defined test transforms, if available
    if test_transforms and len(test_transforms.transforms) >= 1:
        custom_dataset.transform_test.transforms = [transforms.Resize((224, 224))]
        for test_transform in test_transforms.transforms:
            custom_dataset.transform_test.transforms.append(test_transform)

    # Create train dataset using the custom train transforms
    train_dataset = datasets.ImageFolder(
        traindir,
        custom_dataset.transform_train  # Use custom train transform
    )

    # Handle distributed training
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    # Train DataLoader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    # Validation DataLoader with default transforms for validation
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    # Add normalization if needed, e.g., uncomment the below line:
                    # normalize,
                ]
            ),
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )

    # Set number of classes
    train_loader.num_classes = 1000
    val_loader.num_classes = 1000

    return train_loader, train_sampler, val_loader
    
# def load_imagenet(
#     imagenet_path: str = "/share/klab/datasets/imagenet/",
#     batch_size: int = 1024,
#     distributed: bool = False,
#     workers: int = 4,
# ) -> Tuple[Any, Any, Any]:
#     traindir = os.path.join(imagenet_path, "train")
#     valdir = os.path.join(imagenet_path, "val")
#     # normalize = transforms.Normalize(
#     #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#     # )

#     train_dataset = datasets.ImageFolder(
#         traindir,
#         transforms.Compose(
#             [
#                 transforms.RandomResizedCrop(224),
#                 transforms.RandomHorizontalFlip(),
#                 transforms.ToTensor(),
#                 normalize,
#             ]
#         ),
#     )
    
#     # convert half percision
#     custom_dataset.transform_train.transforms.append(transforms.ConvertImageDtype(torch.float16))

#     if train_transforms.transforms and len(train_transforms.transforms) >= 1:
#     #* Clean the previous default image transforms, using our own transforms    
#         custom_dataset.transform_train.transforms = [transforms.Resize((224, 224))] #! ImageSize might be differernt, e.g. got [3, 333, 500] at entry 0 and [3, 450, 600] at entry 2
#         for train_transfrom in train_transforms.transforms:
#             custom_dataset.transform_train.transforms.append(train_transfrom)
#     if test_transforms and len(test_transforms.transforms) >= 1:
#         custom_dataset.transform_test.transforms = [transforms.Resize((224, 224))]
#         for test_transfrom in test_transforms.transforms:
#             custom_dataset.transform_test.transforms.append(test_transfrom)


#     if distributed:
#         train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
#     else:
#         train_sampler = None

#     train_loader = torch.utils.data.DataLoader(
#         train_dataset,
#         batch_size=batch_size,
#         shuffle=(train_sampler is None),
#         num_workers=workers,
#         pin_memory=True,
#         sampler=train_sampler,
#     )

#     val_loader = torch.utils.data.DataLoader(
#         datasets.ImageFolder(
#             valdir,
#             transforms.Compose(
#                 [
#                     transforms.Resize(256),
#                     transforms.CenterCrop(224),
#                     transforms.ToTensor(),
#                     normalize,
#                 ]
#             ),
#         ),
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=workers,
#         pin_memory=True,
#     )

#     train_loader.num_classes = 1000
#     val_loader.num_classes = 1000

#     return train_loader, train_sampler, val_loader