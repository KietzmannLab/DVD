import os
import pathlib

import torch
import torchvision.transforms as transforms
from robustness import datasets
from robustness.tools.imagenet_helpers import common_superclass_wnid, ImageNetHierarchy
from torchvision.datasets import ImageFolder

current_dir = pathlib.Path(os.path.abspath(__file__)).parent

num_classes = 16
num_channels = 3
height = 224
width = 224

categories = sorted(
    [
        "knife",
        "keyboard",
        "elephant",
        "bicycle",
        "airplane",
        "clock",
        "oven",
        "chair",
        "bear",
        "boat",
        "cat",
        "bottle",
        "truck",
        "car",
        "bird",
        "dog",
    ]
)
label_map = {k: v for k, v in enumerate(categories)}


def load_imagenet16(
    imagenet_path: str = "/share/klab/datasets/imagenet/",
    batch_size: int = 32,
    info_path: str = str(current_dir) + "/info/",
    normalization: bool = True,
    train_transforms: bool = None, # add by zejin
    test_transforms: bool = None, # add by zejin
):
    """
    load 16-class-ImageNet
    Arguments:
        batch_size (int): the batch size used in training and test
        imagenet_path (str): the path to ImageNet
        info_path (str): the path to the directory that contains
                    imagenet_class_index.json, wordnet.is_a.txt, words.txt
        normalization (bool): Use normalization on images or nor. (default: True)
    Returns: train_loader, test_loader
    """

    # 16-class-imagenet
    in_hier = ImageNetHierarchy(imagenet_path, info_path)
    superclass_wnid = common_superclass_wnid("geirhos_16")
    class_ranges, label_map = in_hier.get_subclasses(superclass_wnid, balanced=True)

    custom_dataset = datasets.CustomImageNet(imagenet_path, class_ranges)
    # data augumentation for imagenet in robustness library is:
    # https://github.com/MadryLab/robustness/blob/master/robustness/data_augmentation.py

    # convert half percision
    custom_dataset.transform_train.transforms.append(transforms.ConvertImageDtype(torch.float16))



    # import pdb; pdb.set_trace();
    if train_transforms.transforms and len(train_transforms.transforms) >= 1:
    #* Clean the previous default image transforms, using our own transforms    
        custom_dataset.transform_train.transforms = [transforms.Resize((224, 224))] #! ImageSize might be differernt, e.g. got [3, 333, 500] at entry 0 and [3, 450, 600] at entry 2
        for train_transfrom in train_transforms.transforms:
            custom_dataset.transform_train.transforms.append(train_transfrom)
    if test_transforms and len(test_transforms.transforms) >= 1:
        custom_dataset.transform_test.transforms = [transforms.Resize((224, 224))]
        for test_transfrom in test_transforms.transforms:
            custom_dataset.transform_test.transforms.append(test_transfrom)

    # import pdb; pdb.set_trace(); 
    train_loader, test_loader = custom_dataset.make_loaders(
        workers=10, batch_size=batch_size
    )

    train_loader.num_classes = 16
    test_loader.num_classes = 16

    train_loader.num_images = 40517
    test_loader.num_images = 1600

    return train_loader, test_loader

def make_local_in16_test_loader(
    data_path: str = "/share/klab/datasets/imagenet/test/",
    batch_size: int = 32,
    normalization: bool = True,
    shuffle: bool = False,
):
    """
    Args:
        data_path: path to the directory that contains imagenet16
        batch_size: the size of each batch set

    Returns: test_loader
    """
    # standard ImageNet normalization:
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    transform_list = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]

    if normalization:
        transform_list.append(normalize)

    # data augmentations for imagenet in the `robustness` library is here:
    # https://github.com/MadryLab/robustness/blob/master/robustness/data_augmentation.py
    dataset = ImageFolder(
        data_path,
        transforms.Compose(transform_list),
    )

    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=5, pin_memory=True
    )

    test_loader.num_classes = 16
    test_loader.num_images = 1600

    return test_loader


def make_test_images_by_class(
    dataset_path: str = "/share/klab/datasets/imagenet/", num_images: int = 10
) -> torch.Tensor:
    """Makes test images along class labels.
    Args:
        num_images (int): number of images for each class. Default: 10

    Returns: test images (num_classes, N, C, H, W)
        where: num_classes = 16
    """
    _, test_loader = load_imagenet16(imagenet_path=dataset_path, batch_size=32)

    counts = torch.zeros(num_classes)
    test_images = torch.zeros([num_classes, num_images, num_channels, height, width])
    for images, labels in test_loader:
        for image, label in zip(images, labels):
            label_id = label.item()
            if counts[label_id] < num_images:
                test_images[label_id][int(counts[label_id])] = image
                counts[label_id] += 1

    return test_images