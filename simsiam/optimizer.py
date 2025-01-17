# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# Implemented by @ananyahjha93
# also found at: https://github.com/gridai-labs/aavae/tree/main/src/optimizers
# References:
#     - https://arxiv.org/pdf/1708.03888.pdf
#     - https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py
import torch
from torch.optim.optimizer import Optimizer, required

from flash.core.utilities.imports import _TOPIC_CORE_AVAILABLE

import json
from collections import defaultdict
from collections import Counter
from torchvision.datasets import ImageFolder
import numpy as np


# Skip doctests if requirements aren't available
if not _TOPIC_CORE_AVAILABLE:
    __doctest_skip__ = ["LARS"]


class LARS(Optimizer):
    r"""Extends SGD in PyTorch with LARS scaling.

    See the paper `Large batch training of Convolutional Networks <https://arxiv.org/pdf/1708.03888.pdf>`_

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)
        trust_coefficient (float, optional): trust coefficient for computing LR (default: 0.001)
        eps (float, optional): eps for division denominator (default: 1e-8)

    Example:
        >>> from torch import nn
        >>> model = nn.Linear(10, 1)
        >>> optimizer = LARS(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> # loss_fn(model(input), target).backward()
        >>> optimizer.step()

    .. note::
        The application of momentum in the SGD part is modified according to
        the PyTorch standards. LARS scaling fits into the equation in the
        following fashion.

        .. math::
            \begin{aligned}
                g_{t+1} & = \text{lars\_lr} * (\beta * p_{t} + g_{t+1}), \\
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}

        where :math:`p`, :math:`g`, :math:`v`, :math:`\mu` and :math:`\beta` denote the
        parameters, gradient, velocity, momentum, and weight decay respectively.
        The :math:`lars_lr` is defined by Eq. 6 in the paper.
        The Nesterov version is analogously modified.

    .. warning::
        Parameters with weight decay set to 0 will automatically be excluded from
        layer-wise LR scaling. This is to ensure consistency with papers like SimCLR
        and BYOL.
    """

    def __init__(
        self,
        params,
        lr=required,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        trust_coefficient: float = 0.001,
        eps: float = 1e-8,
    ):
        if lr is not required and lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
        }
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        self.eps = eps
        self.trust_coefficient = trust_coefficient

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)

        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # exclude scaling for params with 0 weight decay
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                d_p = p.grad
                p_norm = torch.norm(p.data)
                g_norm = torch.norm(p.grad.data)

                # lars scaling + weight decay part
                if weight_decay != 0 and p_norm != 0 and g_norm != 0:
                    lars_lr = p_norm / (g_norm + p_norm * weight_decay + self.eps)
                    lars_lr *= self.trust_coefficient

                    d_p = d_p.add(p, alpha=weight_decay)
                    d_p *= lars_lr

                # sgd part
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    d_p = d_p.add(buf, alpha=momentum) if nesterov else buf

                p.add_(d_p, alpha=-group["lr"])

        return loss



def class_weights_from_json(json_path, normalize=None):
        with open(json_path) as file:
            data = json.load(file)
        category_image_count = defaultdict(int)
            
        for k, v in data.items():
            
            zero_indexed_category_label = int(k) - 1
            images_in_category = data[k]["images"]
            category_image_count[zero_indexed_category_label] = int(images_in_category)
            
        max_key = max(category_image_count, key=category_image_count.get)
        min_key = min(category_image_count, key=category_image_count.get)    
        print(f'Max images per class {max(category_image_count.values())} for class label {max_key}, Min images per class {min(category_image_count.values())} for class label {min_key}')
        total_classes = len(data.keys())    
        total_images = sum(category_image_count.values())  
        class_weights = {
            category_label: total_images / (image_count * total_classes)
            for category_label, image_count in category_image_count.items()
        }  
   
        
        if normalize == 'max':
            max_weight = max(class_weights.values())
            class_weights = {
                category_label: weight / max_weight
                for category_label, weight in class_weights.items()
            }
        elif normalize == 'sum':
            weight_sum = sum(class_weights.values())
            class_weights = {
                category_label: weight / weight_sum
                for category_label, weight in class_weights.items()
            }
        
        return class_weights

def calculate_class_weights_from_imagefolder(dataset):
    """
    Calculate class weights for CrossEntropyLoss based on the dataset loaded with ImageFolder.

    Args:
        dataset (torchvision.datasets.ImageFolder): Dataset loaded using ImageFolder.

    Returns:
        torch.Tensor: Tensor of class weights to use with CrossEntropyLoss.
    """
    # Get the list of labels for all samples in the dataset
    labels = [sample[1] for sample in dataset.samples]

    # Count occurrences of each class
    class_counts = Counter(labels)

    # Get total number of samples
    total_samples = sum(class_counts.values())

    # Calculate class weights: inverse proportional to class frequency
    num_classes = len(class_counts)
    class_weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]

    # Print min and max counts
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    print(f"Minimum count per class: {min_count}")
    print(f"Maximum count per class: {max_count}")

    # Normalize weights (optional, depends on preference)
    class_weights = np.array(class_weights) / sum(class_weights)

    # Convert to a tensor for use in PyTorch
    return torch.tensor(class_weights, dtype=torch.float)



def get_loss_function(dataset_name, class_weights_json_path):
    """Set up the loss function (criterion) based on the dataset and hyperparameters."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load class weights for different datasets
    if class_weights_json_path and dataset_name in ['ecoset_square256_patches', 'ecoset_square256']:
        # Load class weights from JSON file
        print(f"Loading class weights from {class_weights_json_path}")
        class_weights_dict = class_weights_from_json(class_weights_json_path, normalize='max')

        # Convert class weights dictionary to a sorted list
        num_classes = max(class_weights_dict.keys()) + 1  # Assuming class indices start at 0
        class_weights_list = [class_weights_dict[i] for i in range(num_classes)]

        # Convert the list to a PyTorch tensor
        class_weights_tensor = torch.tensor(class_weights_list, dtype=torch.float32).to(device)

        # Define the loss function with class weights
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights_tensor)

    elif dataset_name == 'imagenet':
        # Calculate class weights for ImageNet
        imagenet_path = "/share/klab/datasets/imagenet/train"  # Adjust the path as necessary
        print(f"Calculating class weights from {imagenet_path} for ImageNet")

        # Load the ImageNet training dataset
        imagenet_train_dataset = ImageFolder(root=imagenet_path)
        class_weights = calculate_class_weights_from_imagefolder(imagenet_train_dataset).to(device)

        # Define the loss function with class weights
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    else:
        # Default loss function without class weights
        criterion = torch.nn.CrossEntropyLoss()

    return criterion

def calculate_class_weights_from_imagefolder(dataset):
    """
    Calculate class weights for CrossEntropyLoss based on the dataset loaded with ImageFolder.

    Args:
        dataset (torchvision.datasets.ImageFolder): Dataset loaded using ImageFolder.

    Returns:
        torch.Tensor: Tensor of class weights to use with CrossEntropyLoss.
    """
    # Get the list of labels for all samples in the dataset
    labels = [sample[1] for sample in dataset.samples]

    # Count occurrences of each class
    class_counts = Counter(labels)

    # Get total number of samples
    total_samples = sum(class_counts.values())

    # Calculate class weights: inverse proportional to class frequency
    num_classes = len(class_counts)
    class_weights = [total_samples / (num_classes * class_counts[i]) for i in range(num_classes)]

    # Print min and max counts
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    print(f"Minimum count per class: {min_count}")
    print(f"Maximum count per class: {max_count}")

    # Normalize weights (optional, depends on preference)
    class_weights = np.array(class_weights) / sum(class_weights)

    # Convert to a tensor for use in PyTorch
    return torch.tensor(class_weights, dtype=torch.float)