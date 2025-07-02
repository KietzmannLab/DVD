import math
import os
import shutil
import sys
import yaml
import numpy as np
import torch
import json
from collections import defaultdict, Counter
import torch.nn as nn
import torch.distributed as dist

# Define a helper to save args to a YAML file
def save_config(args, filename):
    with open(filename, 'w') as f:
        # Convert Namespace to a dict with vars(args)
        yaml.dump(vars(args), f)
    print(f"Saving config to {filename}")

# Define a helper to load the configuration from a YAML file
def load_config(filename):
    with open(filename, 'r') as f:
        return yaml.safe_load(f)

def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    # launched with torch.distributed.launch
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    # launched with submitit on a slurm cluster
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    # launched naively with `python main_dino.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print("Will run the code on one GPU.")
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
    else:
        print("Does not support training without GPU.")
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print(
        "| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        return "\t".join(entries)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def save_checkpoint(state, is_best, filename="checkpoint.pth"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(os.path.dirname(filename), "checkpoint_best.pth"))

def save_last_checkpoint(state, filename="checkpoint_last.pth"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)

def save_initial_checkpoint(log_dir, args, model, optimizer, scaler, logger, net_name):
    """
    Saves an initial (random) checkpoint to the specified log directory.

    Parameters:
        log_dir (str): The base directory where the checkpoint will be saved.
        args: An object containing model architecture details (expects attribute 'arch').
        model: The model whose state will be saved.
        optimizer: The optimizer whose state will be saved.
        scaler: The scaler whose state will be saved.
        logger: A logger instance for logging messages.
    """
    # Create the weights directory if it doesn't exist
    weights_dir = os.path.join(log_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    # Define the path for the initial checkpoint file
    init_checkpoint_path = os.path.join(weights_dir, "checkpoint_init.pth")
    
    # Save the checkpoint using evd's utility function
    save_checkpoint(
        {
            "epoch": -1,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
        },
        is_best=False,
        filename=init_checkpoint_path,
    )
    
    # Log the checkpoint saving event
    logger.info(f"Saved initial (random) checkpoint at {init_checkpoint_path}")

def adjust_learning_rate(optimizer, epoch, args):
    """Decays the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = (
            args.lr
            * 0.5
            * (
                1.0
                + math.cos(
                    math.pi
                    * (epoch - args.warmup_epochs)
                    / (args.epochs - args.warmup_epochs)
                )
            )
        )
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


############################################################################################################\
## Loss function with class weights
############################################################################################################

def get_loss_function(args):
    """
    Set up the loss function (criterion) based on the dataset and hyperparameters.
    Only used for supervised learning now.

    Args:
        hyp (dict): Hyperparameters, including dataset details and class weights.

    Returns:
        torch.nn.Module: The loss function configured based on the dataset and hyperparameters.
    """
    dataset_name = args.dataset_name
    class_weights_json_path = args.class_weights_json_path


    def create_criterion_with_weights(weights):
        """Helper to create a loss function with class weights."""
        return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, weight=weights).cuda(args.gpu)

    # Handle specific datasets with class weights
    if class_weights_json_path and dataset_name in ['ecoset_square256_patches', 'ecoset_square256']:
        print(f"Loading class weights from {class_weights_json_path}")
        class_weights_dict = class_weights_from_json(class_weights_json_path, normalize='max')
        num_classes = max(class_weights_dict.keys()) + 1
        class_weights_tensor = torch.tensor(
            [class_weights_dict[i] for i in range(num_classes)],
            dtype=torch.float32
        ).cuda(args.gpu)
        return create_criterion_with_weights(class_weights_tensor)

    if dataset_name == 'imagenet':
        from torchvision.datasets import ImageFolder

        imagenet_path = "/share/klab/datasets/imagenet/train"  # Adjust as needed
        print(f"Calculating class weights from {imagenet_path} for ImageNet")
        imagenet_train_dataset = ImageFolder(root=imagenet_path)
        class_weights = calculate_class_weights_from_imagefolder(imagenet_train_dataset).cuda(args.gpu)
        return create_criterion_with_weights(class_weights)

    # Default criterion for other datasets
    return create_criterion_with_weights(None)


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


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)