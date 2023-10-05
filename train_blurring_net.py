import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
import os
import subprocess
import wandb
import argparse

##############################
## Hyperparameters
##############################
def get_args():
    """Get command-line arguments."""
    parser = argparse.ArgumentParser(description='Hyperparameters for blurring project')

    parser.add_argument('--blurring_strategy', type=str, default= ['sharp','first_few_epochs_exponentially_decreasing'][1])

    parser.add_argument('--show_progress_bar', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--batch_size_val_test', type=int, default=256)
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--n_epochs', type=int, default=150)
    parser.add_argument('--model_name', type=str, default='resnet50')
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--id', type=int, default=0)

    return parser.parse_args()

def get_hyp(args):
    """Return hyperparameters as a dictionary."""

    return {
        'dataset': {
            'name': 'texture2shape_miniecoset',
            'image_size': args.image_size,
            'dataset_path': '/home/student/l/lzejin/datasets/',
            'augment': {'randomrotation', 'randomflip', 'grayscale'}, # normalise happens in the blurring class for training
            'num_classes': 112,
        },
        'network': {
            'model': args.model_name,
            'identifier': f'id_{args.id}_lr_{args.learning_rate}',
            'blurring_strategy': args.blurring_strategy,
        },
        'optimizer': {
            'type': 'adam',
            'lr': args.learning_rate,
            'batch_size': args.batch_size,
            'n_epochs': args.n_epochs,
            'device': 'cuda',
            'dataloader': {
                'num_workers_train': 10,
                'prefetch_factor_train': 4,
                'num_workers_val_test': 3,
                'prefetch_factor_val_test': 4 
            },
            'show_progress_bar': args.show_progress_bar,
        },
        'misc': {
            'use_amp': True,
            'batch_size_val_test': args.batch_size_val_test,
            'save_logs': 5,
            'save_net': 5
        }
    }

##############################
## Loading the dataset loaders
##############################
def get_Dataset_loaders(hyp):
    """Return train, validation, and test dataloaders based on given hyperparameters."""
    if hyp['dataset']['name'] == 'texture2shape_miniecoset':
        dataset_path = f"{hyp['dataset']['dataset_path']}texture2shape_miniecoset_{hyp['dataset']['image_size']}px.h5"
        print(f"Loading dataset from: {dataset_path}")

        with h5py.File(dataset_path, "r") as f:
            hyp['dataset']['train_img_mean_channels'] = f['train_img_mean_channels'][()]
            hyp['dataset']['train_img_std_channels'] = f['train_img_std_channels'][()]

        train_transform = get_transform(hyp['dataset']['augment'], hyp)
        val_test_transform = get_transform(['normalize'], hyp)

        train_dataset = MiniEcoset('train', dataset_path, train_transform)
        val_dataset = MiniEcoset('val', dataset_path, val_test_transform)
        test_dataset = MiniEcoset('test', dataset_path, val_test_transform)

        hyp['dataset']['output_size'] = 112
    else:
        raise ValueError(f"Unknown dataset: {hyp['dataset']['name']}")

    # Create Dataloaders for the splits
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=hyp['optimizer']['batch_size'], shuffle=True,
                                                num_workers=hyp['optimizer']['dataloader']['num_workers_train'],
                                                prefetch_factor=hyp['optimizer']['dataloader']['prefetch_factor_train'])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=hyp['misc']['batch_size_val_test'],
                                                num_workers=hyp['optimizer']['dataloader']['num_workers_val_test'],
                                                prefetch_factor=hyp['optimizer']['dataloader']['prefetch_factor_val_test'])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=hyp['misc']['batch_size_val_test'],
                                                num_workers=hyp['optimizer']['dataloader']['num_workers_val_test'],
                                                prefetch_factor=hyp['optimizer']['dataloader']['prefetch_factor_val_test'])
            
    return train_loader, val_loader, test_loader, hyp


class MiniEcoset(torch.utils.data.Dataset):
    """A PyTorch dataset for MiniEcoset."""

    def __init__(self, split, dataset_path, transform=None):
        """
        Args:
            dataset_path (string): Path to the .h5 file
            transform (callable, optional): Optional transforms to be applied
                on a sample.
        """
        self.root_dir = dataset_path
        self.transform = transform

        with h5py.File(dataset_path, "r") as f:
            self.images = torch.from_numpy(f[split]['data'][()]).permute((0, 3, 1, 2)) # to match the CHW expectation of pytorch
            self.labels = torch.from_numpy(f[split]['labels'][()])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx): 
        # accepts ids and returns the images and labels transformed to the Dataloader
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imgs = self.images[idx]
        labels = self.labels[idx]

        if self.transform:
            imgs = self.transform(imgs)

        return imgs, labels
    
##############################
## transforms
##############################
def get_transform(aug_str,hyp=None):
    # Returns a transform compose function given the transforms listed in "aug_str"

    transform_list = []

    if 'randomrotation' in aug_str:
        transform_list.append(transforms.RandomRotation(degrees=45))
    if 'randomflip' in aug_str:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    if 'grayscale' in aug_str:
        transform_list.append(transforms.RandomGrayscale(p=0.5))
    transform_list.append(transforms.ConvertImageDtype(torch.float))
    if 'normalize' in aug_str:
        transform_list.append(transforms.Normalize(mean = hyp['dataset']['train_img_mean_channels']/255., std = hyp['dataset']['train_img_std_channels']/255.))

    transform = transforms.Compose(transform_list)
    
    return transform

##############################
## Logging functions
##############################
def ensure_directory_exists(directory_name):
    """Ensure the directory exists; if not, create it."""
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
        print(f'{directory_name} is created!')

def setup_logging_directories(net_name):
    """Set up log directories and return their paths."""

    print('Setting up log folders...')

    base_log_dir = 'logs/perf_logs'
    base_net_dir = 'logs/net_params'
    
    ensure_directory_exists(base_log_dir)
    ensure_directory_exists(base_net_dir)

    log_directory = os.path.join(base_log_dir, net_name)
    net_directory = os.path.join(base_net_dir, net_name)

    ensure_directory_exists(log_directory)
    ensure_directory_exists(net_directory)

    return log_directory, net_directory 


##################################
# move optimizer to cuda    (do we really need this?)
##################################
def move_optimizer_to_device(optim, device):
    """Move optimizer state to a device."""
    for state in optim.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


##################################
## Importing the network
##################################
def get_network_model(hyp):
    """Create a network model based on provided hyperparameters."""
    
    model_type = hyp['network']['model']
    dataset_name = hyp['dataset']['name']
    blurring_strategy = hyp['network']['blurring_strategy']
    identifier = hyp['network']['identifier']
    num_classes = hyp['dataset']['num_classes']

    if model_type == 'resnet50':
        model = torchvision.models.resnet50(pretrained=False)
        model.fc = nn.Linear(2048, num_classes)
        model_name = f'resnet50_not_pretrained_num_{identifier}_{dataset_name}_blurring_strategy_{blurring_strategy}'
    elif model_type == 'resnet18':
        model = torchvision.models.resnet18(pretrained=False)
        model.fc = nn.Linear(512, num_classes)
        model_name = f'resnet18_not_pretrained_num_{identifier}_{dataset_name}_blurring_strategy_{blurring_strategy}'
    else:
        raise ValueError(f"Unknown model: {model_type}")

    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nThe network has {num_trainable_params} trainable parameters\n")

    return model, model_name

##################################
## Initializing the network and optimizer
##################################
def initialize_weights(module):
    """Xavier weight initialization for given module."""
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(module.weight)

def get_optimizer(hyp, model):
    """Return the optimizer based on hyperparameters."""
    if hyp['optimizer']['type'] == 'adam':
        return optim.Adam(model.parameters(), lr=hyp['optimizer']['lr'])
    else:
        raise ValueError(f"Unknown optimizer: {hyp['optimizer']['type']}")

##################################
## evaluation
##################################
def compute_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    total_samples = labels.size(0)
    correct_predictions = (predicted == labels).sum().item()
    return 100.0 * correct_predictions / total_samples

def evaluate_model(data_loader, model, criterion, hyp, transform=None):
    total_loss = 0.0
    total_accuracy = 0.0 
    device = hyp['optimizer']['device']
    
    with torch.no_grad():
        for images, labels in data_loader:

            images = images.to(device).byte().half()
            labels = labels.to(device).long()
            if transform:
                images = transform(images)
            
            if device == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            total_accuracy += compute_accuracy(outputs, labels)

    return total_loss, total_accuracy


##########################
## Blurring strategy 
##########################

class CustomBlur(transforms.GaussianBlur): # normalise is done here, so we always need to call this transform
    def __init__(self, epoch, blurring_strategy, hyp):
        super().__init__(kernel_size=5)  # Initialize with a fixed kernel size
        self.blurring_strategy = blurring_strategy
        self.epoch = epoch
        self.hyp = hyp

    def forward(self, img):
        if self.blurring_strategy == "sharp":
            return transforms.Normalize(mean = self.hyp['dataset']['train_img_mean_channels']/255., std = self.hyp['dataset']['train_img_std_channels']/255.)(img)

        elif self.blurring_strategy in ["first_few_epochs_exponentially_decreasing", "fisrt_few_epochs_linearly_decreasing"] and self.epoch <= 20:
            if self.blurring_strategy == "first_few_epochs_exponentially_decreasing":
                sigma = 28 * (0.01 ** ((self.epoch-1) / 19))
            else:
                sigma = 28 - 1.4 * self.epoch + 0.001
            kernel_size = int(8*sigma) + (0 if int(8*sigma) % 2 else 1)
            return transforms.Normalize(mean = self.hyp['dataset']['train_img_mean_channels']/255., std = self.hyp['dataset']['train_img_std_channels']/255.)(transforms.functional.gaussian_blur(img, kernel_size=kernel_size, sigma=sigma))
        else:
            return transforms.Normalize(mean = self.hyp['dataset']['train_img_mean_channels']/255., std = self.hyp['dataset']['train_img_std_channels']/255.)(img)

##########################
## Loading the checkpoint and initializing or resuming  the logs
##########################
def load_checkpoint(net, net_name, net_path, log_path, hyp):
    """Load the latest checkpoint if available."""
    # Get the list of model files in the folder if available
    model_files = [f for f in os.listdir(net_path) if f.endswith(".pth")]
    if not model_files:
        return net, None, 1

    # Sort the model files by epoch number and Get the latest model file
    model_files.sort(key=lambda f: int(f.split("_")[-1].split(".")[0]))
    latest_model_file = model_files[-1]
    
    # Load log data
    log_prefix = f'/loss_ep{hyp["optimizer"]["n_epochs"]}'
    data = np.load(log_path + log_prefix + net_name + '.npz')
    logs = {
        key: (list(data[key]) if np.ndim(data[key]) > 0 else [data[key]]) + [default_val] * (hyp['optimizer']['n_epochs'] - len(data[key]) + 1)
        for key, default_val in [
            ("train_loss", 0), 
            ("val_loss", 0), 
            ("train_accuracies", 0), 
            ("val_accuracies", 0), 
            ("lrs", hyp['optimizer']['lr'])
        ]
    }
    data.close()

    # Load the model and optimizer and Load the optimizer state dict if available
    checkpoint = torch.load(os.path.join(net_path, latest_model_file))
    net.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    try:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        move_optimizer_to_device(optimizer, hyp['optimizer']['device'])  # Assuming this function is defined elsewhere
    except:
        pass

    start_epoch = int(latest_model_file.split("_")[-1].split(".")[0])
    return net, logs, start_epoch


def initialize_or_resume_training(net, net_name, hyp):
    """
    Initialize or resume training based on the presence of a checkpoint.
    
    Args:
    - net (torch.nn.Module): The network model.
    - net_name (str): The name of the network.
    - hyp (dict): Hyperparameters.
    
    Returns:
    - logs (dict): Logs dictionary.
    - start_epoch (int): The epoch to start/resume training from.
    """

    # Create folders for logging
    log_path, net_path = setup_logging_directories(net_name)
    print(f'Log_folders: {log_path} -- {net_path}')

    # Attempt to load a checkpoint if available
    net, logs, start_epoch = load_checkpoint(net, net_name, net_path, log_path, hyp)
    
    if not logs:
        logs = initialize_logs(hyp)
        start_epoch = 1
        net.apply(initialize_weights)
        print('\nTraining begins here from scratch!\n')
    else:
        print(f'\nTraining resumes from epoch {start_epoch}.\n')

    return net, start_epoch, log_path, net_path


def initialize_logs(hyp):
    """
    Initialize logs for a fresh training.
    """
    epochs = hyp['optimizer']['n_epochs']
    
    return {
        "train_losses": [0 for _ in range(epochs+1)],
        "val_losses": [0 for _ in range(epochs+1)],
        "train_accuracies": [0 for _ in range(epochs+1)],
        "val_accuracies": [0 for _ in range(epochs+1)],
        "lrs": [hyp['optimizer']['lr'] for _ in range(epochs+1)]
    }

def log_and_save_metrics(epoch, train_loss, train_acc, val_loss, val_acc, net, optimizer, log_path, net_name, hyp):
    """
    Log the training and validation metrics, and save the model and logs periodically.
    
    Args:
    - epoch (int): The current epoch.
    - train_loss (float): Training loss for the current epoch.
    - train_acc (float): Training accuracy for the current epoch.
    - val_loss (float): Validation loss for the current epoch.
    - val_acc (float): Validation accuracy for the current epoch.
    - net (torch.nn.Module): The network model.
    - optimizer (torch.optim.Optimizer): The optimizer.
    - log_path (str): Path to the logging directory.
    - net_name (str): The name of the network.
    - hyp (dict): Hyperparameters.
    """

    # Print metrics
    print(f'Train loss: {train_loss:.2f}; acc: {train_acc:.2f}%')
    print(f'Val loss: {val_loss:.2f}; acc: {val_acc:.2f}%\n')
    
    # Log metrics to wandb
    # wandb.log({
    #     "train_loss": train_loss, 
    #     "val_loss": train_acc,
    #     "train_acc": val_loss,
    #     "val_acc": val_acc,
    #     "lr": optimizer.param_groups[0]['lr']
    # })

    # Save logs
    if (epoch + 1) % hyp['misc']['save_logs'] == 0:
        np.savez(log_path + f'/loss_ep{hyp["optimizer"]["n_epochs"]}_' + net_name + '.npz', 
                 train_loss=train_loss, 
                 val_loss=val_loss, 
                 train_accuracies=train_acc, 
                 val_accuracies=val_acc, 
                 lrs=optimizer.param_groups[0]['lr'])

    # Save model
    if (epoch + 1) % hyp['misc']['save_net'] == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_history': train_loss,
            'val_loss_history': val_loss
        }, f'{log_path}/{net_name}_epoch_{epoch}.pth')


##########################
## Training and evaluation
##########################
def train_epoch(epoch, net, train_loader, optimizer, criterion, scaler, blur_transform, show_progress_bar=True):
    """
    Train the network for one epoch.
    
    Args:
    - epoch (int): Current epoch number.
    - net (torch.nn.Module): Neural network model.
    - train_loader (DataLoader): Training data loader.
    - optimizer (torch.optim.Optimizer): Optimizer.
    - criterion (torch.nn.Module): Loss function.
    - scaler (torch.cuda.amp.GradScaler): Gradient scaler for mixed precision training.
    - show_progress_bar (bool): Flag to show training progress bar.
    
    Returns:
    - tuple: Average training loss and accuracy for the epoch.
    """
    net.train()
    net = net.float().to(hyp['optimizer']['device'])
    train_loss_running, train_acc_running, batch = 0.0, 0.0, 0
    start = time.time()
    
    for images, labels in train_loader:
        # Move images and labels to the desired device.
        imgs, lbls = images.to(hyp['optimizer']['device']), labels.to(hyp['optimizer']['device'])
        
        # Apply transformations(norm after blurring).
        imgs = blur_transform(imgs)
        
        # Compute the forward pass and loss.
        if hyp['optimizer']['device'] == 'cuda':
             with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=hyp['misc']['use_amp']):
                outputs = net(imgs)
                loss = criterion(outputs, lbls.long())
        else:
            outputs = net(imgs)
            loss = criterion(outputs, lbls.long())
        
        # Backward pass and optimization.
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Update training stats.
        train_loss_running += loss.item()
        train_acc_running += compute_accuracy(outputs, lbls)
        
        batch += 1
        if show_progress_bar:
            print(f'Training Epoch {epoch}: Batch {batch} of {len(train_loader)}', end="\r")
    
    avg_train_loss = train_loss_running / len(train_loader)
    avg_train_acc = train_acc_running / len(train_loader)
    
    print(f'\nEpoch time: {time.time() - start:.2f} seconds')
    return avg_train_loss, avg_train_acc

def validate_epoch(model, val_loader, criterion, hyp):
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
    val_loss, val_acc = evaluate_model(val_loader, model, criterion, hyp)
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = np.mean(val_acc) / len(val_loader)
    return avg_val_loss, avg_val_acc


def evaluate_on_test_set(net, test_loader, criterion, hyp):
    """
    Evaluate the network on the test set and print the accuracy.
    
    """
    net.eval()
    _, test_acc_running = evaluate_model(test_loader, net, criterion, hyp)

    test_acc = test_acc_running / len(test_loader)
    print(f'Test accuracy: {test_acc:.2f}%')


if __name__ == '__main__':
    # Get the hyperparameters
    args = get_args()
    hyp = get_hyp(args)

    # Start a new wandb run to track this script
    # wandb.init(project="Blurring_projecy", config=hyp)

    # # Set the seed for reproducibility
    # torch.manual_seed(1234)

    # Load the datasets
    train_loader, val_loader, test_loader, hyp = get_Dataset_loaders(hyp)

    # Initialize the network and optimizer
    net, net_name = get_network_model(hyp)
    optimizer = get_optimizer(hyp, net)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=hyp['misc']['use_amp'])

    # initialize_or_resume_training from logs
    net, start_epoch, log_path, net_path = initialize_or_resume_training(net, net_name, hyp)

    # Train the network
    for epoch in range(start_epoch, hyp['optimizer']['n_epochs']+1):

        blur_transform = CustomBlur(epoch=epoch, blurring_strategy=hyp['network']['blurring_strategy'])
        train_loss, train_acc = train_epoch(epoch, net, train_loader, optimizer, criterion, scaler, \
                                    blur_transform)
        val_loss, val_acc = validate_epoch(net, val_loader, criterion,hyp)
        
        # Log and save metrics
        log_and_save_metrics(epoch, train_loss, train_acc, val_loss, val_acc, net, optimizer, log_path, net_name, hyp)


    print('\nDone training and evaluating in testset now!\n')
    if os.path.exists(f'{net_path}/{net_name}_epoch_{epoch}.pth'):
        torch.save(net.state_dict(), f'{net_path}/{net_name}_epoch_{epoch}.pth')
    evaluate_on_test_set(net, test_loader, criterion, hyp)














