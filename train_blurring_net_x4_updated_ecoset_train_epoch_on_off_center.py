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
import random
import torch.nn.functional as F
import cv2
from PIL import Image, ImageFilter, ImageChops

##############################
## Hyperparameters
##############################
def get_args():
    """Get command-line arguments."""
    parser = argparse.ArgumentParser(description='Hyperparameters for blurring project')

    parser.add_argument('--blurring_strategy', type=str, default= ['sharp','first_few_epochs_exponentially_decreasing'][1])
    parser.add_argument('--blur_norm_order', type=str, default= ['blur_first','norm_first'][0])
    parser.add_argument('--norm_flag', type=int, default= 1)
    parser.add_argument('--byte_flag', type=str, default= ['use_byte','no_byte',
                                                            'strong_leaky_relu', 'strong_flatter_leaky_relu', 'strong_flipped_leaky_relu','strong_rising_leaky_relu'
                                                            'strong_no_int_leaky_relu', 'negative_weak_flatter_leaky_relu', 'positive_weak_flatter_leaky_relu',
                                                            'adjusted_symmetric_relu','relu', 
                                                            'int_step', '6_channel_negative_255_step','mixed_byte','u_step'
                                                            'negative_255_step', 'positive_255_step',
                                                            'negative_255_square', 'positive_255_square', 'negative_square', 'positive_square',
                                                            'sigmoid_byte', "sigmoid", "tanh", "tahn_byte", "tahn_int",
                                                            'flipped_sigmoid_byte', 'flipped_sigmoid', 'flipped_tanh', 'flipped_tanh_byte', "flipped_tanh_int",
                                                            
                                                            'positive_square', 'negative_square',
                                                            'positive_step', 'negative_step',
                                                            'sigmoid', 'fliped_sigmoid',
                                                            'tanh', 'fliped_tanh',
                                                            'triangle', 'neagtive_triangle',
                                                            'relu', 'leaky_relu',
                                                            "positive_byte", 'negative_byte',
                                                            'use_value_x2_byte','use_value_x32_byte','use_value_x128_byte',
                                                            'byte_byte_plus_imgs', 'byte_plus_imgs','on_off_center', 'watershed_segmentation_rgb',
                                                            'no_norm_x5_byte','no_norm_x8_byte','DOG',
                                                            ][0])
    parser.add_argument('--on_off_kernel_size', type=int, default=3)
    parser.add_argument('--enhance_factor', type=int, default=[1,2,8][0]) 
    parser.add_argument('--dataset', type=str, default= ['texture2shape_miniecoset','ecoset_square256'][0])
    parser.add_argument('--pretrained', type=int, default=0)
    parser.add_argument('--uniform_blur', type=int, default=0) # blur first 20 batch for every epoch

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
            'name': args.dataset,
            'image_size': args.image_size,
            'dataset_path': '/home/student/l/lzejin/datasets/',
            'augment': {'randomrotation', 'randomflip', 'grayscale'}, # normalise happens in the blurring class for training
            'num_classes': [112,565,16][['texture2shape_miniecoset','ecoset_square256','imagenet16'].index(args.dataset)],
        },
        'network': {
            'model': args.model_name,
            'identifier': f'id_{args.id}_lr_{args.learning_rate}',
            'blurring_strategy': args.blurring_strategy,
            'blur_norm_order': args.blur_norm_order,
            'norm_flag': args.norm_flag,
            'byte_flag': args.byte_flag,
            "enhance_factor":args.enhance_factor,
            'on_off_kernel_size': args.on_off_kernel_size,
            'pretrained': args.pretrained,
            'uniform_blur': args.uniform_blur,
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


###
## on-center and off-center kernel size
###
# Function to create on-center and off-center kernels for a given size

def rescale_to_01(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)


# def apply_gaussian_difference_batch(imgs, sigma1=1, sigma2=28, scale_to_0_1=False):
#     processed_imgs = []
    
#     for image in imgs:
#         # Convert each input image to a PIL Image
#         img_pil = transforms.functional.to_pil_image(image.clone().cpu())

#         # Apply Gaussian blurs with specified radii
#         gaussian1 = img_pil.filter(ImageFilter.GaussianBlur(radius=sigma1))
#         gaussian2 = img_pil.filter(ImageFilter.GaussianBlur(radius=sigma2))

#         # Calculate the difference between the two blurred imgs
#         result_image = ImageChops.difference(gaussian1, gaussian2)
        
#         # Convert to a PyTorch tensor
#         result_image = transforms.functional.to_tensor(result_image)
#         if scale_to_0_1:  
#             result_image /= 255.0  # Scale to the range [0, 1]

#         processed_imgs.append(result_image)
    
#     return processed_imgs

def apply_gaussian_difference_batch(imgs, sigma1=1, sigma2=28, scale_to_0_1=False):
    device = imgs.device  # Get the device (CPU or GPU) of the input imgs
    processed_imgs = []

    for image in imgs:
        # Convert the PyTorch tensor to a NumPy array
        img_np = image.cpu().numpy()
        
        # Apply Gaussian blurs using PyTorch operations on the GPU
        img_tensor = torch.from_numpy(img_np).to(device)
        kernel_size1 = int(8*sigma1) + (0 if int(8*sigma1) % 2 else 1)
        kernel_size2 = int(8*sigma2) + (0 if int(8*sigma2) % 2 else 1)
        gaussian1 = transforms.functional.gaussian_blur(img_tensor, kernel_size=kernel_size1, sigma=sigma1)
        gaussian2 = transforms.functional.gaussian_blur(img_tensor, kernel_size=kernel_size2, sigma=sigma2)

        # Calculate the difference between the two blurred imgs
        result_image = torch.abs(gaussian1 - gaussian2)
        
        if scale_to_0_1:
            result_image /= result_image.max()  # Scale to the range [0, 1]

        processed_imgs.append(result_image)

    return processed_imgs


def create_kernels(size, enhance_factor=1):
    """Create on-center and off-center kernels for a given size."""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Central value
    central = 1/((size-2)*(size-2)) *enhance_factor # Adjust as needed
    
    # Surrounding value
    surrounding = -1/(size*size-(size-2)*(size-2)) *enhance_factor  # Adjust as needed

    # Create on-center kernel
    on_center_kernel = torch.full((3, 1, size, size), surrounding, dtype=torch.float32).to(device)
    on_center_kernel[:, :, 1:size-1, 1:size-1] = central # the only center is size // 2

    # Create off-center kernel (reverse of on-center)
    off_center_kernel = -on_center_kernel.clone()
    off_center_kernel[:, :, 1:size-1, 1:size-1] = -central

    return on_center_kernel, off_center_kernel

def rescale_to_0_1(tensor):
    # Min-Max normalization
    min_val = tensor.min()
    max_val = tensor.max()
    tensor = (tensor - min_val) / (max_val - min_val)
    return tensor

def rescale_to_0_255(tensor):
    return tensor * 255.0

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
        if hyp['network']['norm_flag']:
            val_test_transform = get_transform(['normalize'], hyp)
        else:
            val_test_transform = get_transform([], hyp)

        train_dataset = MiniEcoset('train', dataset_path, train_transform)
        val_dataset = MiniEcoset('val', dataset_path, val_test_transform)
        test_dataset = MiniEcoset('test', dataset_path, val_test_transform)

        hyp['dataset']['output_size'] = 112

    elif hyp['dataset']['name'] == 'ecoset_square256':
        
        from pytorch_dataset_loaders.pytorch_datasets import Ecoset
        dataset_path = f"{hyp['dataset']['dataset_path']}ecoset_square{hyp['dataset']['image_size']}_chunked.h5"
        print(f"Loading dataset from: {dataset_path}")

        # with h5py.File(dataset_path, "r") as f:
        hyp['dataset']['train_img_mean_channels'] = np.array([0.49081137, 0.47463922, 0.44580941])*255.
        hyp['dataset']['train_img_std_channels'] = np.array([0.28261176, 0.27914157, 0.28713294])*255.

        

        train_transform = get_transform(hyp['dataset']['augment'], hyp)
        if hyp['network']['norm_flag']:
            val_test_transform = get_transform(['normalize'], hyp)
        else:
            val_test_transform = get_transform([], hyp)

        train_dataset = Ecoset('train', dataset_path, train_transform, label_transform=True)
        val_dataset = Ecoset('val', dataset_path, val_test_transform, label_transform=True)
        test_dataset = Ecoset('test', dataset_path, val_test_transform, label_transform=True)

        hyp['dataset']['output_size'] = 565

    elif hyp['dataset']['name'] == 'imagenet16':
        from datasets.imagenet16 import load_imagenet16
        imagenet_path= "/share/klab/datasets/imagenet/"
        
        hyp['dataset']['train_img_mean_channels'] = np.array([0.485, 0.456, 0.406])*255.
        hyp['dataset']['train_img_std_channels'] = np.array([0.229, 0.224, 0.225])*255.

        train_transform = get_transform(hyp['dataset']['augment'], hyp)
        if hyp['network']['norm_flag']:
            val_test_transform = get_transform(['normalize'], hyp)
        else:
            val_test_transform = get_transform([], hyp)

        train_loader, val_loader = load_imagenet16(imagenet_path=imagenet_path,
                                                    batch_size=hyp['optimizer']['batch_size'],
                                                    normalization = False, # Not setting norm here
                                                    train_transforms = train_transform,
                                                    test_transforms= val_test_transform,
        )
        
        hyp['dataset']['classes'] = 16
        hyp['dataset']['class_names'] = [b'knife', b'keyboard', b'elephant', b'bicycle', b'airplane', b'clock', b'oven', b'chair', b'bear', b'boat', b'cat', b'bottle', b'truck', b'car', b'bird', b'dog']
        # ['knife', 'keyboard', 'elephant', 'bicycle', 'airplane', 'clock', 'oven', 'chair', 'bear', 'boat', 'cat', 'bottle', 'truck', 'car', 'bird', 'dog']
        
        return train_loader, val_loader, val_loader, hyp #! no test data for imagenet
    
    elif hyp['dataset']['name'] == 'imagenet':
        raise NotImplementedError
    
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
            self.imgs = torch.from_numpy(f[split]['data'][()]).permute((0, 3, 1, 2)) # to match the CHW expectation of pytorch
            self.labels = torch.from_numpy(f[split]['labels'][()])
            # self.imgs = torch.from_numpy(f[split]['data'][()].astype(np.int8)).permute((0, 3, 1, 2))
            # self.labels = torch.from_numpy(f[split]['labels'][()].astype(np.int8))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx): 
        # accepts ids and returns the imgs and labels transformed to the Dataloader
        if torch.is_tensor(idx):
            idx = idx.tolist()

        imgs = self.imgs[idx]
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

def watershed_segmentation_rgb(img, save_path=None):
    # in the format array is expected by OpenCV.
    image = np.array(img.cpu().numpy())
    # Check if the image is grayscale (1 channel)
    
    if image.shape[-1] not in [3,1]:
        image = image.transpose((1, 2, 0))
    if image.shape[-1] == 1:
        return watershed_segmentation_single_channel(image, save_path)
    if image is None:
        raise ValueError("Failed to read the image.")

    # Separate the image into RGB channels
    b, g, r = cv2.split(image) #image[0,:,:],image[1,:,:], image[2,:,:] #cv2.split(image) #  the shape is (3, 224, 224) as you mentioned, it means that image is already a 3-channel RGB imag

    # Apply watershed segmentation to each channel
    segmented_b = watershed_segmentation_single_channel(b)
    segmented_g = watershed_segmentation_single_channel(g)
    segmented_r = watershed_segmentation_single_channel(r)

    # Merge the segmented channels back into an RGB image
    segmented_image = cv2.merge((segmented_b, segmented_g, segmented_r))

    # Convert the segmented image to a PIL image
    pil_image = Image.fromarray(segmented_image)

    # Save the segmented image if save_path is provided
    if save_path:
        pil_image.save(save_path)

    return pil_image

def watershed_segmentation_single_channel(channel, save_path=None):
    # Convert the channel to grayscale
    # Check if the channel has 2 dimensions, and if so, add a third dimension
    if len(channel.shape) == 2:
        channel = np.expand_dims(channel, axis=-1)
     # Check if the channel values are in the range [0, 1]
    if np.min(channel) >= 0 and np.max(channel) <= 1:
        # Rescale the channel from [0, 1] to [0, 255]
        channel = (channel * 255).astype(np.uint8)
    gray = channel.astype(np.uint8)  # Convert to 8-bit unsigned integer


    # Apply thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove noise
    kernel = np.ones((3, 3), np.uint8)
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    opening = thresh # with noise

    # Determine background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Find the unknown area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Directly segment the image
    sure_fg = np.uint8(sure_fg)
    segmented_channel = cv2.subtract(sure_bg, sure_fg)

    # # Convert the segmented channel to a PIL image if needed
    # pil_image = Image.fromarray(segmented_channel)

    return segmented_channel

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

        if hyp['network']['pretrained']:
            pretrained_path = "/home/student/l/lzejin/codebase/blurring4texture2shape1file/logs/perf_logs/resnet50_not_pretrained_num_id_4018_lr_0.001_texture2shape_miniecoset_blurring_strategy_first_few_epochs_exponentially_decreasing/resnet50_not_pretrained_num_id_4018_lr_0.001_texture2shape_miniecoset_blurring_strategy_first_few_epochs_exponentially_decreasing_epoch_99.pth"
            model = torchvision.models.resnet50(pretrained=False)
            model.fc = nn.Linear(2048, 112)
            model.load_state_dict(torch.load(pretrained_path)['model_state_dict'])
            model.fc = nn.Linear(2048, num_classes)
        else:
            model = torchvision.models.resnet50(pretrained=False)
            if hyp['network']['byte_flag'] in ['on_off_center', 'byte_plus_imgs', 'mixed_byte', '6_channel_negative_255_step']:
                # Replace the first convolution layer with the modified layer
                num_channels = 6
                model.conv1 = torch.nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
                #TODO no need to set params?
            if hyp['network']['byte_flag'] in ['byte_byte_plus_imgs']:
                num_channels = 9
                model.conv1 = torch.nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

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
        for imgs, labels in data_loader:
            if hyp["network"]["byte_flag"]=='use_byte':
                imgs = imgs.to(device).byte().half() #! since training also have this, we need to do this here
            elif hyp["network"]["byte_flag"]=='positive_byte':
                m = nn.ReLU()
                imgs = m(imgs.to(device)).byte().half()
            elif hyp["network"]["byte_flag"]=='negative_byte':
                m = nn.ReLU()
                imgs = -m(-imgs.to(device)).byte().half()
            elif hyp["network"]["byte_flag"]=='mixed_byte':
                m = nn.ReLU()
                imgs = torch.cat((m(imgs).byte(), m(-imgs).byte()), dim=1).to(device).half()
            
            
            
            
            elif hyp["network"]["byte_flag"]=='u_step':
                imgs[(imgs <= -1)] = 3
                imgs[(imgs <= -2)] = 4
                imgs[(imgs > -1) & (imgs < 1)] = 0
                imgs[(imgs >= 1)] = 1
                imgs[(imgs >= 2)] = 2
                imgs = imgs.to(device).half()
            #* Promising?
            elif hyp["network"]["byte_flag"]=='int_step':
                '''-2-1, 000, 1-2'''
                imgs = imgs.to(device).int().half()
            elif hyp["network"]["byte_flag"]=='6_channel_negative_255_step':
                '''254,255, 000, 1-2 --> equal to byte'''
                imgs_1, imgs_2 = imgs.clone(), imgs.clone()
                imgs_1[(imgs_1 <= -1) & (imgs_1 > -2)] = 255
                imgs_1[(imgs_1 <= -2)] += 254
                imgs_1[(imgs_1 > -1)] = 0
                imgs_1 = imgs_1.to(device).int().half()

                imgs_2[(imgs_2 < 1)] = 0
                imgs_2 = imgs_2.to(device).int().half()

                imgs = torch.cat((imgs_1, imgs_2), dim=1).to(device).half()
            elif hyp["network"]["byte_flag"]=='negative_255_step':
                '''254,255, 000, 1-2 --> equal to byte'''
                imgs[(imgs <= -1) & (imgs > -2)] = 255
                imgs[(imgs <= -2)] += 254
                imgs = imgs.to(device).int().half()
            elif hyp["network"]["byte_flag"]=='positive_255_step':
                '''-2-1, 000, 254, 255'''
                imgs[(imgs < 2) & (imgs > 1)] = 254
                imgs[(imgs >= 2)] += 255
                imgs = imgs.to(device).int().half()

            elif hyp["network"]["byte_flag"]=='relu':
                m = nn.ReLU()
                imgs = m(imgs).to(device).half()
            elif hyp["network"]["byte_flag"]=='strong_leaky_relu':
                imgs[(imgs <= -1)] *= -100 
                imgs[(imgs <= -2)] *= -100 
                imgs[(imgs >= 1)] *= 1 
                imgs[(imgs >= 2)] *= 1 
                imgs[(imgs > -1) & (imgs < 1)] = 0
                imgs = imgs.to(device).int().half() #! Notice that we done the int() 
            elif hyp["network"]["byte_flag"]=='strong_flatter_leaky_relu':
                imgs[(imgs <= -1)] = -100 * (imgs[(imgs <= -1)]+1)
                imgs[(imgs >= 1)] *= 1
                imgs[(imgs > -1) & (imgs < 1)] = 0
                imgs = imgs.to(device).int().half()  #! Notice that we done the int()

            elif hyp["network"]["byte_flag"]=='custom_steep_double_sigmoid':
                def custom_steep_double_sigmoid(x, x0=-1, k=0.05):
                    x0_tensor = torch.tensor(x0, dtype=torch.float32, device=x.device)
                    k_tensor = torch.tensor(k, dtype=torch.float32, device=x.device)
                    y = torch.where(x < 0,
                                    255 / (1 + torch.exp((x - x0_tensor) / k_tensor)) + 2,
                                    -2 / (1 + torch.exp((-x - x0_tensor) / k_tensor)) + 2)
                    y = torch.clamp(y, 0, 255)
                    return y.to(device).int().half()

                imgs = custom_steep_double_sigmoid(x, x0=-1, k=0.05) #! Notice that we done the int()
            elif hyp["network"]["byte_flag"]=='custom_steep_no_clamp_double_sigmoid':
                def custom_steep_no_clamp_double_sigmoid(x, x0=-1, k=0.05):
                    x0_tensor = torch.tensor(x0, dtype=torch.float32, device=x.device)
                    k_tensor = torch.tensor(k, dtype=torch.float32, device=x.device)
                    y = torch.where(x < 0,
                                    255 / (1 + torch.exp((x - x0_tensor) / k_tensor)) + 2,
                                    -2 / (1 + torch.exp((-x - x0_tensor) / k_tensor)) + 2)
                    # y = torch.clamp(y, 0, 255)
                    return y.to(device).int().half()
                imgs = custom_steep_no_clamp_double_sigmoid(x, x0=-1, k=0.05) #! Notice that we done the int()
            elif hyp["network"]["byte_flag"]=='custom_steep_positive_double_sigmoid':
                def custom_steep_positive_double_sigmoid(x, x0=-1, k=0.05):
                    x0_tensor = torch.tensor(x0, dtype=torch.float32, device=x.device)
                    k_tensor = torch.tensor(k, dtype=torch.float32, device=x.device)
                    y = torch.where(x < 0,
                                    255 / (1 + torch.exp((x - x0_tensor) / k_tensor)),
                                    2 / (1 + torch.exp((-x - x0_tensor) / k_tensor)))
                    return y.to(device).int().half()
                imgs = custom_steep_positive_double_sigmoid(x, x0=-1, k=0.05) #! Notice that we done the int()
            elif hyp["network"]["byte_flag"]=='custom_shifted_minus_1_steep_sigmoid':
                def custom_shifted_minus_1_steep_sigmoid(x, x0=0, k=0.05, left_shift=-1, down_shift=-2):
                    x0_adjusted = x0 + left_shift
                    x0_tensor = torch.tensor(x0_adjusted, dtype=torch.float32, device=x.device)
                    k_tensor = torch.tensor(k, dtype=torch.float32, device=x.device)
                    y = 255 / (1 + torch.exp((x - x0_tensor) / k_tensor)) - down_shift
                    y = torch.clamp(y, 0, 255)
                    return y.to(device).int().half()
                imgs = custom_shifted_minus_1_steep_sigmoid(x, x0=0, k=0.05, left_shift=-1, down_shift=-2) #! Notice that we done the int()
            elif hyp["network"]["byte_flag"]=='custom_shifted_minus_1_not_steep_sigmoid':
                def custom_shifted_minus_1_not_steep_sigmoid(x, x0=0, k=0.2, left_shift=-1, down_shift=-2):
                    x0_adjusted = x0 + left_shift
                    x0_tensor = torch.tensor(x0_adjusted, dtype=torch.float32, device=x.device)
                    k_tensor = torch.tensor(k, dtype=torch.float32, device=x.device)
                    y = 255 / (1 + torch.exp((x - x0_tensor) / k_tensor)) - down_shift
                    y = torch.clamp(y, 0, 255)
                    return y.to(device).int().half()
                imgs = custom_shifted_minus_1_not_steep_sigmoid(x, x0=0, k=0.2, left_shift=-1, down_shift=-2) #! Notice that we done the int()
            elif hyp["network"]["byte_flag"]=='custom_shifted_steep_sigmoid':
                def custom_shifted_steep_sigmoid(x, x0=0, k=0.05, left_shift=-1, down_shift=-2):
                    x0_adjusted = x0 + left_shift
                    x0_tensor = torch.tensor(x0_adjusted, dtype=torch.float32, device=x.device)
                    k_tensor = torch.tensor(k, dtype=torch.float32, device=x.device)
                    y = 255 / (1 + torch.exp((x - x0_tensor) / k_tensor)) - down_shift
                    y = torch.clamp(y, 0, 255)
                    return y.to(device).int().half()
                imgs = custom_shifted_steep_sigmoid(x, x0=0, k=0.05, left_shift=-1, down_shift=-2) #! Notice that we done the int()
            elif hyp["network"]["byte_flag"]=='custom_shifted_not_steep_sigmoid':
                def custom_shifted_not_steep_sigmoid(x, x0=0, k=0.2, left_shift=0, down_shift=-2):
                    x0_adjusted = x0 + left_shift
                    x0_tensor = torch.tensor(x0_adjusted, dtype=torch.float32, device=x.device)
                    k_tensor = torch.tensor(k, dtype=torch.float32, device=x.device)
                    y = 255 / (1 + torch.exp((x - x0_tensor) / k_tensor)) - down_shift
                    y = torch.clamp(y, 0, 255)
                    return y.to(device).int().half()
                imgs = custom_shifted_not_steep_sigmoid(x, x0=0, k=0.2, left_shift=0, down_shift=-2)


            elif hyp["network"]["byte_flag"]=='strong_flipped_flatter_leaky_relu':
                imgs[(imgs <= -1)] = -1 * (imgs[(imgs <= -1)]+1)
                imgs[(imgs >= 1)] *= 100 
                imgs[(imgs > -1) & (imgs < 1)] = 0
                imgs = imgs.to(device).int().half() #! Notice that we done the int()
            elif hyp["network"]["byte_flag"]=='strong_flipped_leaky_relu':
                imgs[(imgs <= -1)] *= -1 
                imgs[(imgs >= 1)] *= 100
                imgs[(imgs > -1) & (imgs < 1)] = 0 
                imgs = imgs.to(device).int().half()   #! Notice that we done the int()
            elif hyp["network"]["byte_flag"]=='strong_rising_leaky_relu':
                imgs[(imgs <= -1)] = 100 * (imgs[(imgs <= -1)]+1)
                imgs[(imgs >= 1)] *= 1
                imgs[(imgs > -1) & (imgs < 1)] = 0
                imgs = imgs.to(device).int().half()   #! Notice that we done the int()
            elif hyp["network"]["byte_flag"]=='strong_decreasing_leaky_relu':
                imgs[(imgs <= -1)] = -100 * (imgs[(imgs <= -1)]+1)
                imgs[(imgs >= 1)] *= -1
                imgs[(imgs > -1) & (imgs < 1)] = 0
                imgs = imgs.to(device).int().half()   #! Notice that we done the int()
            elif hyp["network"]["byte_flag"]=='strong_no_int_leaky_relu':
                imgs[(imgs <= -1)] *= -100 
                imgs[(imgs <= -2)] *= -100 
                imgs[(imgs >= 1)] *= 1 
                imgs[(imgs >= 2)] *= 1 
                imgs[(imgs > -1) & (imgs < 1)] = 0
                imgs = imgs.to(device).half() #! Notice that we done the int()
            elif hyp["network"]["byte_flag"]=='negative_weak_flatter_leaky_relu':
                imgs[(imgs <= -1)] = -0.01 * (imgs[(imgs <= -1)]+1)
                imgs[(imgs >= 1)] *= 1
                imgs[(imgs > -1) & (imgs < 1)] = 0
                imgs = imgs.to(device).half() #* no int()
            elif hyp["network"]["byte_flag"]=='positive_weak_flatter_leaky_relu':
                imgs[(imgs <= -1)] = -1 * (imgs[(imgs <= -1)]+1)
                imgs[(imgs >= 1)] *= 0.01
                imgs[(imgs > -1) & (imgs < 1)] = 0
                imgs = imgs.to(device).half()

            elif hyp["network"]["byte_flag"]=='adjusted_symmetric_relu':
                imgs[(imgs <= -1)] *= 1 
                imgs[(imgs <= -2)] *= 1 
                imgs[(imgs >= 1)] *= 1 
                imgs[(imgs >= 2)] *= 1 
                imgs[(imgs > -1) & (imgs < 1)] = 0
                imgs = imgs.to(device).int().half()

            elif hyp["network"]["byte_flag"]=='negative_255_square':
                '''255-254, 0, 253-252'''
                imgs[(imgs <= -1)] = 255
                imgs[(imgs <= -2)] = 254 
                imgs[(imgs >= 1)] = 252
                imgs[(imgs >= 2)] = 253
                imgs = imgs.to(device).int().half()
            elif hyp["network"]["byte_flag"]=='postive_255_square':
                '''3-4, 255, 1-2'''
                imgs[(imgs > -1) & (imgs < 1)] = 255
                imgs[(imgs <= -1)] = 4
                imgs[(imgs <= -2)] = 3
                imgs = imgs.to(device).int().half()

            
            elif hyp["network"]["byte_flag"]=='negative_square':
                imgs[(imgs <= -1)] *= -1
                imgs = imgs.to(device).int().half()
            elif hyp["network"]["byte_flag"]=='postive_square':
                imgs[(imgs >= 1)] *= -1
                imgs = imgs.to(device).int().half()

            elif hyp["network"]["byte_flag"]=='sigmoid_byte':
                m = nn.Sigmoid()
                imgs = m(imgs.to(device)).byte().half()
            elif hyp["network"]["byte_flag"]=='sigmoid':
                m = nn.Sigmoid()
                imgs = m(imgs.to(device)).half()
            elif hyp["network"]["byte_flag"]=='flipped_sigmoid_byte':
                m = lambda x: 1 - nn.Sigmoid()(x)
                imgs = m(imgs.to(device)).byte().half()
            elif hyp["network"]["byte_flag"]=='flipped_sigmoid':
                m = lambda x: 1 - nn.Sigmoid()(x)
                imgs = m(imgs.to(device)).half()
            elif hyp["network"]["byte_flag"]=='tanh_byte':
                m = nn.Tanh()
                imgs = m(imgs.to(device)).byte().half()
            elif hyp["network"]["byte_flag"]=='tanh':
                m = nn.Tanh()
                imgs = m(imgs.to(device)).half()
            elif hyp["network"]["byte_flag"]=='flipped_tanh_byte':
                m = lambda x: -nn.Tanh()(x)
                imgs = m(imgs.to(device)).byte().half()
            elif hyp["network"]["byte_flag"]=='flipped_tanh':
                m = lambda x: -nn.Tanh()(x)
                imgs = m(imgs.to(device)).half()
            elif hyp["network"]["byte_flag"]=='tanh_int':
                '''-1, 000, 1'''
                m = nn.Tanh()
                imgs = m(imgs.to(device)).int().half()
            elif hyp["network"]["byte_flag"]=='flipped_tanh_int':
                m = lambda x: -nn.Tanh()(x)
                imgs = m(imgs.to(device)).int().half()

            elif hyp["network"]["byte_flag"]=='use_value_x2_byte':
                imgs = (imgs*2).to(device).byte().half()
            elif hyp["network"]["byte_flag"]=='use_value_x32_byte':
                imgs = (imgs*32).to(device).byte().half()
            elif hyp["network"]["byte_flag"]=='use_value_x128_byte':
                imgs = (imgs*128).to(device).byte().half()
            elif hyp["network"]["byte_flag"]=='watershed_segmentation_rgb':
                imgs = [watershed_segmentation_rgb(img) for img in imgs]
                # Convert the list of PIL imgs to a list of PyTorch tensors
                img_tensors = [transforms.ToTensor()(img) for img in imgs]
                # Stack the tensors to create a batch
                imgs = torch.stack(img_tensors, dim=0).to(device).half()
            elif hyp["network"]["byte_flag"]=='no_norm_x5_byte':
                # convert imgs into range 0-1
                imgs = rescale_to_0_1(imgs)
                imgs = (imgs.to(device)*5).byte().half()
            elif hyp["network"]["byte_flag"]=='no_norm_x8_byte':
                # convert imgs into range 0-1
                imgs = rescale_to_0_1(imgs)
                imgs = (imgs.to(device)*8).byte().half()
            elif hyp["network"]["byte_flag"]=='DOG':
                # convert imgs into range 0-1
                imgs = apply_gaussian_difference_batch(imgs, sigma1=1, sigma2=28, scale_to_0_1=True)
                imgs = torch.stack(imgs, dim=0).to(device).half()
            elif hyp["network"]["byte_flag"]=='byte_plus_imgs':
                imgs = torch.cat((imgs.to(device).byte().half(), imgs.to(device).clone()), dim=1)
            elif hyp["network"]["byte_flag"]=='byte_byte_plus_imgs':
                imgs = torch.cat((imgs.to(device).byte().half(), imgs.to(device).byte().half(), imgs.to(device).clone()), dim=1)
            elif hyp["network"]["byte_flag"]=='on_off_center':
                preprocessed_img_on_center = F.conv2d(imgs.to(device).clone(), on_center_kernel, padding=on_off_kernel_size // 2, groups=3)
                preprocessed_img_off_center = F.conv2d(imgs.to(device).clone(), off_center_kernel, padding=on_off_kernel_size // 2, groups=3)
                imgs = torch.cat((preprocessed_img_on_center, preprocessed_img_off_center), dim=1).to(device).half()         
            elif hyp["network"]["byte_flag"]=='no_byte':
                imgs = imgs.to(device).half() 
            else:
                raise ValueError(f"Unknown byte_flag: {hyp['network']['byte_flag']}")

            labels = labels.to(device).long()
            if transform:
                imgs = transform(imgs)
            
            if device == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            total_accuracy += compute_accuracy(outputs, labels)

    return total_loss, total_accuracy


##########################
## Blurring strategy 
##########################

class CustomBlur(transforms.GaussianBlur): # normalise is done here, so we always need to call this transform during training
    def __init__(self, epoch, blurring_strategy, blur_norm_order, hyp):
        super().__init__(kernel_size=5)  # Initialize with a fixed kernel size
        self.blurring_strategy = blurring_strategy
        self.blur_norm_order = blur_norm_order
        self.epoch = epoch
        self.hyp = hyp

    def forward(self, img):
        if self.blurring_strategy == "sharp":
            return transforms.Normalize(mean = self.hyp['dataset']['train_img_mean_channels']/255., std = self.hyp['dataset']['train_img_std_channels']/255.)(img)

        elif self.blurring_strategy in ["first_few_epochs_exponentially_decreasing", "first_few_epochs_linearly_decreasing"]:
            if self.epoch <= 20:
                if self.blurring_strategy == "first_few_epochs_exponentially_decreasing":
                    sigma = 28 * (0.01 ** ((self.epoch-1) / 19))
                else:
                    sigma = 28 - 1.4 * self.epoch + 0.001
                kernel_size = int(8*sigma) + (0 if int(8*sigma) % 2 else 1)
                if self.blur_norm_order == "blur_first":
                    img = transforms.functional.gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)
                    if self.hyp['network']['norm_flag']:
                        img = transforms.functional.normalize(img, mean = self.hyp['dataset']['train_img_mean_channels']/255., std = self.hyp['dataset']['train_img_std_channels']/255.)
                    return img
                else:
                    img = transforms.functional.normalize(img, mean = self.hyp['dataset']['train_img_mean_channels']/255., std = self.hyp['dataset']['train_img_std_channels']/255.)
                    if self.hyp['network']['norm_flag']:
                        img = transforms.functional.gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)
                    return img
            else:
                return transforms.functional.normalize(img, mean = self.hyp['dataset']['train_img_mean_channels']/255., std = self.hyp['dataset']['train_img_std_channels']/255.)
        else:
            raise ValueError("Invalid blurring mode")
        
##########################
## Loading the checkpoint and initializing or resuming  the logs
##########################
def load_checkpoint(net, net_name, net_path, log_path, hyp):
    """Load the latest checkpoint if available."""
    # Get the list of model files in the folder if available
    model_files = [f for f in os.listdir(net_path) if f.endswith(".pth")]
    if not model_files:
        #TODO fix the log & net conflict problem
        model_files = [f for f in os.listdir(log_path) if f.endswith(".pt")]
        if not model_files:
            print(f"No checkpoints found for {net_name}!")
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
def train_epoch(epoch, net, train_loader, optimizer, criterion, scaler, blur_transform,  on_center_kernel=None, off_center_kernel=None,show_progress_bar=True):
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
    device = hyp['optimizer']['device']
    net = net.float().to(device)
    train_loss_running, train_acc_running, batch = 0.0, 0.0, 0
    start = time.time()
    
    batch_id =0 
    for imgs, lbls in train_loader:
        batch_id+=1
        # Move imgs and labels to the desired device.
        imgs, lbls = imgs.to(device), lbls.to(device)
        
        # Apply transformations(norm after blurring).
        
        if not hyp['network']['uniform_blur']:
            imgs = blur_transform(imgs)
        else:
            random_uniform_epoch = random.randint(1,21) # this is random uniform blur, if epoch = epoch then it just do the blurring for the first 20 batches
            uniform_blur_transform = CustomBlur(epoch=random_uniform_epoch, blurring_strategy=hyp['network']['blurring_strategy'], blur_norm_order=hyp['network']['blur_norm_order'], hyp=hyp)
            imgs = uniform_blur_transform(imgs)
        
        # Zero the gradient buffers
        optimizer.zero_grad()
        if hyp["network"]["byte_flag"]=='use_byte':
            imgs = imgs.to(device).byte().half()
        elif hyp["network"]["byte_flag"]=='positive_byte':
            m = nn.ReLU()
            imgs = m(imgs.to(device)).byte().half()
        elif hyp["network"]["byte_flag"]=='negative_byte':
            m = nn.ReLU()
            imgs = -m(-imgs.to(device)).byte().half()
        elif hyp["network"]["byte_flag"]=='mixed_byte':
            m = nn.ReLU()
            imgs = torch.cat((m(imgs).byte(), m(-imgs).byte()), dim=1).to(device).half()
        elif hyp["network"]["byte_flag"]=='u_step':
            imgs[(imgs <= -1)] = 3
            imgs[(imgs <= -2)] = 4
            imgs[(imgs > -1) & (imgs < 1)] = 0
            imgs[(imgs >= 1)] = 1
            imgs[(imgs >= 2)] = 2
            imgs = imgs.to(device).half()

        #* Promising?
        elif hyp["network"]["byte_flag"]=='int_step':
            '''-2-1, 000, 1-2'''
            imgs = imgs.to(device).int().half()
        elif hyp["network"]["byte_flag"]=='6_channel_negative_255_step':
            '''254,255, 000, 1-2 --> equal to byte'''
            imgs_1, imgs_2 = imgs.clone(), imgs.clone()
            imgs_1[(imgs_1 <= -1) & (imgs_1 > -2)] = 255
            imgs_1[(imgs_1 <= -2)] += 254
            imgs_1[(imgs_1 > -1)] = 0
            imgs_1 = imgs_1.to(device).int().half()
            
            imgs_2[(imgs_2 < 1)] = 0
            imgs_2 = imgs_2.to(device).int().half()

            imgs = torch.cat((imgs_1, imgs_2), dim=1).to(device).half()
        elif hyp["network"]["byte_flag"]=='negative_255_step':
            '''254,255, 000, 1-2 --> equal to byte`?'''
            # imgs[(imgs <= -1)] += 257 # one difference is that -1 is 256 not 255, #! then only 0.6 get
            imgs[(imgs <= -1)] = 255
            imgs[(imgs <= -2)] = 254 
            imgs = imgs.to(device).int().half()
        elif hyp["network"]["byte_flag"]=='positive_255_step':
            '''-2-1, 000, 254, 255'''
            # imgs[(imgs >= 1)] += 253
            imgs[(imgs >= 1)] = 254
            imgs[(imgs >= 2)] = 255
            imgs = imgs.to(device).int().half()

        elif hyp["network"]["byte_flag"]=='relu':
            m = nn.ReLU()
            imgs = m(imgs).to(device).half()
        elif hyp["network"]["byte_flag"]=='strong_leaky_relu':
            imgs[(imgs <= -1)] *= -100 
            imgs[(imgs <= -2)] *= -100 
            imgs[(imgs >= 1)] *= 1 
            imgs[(imgs >= 2)] *= 1 
            imgs[(imgs > -1) & (imgs < 1)] = 0
            imgs = imgs.to(device).int().half()
        elif hyp["network"]["byte_flag"]=='strong_flipped_flatter_leaky_relu':
            imgs[(imgs <= -1)] = -1 * (imgs[(imgs <= -1)]+1)
            imgs[(imgs >= 1)] *= 100 
            imgs[(imgs > -1) & (imgs < 1)] = 0
            imgs = imgs.to(device).int().half() 


        elif hyp["network"]["byte_flag"]=='custom_steep_double_sigmoid':
            def custom_steep_double_sigmoid(x, x0=-1, k=0.05):
                x0_tensor = torch.tensor(x0, dtype=torch.float32, device=x.device)
                k_tensor = torch.tensor(k, dtype=torch.float32, device=x.device)
                y = torch.where(x < 0,
                                255 / (1 + torch.exp((x - x0_tensor) / k_tensor)) + 2,
                                -2 / (1 + torch.exp((-x - x0_tensor) / k_tensor)) + 2)
                y = torch.clamp(y, 0, 255)
                return y.to(device).int().half()
            imgs = custom_steep_double_sigmoid(x, x0=-1, k=0.05) #! Notice that we done the int()
        elif hyp["network"]["byte_flag"]=='custom_steep_no_clamp_double_sigmoid':
            def custom_steep_no_clamp_double_sigmoid(x, x0=-1, k=0.05):
                x0_tensor = torch.tensor(x0, dtype=torch.float32, device=x.device)
                k_tensor = torch.tensor(k, dtype=torch.float32, device=x.device)
                y = torch.where(x < 0,
                                255 / (1 + torch.exp((x - x0_tensor) / k_tensor)) + 2,
                                -2 / (1 + torch.exp((-x - x0_tensor) / k_tensor)) + 2)
                # y = torch.clamp(y, 0, 255)
                return y.to(device).int().half()
            imgs = custom_steep_no_clamp_double_sigmoid(x, x0=-1, k=0.05) #! Notice that we done the int()
        elif hyp["network"]["byte_flag"]=='custom_steep_positive_double_sigmoid':
            def custom_steep_positive_double_sigmoid(x, x0=-1, k=0.05):
                x0_tensor = torch.tensor(x0, dtype=torch.float32, device=x.device)
                k_tensor = torch.tensor(k, dtype=torch.float32, device=x.device)
                y = torch.where(x < 0,
                                255 / (1 + torch.exp((x - x0_tensor) / k_tensor)),
                                2 / (1 + torch.exp((-x - x0_tensor) / k_tensor)))
                return y.to(device).int().half()
            imgs = custom_steep_positive_double_sigmoid(x, x0=-1, k=0.05) #! Notice that we done the int()
        elif hyp["network"]["byte_flag"]=='custom_shifted_minus_1_steep_sigmoid':
            def custom_shifted_minus_1_steep_sigmoid(x, x0=0, k=0.05, left_shift=-1, down_shift=-2):
                x0_adjusted = x0 + left_shift
                x0_tensor = torch.tensor(x0_adjusted, dtype=torch.float32, device=x.device)
                k_tensor = torch.tensor(k, dtype=torch.float32, device=x.device)
                y = 255 / (1 + torch.exp((x - x0_tensor) / k_tensor)) - down_shift
                y = torch.clamp(y, 0, 255)
                return y.to(device).int().half()
            imgs = custom_shifted_minus_1_steep_sigmoid(x, x0=0, k=0.05, left_shift=-1, down_shift=-2) #! Notice that we done the int()
        elif hyp["network"]["byte_flag"]=='custom_shifted_minus_1_not_steep_sigmoid':
            def custom_shifted_minus_1_not_steep_sigmoid(x, x0=0, k=0.2, left_shift=-1, down_shift=-2):
                x0_adjusted = x0 + left_shift
                x0_tensor = torch.tensor(x0_adjusted, dtype=torch.float32, device=x.device)
                k_tensor = torch.tensor(k, dtype=torch.float32, device=x.device)
                y = 255 / (1 + torch.exp((x - x0_tensor) / k_tensor)) - down_shift
                y = torch.clamp(y, 0, 255)
                return y.to(device).int().half()
            imgs = custom_shifted_minus_1_not_steep_sigmoid(x, x0=0, k=0.2, left_shift=-1, down_shift=-2) #! Notice that we done the int()
        elif hyp["network"]["byte_flag"]=='custom_shifted_steep_sigmoid':
            def custom_shifted_steep_sigmoid(x, x0=0, k=0.05, left_shift=-1, down_shift=-2):
                x0_adjusted = x0 + left_shift
                x0_tensor = torch.tensor(x0_adjusted, dtype=torch.float32, device=x.device)
                k_tensor = torch.tensor(k, dtype=torch.float32, device=x.device)
                y = 255 / (1 + torch.exp((x - x0_tensor) / k_tensor)) - down_shift
                y = torch.clamp(y, 0, 255)
                return y.to(device).int().half()
            imgs = custom_shifted_steep_sigmoid(x, x0=0, k=0.05, left_shift=-1, down_shift=-2) #! Notice that we done the int()
        elif hyp["network"]["byte_flag"]=='custom_shifted_not_steep_sigmoid':
            def custom_shifted_not_steep_sigmoid(x, x0=0, k=0.2, left_shift=0, down_shift=-2):
                x0_adjusted = x0 + left_shift
                x0_tensor = torch.tensor(x0_adjusted, dtype=torch.float32, device=x.device)
                k_tensor = torch.tensor(k, dtype=torch.float32, device=x.device)
                y = 255 / (1 + torch.exp((x - x0_tensor) / k_tensor)) - down_shift
                y = torch.clamp(y, 0, 255)
                return y.to(device).int().half()
            imgs = custom_shifted_not_steep_sigmoid(x, x0=0, k=0.2, left_shift=0, down_shift=-2)


        elif hyp["network"]["byte_flag"]=='strong_flatter_leaky_relu':
            imgs[(imgs <= -1)] = -100 * (imgs[(imgs <= -1)]+1)
            imgs[(imgs >= 1)] *= 1
            imgs[(imgs > -1) & (imgs < 1)] = 0
            imgs = imgs.to(device).int().half()
        elif hyp["network"]["byte_flag"]=='strong_flipped_leaky_relu':
            imgs[(imgs <= -1)] *= -1 
            imgs[(imgs >= 1)] *= 100
            imgs[(imgs > -1) & (imgs < 1)] = 0
            imgs = imgs.to(device).int().half()
        elif hyp["network"]["byte_flag"]=='strong_rising_leaky_relu':
            imgs[(imgs <= -1)] = 100 * (imgs[(imgs <= -1)]+1)
            imgs[(imgs >= 1)] *= 1
            imgs[(imgs > -1) & (imgs < 1)] = 0
            imgs = imgs.to(device).int().half()
        elif hyp["network"]["byte_flag"]=='strong_decreasing_leaky_relu':
            imgs[(imgs <= -1)] = -100 * (imgs[(imgs <= -1)]+1)
            imgs[(imgs >= 1)] *= -1
            imgs[(imgs > -1) & (imgs < 1)] = 0
            imgs = imgs.to(device).int().half()   #! Notice that we done the int()
        # 'strong_no_int_leaky_relu', 'negative_weak_flatter_leaky_relu', 'positive_weak_flatter_leaky_relu'
        elif hyp["network"]["byte_flag"]=='strong_no_int_leaky_relu':
            imgs[(imgs <= -1)] *= -100 
            imgs[(imgs <= -2)] *= -100 
            imgs[(imgs >= 1)] *= 1 
            imgs[(imgs >= 2)] *= 1 
            imgs[(imgs > -1) & (imgs < 1)] = 0
            imgs = imgs.to(device).half() #! Notice that we done the int()
        elif hyp["network"]["byte_flag"]=='negative_weak_flatter_leaky_relu':
            imgs[(imgs <= -1)] = -0.01 * (imgs[(imgs <= -1)]+1)
            imgs[(imgs >= 1)] *= 1
            imgs[(imgs > -1) & (imgs < 1)] = 0
            imgs = imgs.to(device).half() #* no int()
        elif hyp["network"]["byte_flag"]=='positive_weak_flatter_leaky_relu':
            imgs[(imgs <= -1)] = -1 * (imgs[(imgs <= -1)]+1)
            imgs[(imgs >= 1)] *= 0.01
            imgs[(imgs > -1) & (imgs < 1)] = 0
            imgs = imgs.to(device).half()

        elif hyp["network"]["byte_flag"]=='adjusted_symmetric_relu':
            imgs[(imgs <= -1)] *= 1 
            imgs[(imgs <= -2)] *= 1 
            imgs[(imgs >= 1)] *= 1 
            imgs[(imgs >= 2)] *= 1 
            imgs[(imgs > -1) & (imgs < 1)] = 0
            imgs = imgs.to(device).int().half()

        elif hyp["network"]["byte_flag"]=='negative_255_square':
            imgs[(imgs <= -1)] = 255
            imgs[(imgs <= -2)] = 254 
            imgs[(imgs >= 1)] = 253
            imgs[(imgs >= 2)] = 252
            imgs = imgs.to(device).int().half()
        elif hyp["network"]["byte_flag"]=='postive_255_square':
            imgs[(imgs > -1) & (imgs < 1)] = 255
            imgs = imgs.to(device).int().half()

        elif hyp["network"]["byte_flag"]=='negative_square':
            imgs[(imgs <= -1)] *= -1
            imgs = imgs.to(device).int().half()
        elif hyp["network"]["byte_flag"]=='postive_square':
            imgs[(imgs >= 1)] *= -1
            imgs = imgs.to(device).int().half()

        elif hyp["network"]["byte_flag"]=='sigmoid_byte':
            m = nn.Sigmoid()
            imgs = m(imgs.to(device)).byte().half()
        elif hyp["network"]["byte_flag"]=='sigmoid':
            m = nn.Sigmoid()
            imgs = m(imgs.to(device)).half()
        elif hyp["network"]["byte_flag"]=='flipped_sigmoid_byte':
            m = lambda x: 1 - nn.Sigmoid()(x)
            imgs = m(imgs.to(device)).byte().half()
        elif hyp["network"]["byte_flag"]=='flipped_sigmoid':
            m = lambda x: 1 - nn.Sigmoid()(x)
            imgs = m(imgs.to(device)).half()
        elif hyp["network"]["byte_flag"]=='tanh_byte':
            m = nn.Tanh()
            imgs = m(imgs.to(device)).byte().half()
        elif hyp["network"]["byte_flag"]=='tanh':
            m = nn.Tanh()
            imgs = m(imgs.to(device)).half()
        elif hyp["network"]["byte_flag"]=='flipped_tanh_byte':
            m = lambda x: -nn.Tanh()(x)
            imgs = m(imgs.to(device)).byte().half()
        elif hyp["network"]["byte_flag"]=='flipped_tanh':
            m = lambda x: -nn.Tanh()(x)
            imgs = m(imgs.to(device)).half()
        elif hyp["network"]["byte_flag"]=='tanh_int':
            '''-1, 000, 1'''
            m = nn.Tanh()
            imgs = m(imgs.to(device)).int().half()
        elif hyp["network"]["byte_flag"]=='flipped_tanh_int':
            m = lambda x: -nn.Tanh()(x)
            imgs = m(imgs.to(device)).int().half()
        elif hyp["network"]["byte_flag"]=='use_value_x2_byte':
            imgs = (imgs.to(device)*2).byte().half()
        elif hyp["network"]["byte_flag"]=='use_value_x32_byte':
            imgs = (imgs.to(device)*32).byte().half()
        elif hyp["network"]["byte_flag"]=='use_value_x128_byte':
            imgs = (imgs.to(device)*128).byte().half()
        elif hyp["network"]["byte_flag"]=='watershed_segmentation_rgb':
            imgs = [watershed_segmentation_rgb(img) for img in imgs]
            # Convert the list of PIL imgs to a list of PyTorch tensors
            img_tensors = [transforms.ToTensor()(img) for img in imgs]
            # Stack the tensors to create a batch
            imgs = torch.stack(img_tensors, dim=0).to(device).half()
        elif hyp["network"]["byte_flag"]=='no_norm_x5_byte':
            # convert imgs into range 0-1
            imgs = rescale_to_0_1(imgs)
            imgs = (imgs.to(device)*5).byte().half()
        elif hyp["network"]["byte_flag"]=='no_norm_x8_byte':
            # convert imgs into range 0-1
            imgs = rescale_to_0_1(imgs)
            imgs = (imgs.to(device)*8).byte().half()
        elif hyp["network"]["byte_flag"]=='DOG':
            imgs = apply_gaussian_difference_batch(imgs, sigma1=1, sigma2=28, scale_to_0_1=True) # 0-1
            imgs = torch.stack(imgs, dim=0).to(device).half()
        elif hyp["network"]["byte_flag"]=='byte_plus_imgs':
            byte_imgs = imgs.to(device).clone().byte().half()
            normal_imgs = imgs.to(device).clone().half()
            imgs = torch.cat((byte_imgs, normal_imgs), dim=1).to(device).half()
        elif hyp["network"]["byte_flag"]=='byte_byte_plus_imgs':
            imgs = torch.cat((imgs.to(device).clone().byte().half(), imgs.to(device).clone().byte().half(), imgs.to(device).clone().half()), dim=1).to(device).half()
        elif hyp["network"]["byte_flag"]=='on_off_center':
            preprocessed_img_on_center = F.conv2d(imgs.to(device).clone(), on_center_kernel, padding=on_off_kernel_size // 2, groups=3)
            preprocessed_img_off_center = F.conv2d(imgs.to(device).clone(), off_center_kernel, padding=on_off_kernel_size // 2, groups=3)
            imgs = torch.cat((preprocessed_img_on_center, preprocessed_img_off_center), dim=1).to(device).half()
            # print(f"imgs: {imgs.shape}\n type of imgs: {imgs.dtype}\n ")
            # Apply the on-center convolution kernel
            # print(f"preprocessed_img_on_center: {preprocessed_img_on_center.shape}\n type of preprocessed_img_on_center: {preprocessed_img_on_center.dtype}\n ")
            # Apply the off-center convolution kernel
            # print(f"off_center_kernel: {off_center_kernel.shape}\n type of off_center_kernel: {off_center_kernel.dtype}\n")
            # if epoch <= 1 and batch_id <= 5:
            #     save_dir = "train_imgs_examples"
            #     os.makedirs(save_dir, exist_ok=True)

            #     for i in range(imgs.shape[0]):
            #         on_center_img = preprocessed_img_on_center[i].cpu().numpy()
            #         off_center_img = preprocessed_img_off_center[i].cpu().numpy()

            #         # Convert to 8-bit unsigned integer (uint8) format
            #         on_center_img = (on_center_img * 255).astype(np.uint8)
            #         off_center_img = (off_center_img * 255).astype(np.uint8)

            #         # Create PIL imgs from NumPy arrays
            #         on_center_pil_img = Image.fromarray(on_center_img.transpose(1, 2, 0))
            #         off_center_pil_img = Image.fromarray(off_center_img.transpose(1, 2, 0))

            #         # Save the imgs
            #         on_center_pil_img.save(os.path.join(save_dir, f"epoch{epoch}_batch{batch_id}_on_center_{i}.png"))
            #         off_center_pil_img.save(os.path.join(save_dir, f"epoch{epoch}_batch{batch_id}_off_center_{i}.png"))
            
            # Combine on-center and off-center imgs to create a 6-channel RGBRGB image
            # Rescale to [0,1] and then to [0,255] #! no need, just no normalization
            # preprocessed_img_on_center = rescale_to_0_255(rescale_to_0_1(preprocessed_img_on_center))
            # preprocessed_img_off_center = rescale_to_0_255(rescale_to_0_1(preprocessed_img_off_center))
            # print(f"shape of imgs: {imgs.shape}\n type of imgs: {imgs.dtype}\n \
            #       on_center_kernel: {on_center_kernel.shape}\n type of on_center_kernel: {on_center_kernel.dtype}\n\
            #       off_center_kernel: {off_center_kernel.shape}\n type of off_center_kernel: {off_center_kernel.dtype}\n\
            #       preprocessed_img_on_center: {preprocessed_img_on_center.shape}\n type of preprocessed_img_on_center: {preprocessed_img_on_center.dtype}\n\
            #       preprocessed_img_off_center: {preprocessed_img_off_center.shape}\n type of preprocessed_img_off_center: {preprocessed_img_off_center.dtype}\n")
        else:
            print(f"not using byte")
            imgs = imgs.to(device).half()

        # Compute the forward pass and loss.
        if hyp['optimizer']['device'] == 'cuda':
             with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=hyp['misc']['use_amp']):
                outputs = net(imgs)
                loss = criterion(outputs, lbls.long())
        else:
            raise ValueError("Invalid device")

        
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
    # wandb.init(project="Blurring_project", config=hyp)

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
    net = net.float().to(hyp['optimizer']['device'])
    
    # Assuming you have already created on-center and off-center kernels using the create_kernels function
    if hyp["network"]["byte_flag"]=='on_off_center':
        on_off_kernel_size = hyp["network"]["on_off_kernel_size"]
        on_center_kernel, off_center_kernel = create_kernels(size=on_off_kernel_size, enhance_factor=hyp['network']['enhance_factor'])
    else:
        on_center_kernel, off_center_kernel = None, None

    # Train the network
    for epoch in range(start_epoch, hyp['optimizer']['n_epochs']+1):

        blur_transform = CustomBlur(epoch=epoch, blurring_strategy=hyp['network']['blurring_strategy'], blur_norm_order=hyp['network']['blur_norm_order'], hyp=hyp)
        train_loss, train_acc = train_epoch(epoch, net, train_loader, optimizer, criterion, scaler, \
                                    blur_transform, on_center_kernel, off_center_kernel)
        val_loss, val_acc = validate_epoch(net, val_loader, criterion,hyp)
        
        # Log and save metrics
        log_and_save_metrics(epoch, train_loss, train_acc, val_loss, val_acc, net, optimizer, log_path, net_name, hyp)


    print('\nDone training and evaluating in testset now!\n')
    if os.path.exists(f'{net_path}/{net_name}_epoch_{epoch}.pth'):
        torch.save(net.state_dict(), f'{net_path}/{net_name}_epoch_{epoch}.pth')
    evaluate_on_test_set(net, test_loader, criterion, hyp)














