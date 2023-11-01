import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import ImageFilter, ImageChops
import cv2

# Calculate the number of columns based on the maximum number of epochs (e.g., 25)
num_epochs = 25
# Calculate the number of rows based on the number of kernel sizes (sizes) and additional rows (2) for transformed and normalized images
sizes = [] #list(range(3,8,2))
num_rows = len(sizes) * 2 + 6
enhance_factor =1
norm_flag = True #TODO "no_normalize" not in hyp["retina"]["preprocess"]:

mean = np.array([0.49081137, 0.47463922, 0.44580941])
std = np.array([0.28261176, 0.27914157, 0.28713294])

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

    # # Convert the segmented channel to a PIL image
    # pil_image = Image.fromarray(segmented_channel)

    # # Save the segmented channel if save_path is provided
    # if save_path:
    #     pil_image.save(save_path)

    return segmented_channel


def watershed_segmentation_mean_std_rgb(img, mean=None, std=None, save_path=None):
    # Ensure that mean and std are provided
    if mean is None or std is None:
        raise ValueError("Mean and std must be provided.")

    # Convert the input image to a NumPy array
    image = np.array(img.cpu().numpy())

    # Check if the image is grayscale (1 channel)
    if image.shape[-1] not in [3, 1]:
        image = image.transpose((1, 2, 0))
    if image.shape[-1] == 1:
        return watershed_segmentation_single_channel(image, save_path)

    # Separate the image into RGB channels
    b, g, r = cv2.split(image)

    # Calculate the standard deviation of each pixel across channels
    pixel_std = np.std((image - mean) / std, axis=2)

    # Define thresholds to separate pixels into different std groups
    thresholds = [1.0, 2.0, 3.0, 4.0, 5.0]  # You can adjust these thresholds as needed

    # Create multiple binary masks based on std ranges
    masks = []
    for threshold in thresholds:
        within_threshold = pixel_std <= threshold
        masks.append(within_threshold)

    # print(f"shape of masks: {np.array(masks).shape}")
    # Apply watershed segmentation to each channel based on the masks
    segmented_b = watershed_segmentation_mean_std_single_channel(b, mask=masks[0])
    segmented_g = watershed_segmentation_mean_std_single_channel(g, mask=masks[1])
    segmented_r = watershed_segmentation_mean_std_single_channel(r, mask=masks[2])

    # Merge the segmented channels back into an RGB image
    segmented_image = cv2.merge((segmented_b, segmented_g, segmented_r))

    # Convert the segmented image to a PIL image
    pil_image = Image.fromarray(segmented_image)

    # Save the segmented image if save_path is provided
    if save_path:
        pil_image.save(save_path)

    return pil_image

def watershed_segmentation_mean_std_single_channel(channel, mask=None, save_path=None):
    # Convert the channel to grayscale
    # Check if the channel has 2 dimensions, and if so, add a third dimension
    if len(channel.shape) == 2:
        channel = np.expand_dims(channel, axis=-1)

    # Check if the channel values are in the range [0, 1]
    if np.min(channel) >= 0 and np.max(channel) <= 1:
        # Rescale the channel from [0, 1] to [0, 255]
        channel = (channel * 255).astype(np.uint8)

    gray = channel.astype(np.uint8)  # Convert to 8-bit unsigned integer

    # Apply thresholding using the provided mask if available
    #TODO here only binary, if it's 
    if mask is not None:
        # Convert the mask data type to np.uint8
        print(f"mask: {mask.shape}, mask: {mask}")
        mask = mask.astype(np.uint8)
        gray = cv2.bitwise_and(gray, gray, mask=mask)

    # Apply thresholding
    #TODO the threshold is wrong, still the default threshold, not masks
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remove noise
    kernel = np.ones((3, 3), np.uint8)
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    opening = thresh  # with noise

    # Determine background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Find the unknown area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Directly segment the image
    sure_fg = np.uint8(sure_fg)
    segmented_channel = cv2.subtract(sure_bg, sure_fg)

    # # Convert the segmented channel to a PIL image
    # pil_image = Image.fromarray(segmented_channel)

    # # Save the segmented channel if save_path is provided
    # if save_path:
    #     pil_image.save(save_path)

    return segmented_channel

# Assuming the classes CustomBlur1 and CustomBlur2 are already defined
class CustomBlur(transforms.GaussianBlur): # normalise is done here, so we always need to call this transform during training
    def __init__(self, epoch, blurring_strategy, blur_norm_order, hyp):
        super().__init__(kernel_size=5)  # Initialize with a fixed kernel size
        self.blurring_strategy = blurring_strategy
        self.blur_norm_order = blur_norm_order
        self.epoch = epoch
        self.hyp = hyp
        self.norm_flag = False #TODO change this to True if you want to normalize the blurred image

    def forward(self, img):
        # print(f"self.epoch: {self.epoch} self.blurring_strategy: {self.blurring_strategy}")
        if self.blurring_strategy == "sharp":
            return img #transforms.Normalize(mean = self.hyp['dataset']['train_img_mean_channels']/255., std = self.hyp['dataset']['train_img_std_channels']/255.)(img)

        
        elif self.blurring_strategy in ["first_few_epochs_exponentially_decreasing", "fisrt_few_epochs_linearly_decreasing"] and self.epoch <= 20:
            if self.blurring_strategy == "first_few_epochs_exponentially_decreasing":
                sigma = 28 * (0.01 ** ((self.epoch-1) / 19))
            else:
                sigma = 28 - 1.4 * self.epoch + 0.001
            kernel_size = int(8*sigma) + (0 if int(8*sigma) % 2 else 1)
            print(f"CustomBlur kernel_size: {kernel_size}, sigma: {sigma}")
            if self.blur_norm_order == "blur_first":
                img = transforms.functional.gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)
                if self.norm_flag:
                    img = transforms.functional.normalize(img, mean = self.hyp['dataset']['train_img_mean_channels']/255., std = self.hyp['dataset']['train_img_std_channels']/255.)
                return img
            else:
                if self.norm_flag:
                    img = transforms.functional.normalize(img, mean = self.hyp['dataset']['train_img_mean_channels']/255., std = self.hyp['dataset']['train_img_std_channels']/255.)
                img = transforms.functional.gaussian_blur(img, kernel_size=kernel_size, sigma=sigma)
                return img
        else:
            if self.norm_flag:
                return transforms.functional.normalize(img, mean = self.hyp['dataset']['train_img_mean_channels']/255., std = self.hyp['dataset']['train_img_std_channels']/255.)
            else:
                return img

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


# Denormalization function
def denormalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# Rescale function to [0, 1]
def rescale_to_01(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)

# Function to create on-center and off-center kernels for a given size
def create_kernels(size, enhance_factor=1, center_point =False):
    # Central value
    central = 1/((size-2)*(size-2))*enhance_factor # Adjust as needed --> enhance the center | if blurring 1/(size*size) #
    
    # Surrounding value
    surrounding = -1/(size*size-(size-2)*(size-2))*enhance_factor   # Adjust as needed --> suppress the surrounding

    # Create on-center kernel
    on_center_kernel = torch.full((3, 1, size, size), surrounding, dtype=torch.float32).to(device)
    if not center_point:
        on_center_kernel[:, :, 1:size-1, 1:size-1] = central 
    else:
        on_center_kernel[:, :, size // 2, size // 2] = central # one one center point

    # Create off-center kernel (reverse of on-center)
    off_center_kernel = -on_center_kernel.clone()
    if not center_point:
        off_center_kernel[:, :, 1:size-1, 1:size-1] = -central
    else:
        off_center_kernel[:, :, size // 2, size // 2] = -central

    return on_center_kernel, off_center_kernel


# Sample hyperparameters
hyp = {
    'dataset': {
        'train_img_mean_channels': np.array([0.485, 0.456, 0.406]),
        'train_img_std_channels': np.array([0.229, 0.224, 0.225]),
            },
    'retina':
        {
            "preprocess": ["byte","on-center", "off-center",
                            "on-center_5x5", "off-center_5x5",
                            "on_minus_off-center_5x5","on_add_off-center_5x5",
                            "compare_different_size_on_off_center_kernels",
                            "compare_different_size_on_off_1_centered_kernels",
                            "compare_no_normalize_different_size_on_off_1_centered_kernels",
                             "compare_conv2d_with_norm",
                             "compare_conv2d_x2_enhanced_without_norm",
                             "blur_snaity_check_RGB_imgs_conv2d",
                             "compare_conv2d_x8_enhanced_without_norm",
                             "compare_original_images_without_norm",
                             "compare_orginal_norm_byte_vs_on_off_center_size_357",
                             "compare_orginal_norm_byte_and_double_byte_vs_on_off_center_and_DOG_canny_size_357",
                             "compare_orginal_norm_clamp01_vs_on_off_center_and_DOG_canny_size_357",
                             "compare_orginal_norm_clamp_min0_vs_on_off_center_and_DOG_canny_size_357",
                             "compare_orginal_norm_clamp_max0_vs_on_off_center_and_DOG_canny_size_357",
                             "compare_orginal_norm_clamp01_vs_on_off_center_and_DOG_canny_size_357",
                             "compate_with_watershed",
                             "compate_with_watershed_with_noise",
                             "compate_with_watershed_with_mean_std_noise",
                             "compate_with_watershed_and_DOG_28_10",
                             ][-1],
        }
    
}


# Assuming you have a device defined like this
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imgs_dir = "/home/student/l/lzejin/codebase/blurring4texture2shape1file/cue_conflict_images/airplane" #"imgs"
output_dir = "/home/student/l/lzejin/codebase/blurring4texture2shape1file/save_dir"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Load an image and convert it to a tensor
from tqdm import tqdm
for img_name in tqdm(os.listdir(imgs_dir)[:5]):
    img_path = os.path.join(imgs_dir, img_name)
    img = Image.open(img_path).convert("RGB")
    img_tensor = transforms.functional.to_tensor(img).to(device)

    # Create subplots for visualization
    fig, axs = plt.subplots(num_rows, num_epochs, figsize=(5 * num_epochs, 10*(len(sizes)+6)))

    # Define condition labels
    condition_labels = ["Transformed", "Normalized", 'byte']
    for size in sizes:
        condition_labels.extend([f'{size}x{size} On-Center', f'{size}x{size} Off-Center'])

    # Set labels for rows
    for i, label in enumerate(condition_labels):
        axs[i, 0].set_ylabel(label, size='x-large')


    for epoch in range(1, num_epochs+1):
        blur = CustomBlur(epoch, "first_few_epochs_exponentially_decreasing", "blur_first", hyp)

        # Apply transformations
        transform = get_transform(['randomrotation', 'randomflip', 'grayscale'], hyp)
        transformed_img = transform(img_tensor.clone())
        blurred_img = blur(transformed_img.clone())

        # Transformed image without normalization
        transformed_img_pil = transforms.functional.to_pil_image(blurred_img.cpu())
        axs[0, epoch-1].imshow(transformed_img_pil)
        axs[0, epoch-1].set_title(f'Epoch {epoch}', size='xx-large')
        axs[0, epoch-1].axis('off')

        # Normalized image
        if norm_flag:
            normalized_img = transforms.functional.normalize(blurred_img.clone(), mean=hyp['dataset']['train_img_mean_channels'], std=hyp['dataset']['train_img_std_channels'])
            rescaled_img = rescale_to_01(normalized_img.clone())
            denormalized_img_pil = transforms.functional.to_pil_image(rescaled_img.cpu())
            axs[1, epoch-1].imshow(denormalized_img_pil)
            axs[1, 0].set_title(f'Normalized', size='xx-large')
            axs[1, epoch-1].axis('off')
        else:
            rescaled_img = rescale_to_01(blurred_img.clone())
            denormalized_img_pil = transforms.functional.to_pil_image(rescaled_img.cpu())
            axs[1, epoch-1].imshow(denormalized_img_pil)
            axs[1, 0].set_title(f'Normalized', size='xx-large')
            axs[1, epoch-1].axis('off')

        # byte
        # normalized_img = transforms.functional.normalize(blurred_img.clone(), mean=hyp['dataset']['train_img_mean_channels'], std=hyp['dataset']['train_img_std_channels'])
        if norm_flag:
            print(f"range of normalized_img: {normalized_img.min()}, {normalized_img.max()}")
            byte_img = normalized_img.clone().byte()
        else:
            byte_img = rescaled_img.clone().byte()
        print(f"range of byte_img: {byte_img.min()}, {byte_img.max()}")
        preprocessed_byte_img_pil = transforms.functional.to_pil_image(byte_img.cpu())
        print(f"range of preprocessed_byte_img_pil: {preprocessed_byte_img_pil.getextrema()}")
        axs[2, epoch-1].imshow(preprocessed_byte_img_pil)
        axs[2, 0].set_title(f'byte', size='xx-large')
        axs[2, epoch-1].axis('off')

        if norm_flag:
            # Ensure the tensor values are in the range [0, 1]
            # double_byte_clamped_img = torch.clamp(normalized_img,  0, 1)#max=0)#torch.clamp(normalized_img,
            # Scale and translate to [0, 255]
            # double_byte_img = (127.5 * (double_byte_clamped_img + 1)).byte() # if -1 to 1
            # Step 2: Scale the tensor values to [0, 255]
            diffuse_factor = 32.
            scaled_tensor = normalized_img*diffuse_factor #double_byte_clamped_img * 255

            # Step 3: Convert the tensor to torch.uint8
            double_byte_img = scaled_tensor.byte()
        else:
            # Assuming rescaled_img is already in [0, 1]
            double_byte_img = (255 * rescaled_img).byte()

        preprocessed_double_byte_img_pil = transforms.functional.to_pil_image(double_byte_img.cpu())
        axs[3, epoch-1].imshow(preprocessed_double_byte_img_pil)
        axs[3, 0].set_title(f'double_edge_byte', size='xx-large')
        axs[3, epoch-1].axis('off')
        
        imgs = normalized_img.clone()
        imgs[(imgs < 2) & (imgs > 1)] = 254
        imgs[(imgs >= 2)] += 255
        filpped_byte_imgs = imgs.to(device).int().half()
        axs[4, epoch-1].imshow(watershed_Pil_img)
        axs[4, 0].set_title('Watershed', size='xx-large')
        axs[4, epoch-1].axis('off')

        # watershed_Pil_mean_std_img = watershed_segmentation_mean_std_rgb(blurred_img, mean, std)
        # axs[5, epoch-1].imshow(watershed_Pil_mean_std_img)
        # axs[5, 0].set_title('Watershed with mean and std', size='xx-large')
        # axs[5, epoch-1].axis('off')
                
        #DOG using PIL
        # sigma1 = 1
        # sigma2 = 28
        # preprocessed_DOG_img_pil = transforms.functional.to_pil_image(blurred_img.clone().cpu())
        # gaussian1 = preprocessed_DOG_img_pil.filter(ImageFilter.GaussianBlur(radius=sigma1))
        # gaussian2 = preprocessed_DOG_img_pil.filter(ImageFilter.GaussianBlur(radius=sigma2))
        # dog_pil = ImageChops.difference(gaussian1, gaussian2)

        # strong_leaky_relu 
        imgs = normalized_img.clone()
        imgs[(imgs <= -1)] *= -100 
        imgs[(imgs >= 1)] *= 1 
        imgs[(imgs > -1) & (imgs < 1)] = 0
        imgs_strong_leaky_relu = imgs.int().half()

        axs[5, epoch-1].imshow(imgs_strong_leaky_relu)
        axs[5, 0].set_title('Strong_Leaky_ReLU', size='xx-large')
        axs[5, epoch-1].axis('off')

        # # Edge detection using PIL's FIND_EDGES
        # preprocessed_edge_img_pil = transforms.functional.to_pil_image(blurred_img.clone().cpu())
        # edges_pil = preprocessed_edge_img_pil.filter(ImageFilter.FIND_EDGES)
        # axs[5, epoch-1].imshow(edges_pil, cmap='gray')
        # axs[5, 0].set_title('Edges', size='xx-large')
        # axs[5, epoch-1].axis('off')     

        # previous_rows_len = 6
        # # Plot on-center and off-center images for different kernel sizes
        # for i, size in enumerate(sizes):
        #     on_center_kernel, off_center_kernel = create_kernels(size)
            
        #     # Apply the on-center convolution kernel
        #     if not norm_flag:
        #         preprocessed_img_on_center = F.conv2d(rescaled_img.unsqueeze(0), on_center_kernel, padding=size // 2, groups=3)
        #     else:
        #         preprocessed_img_on_center = F.conv2d(normalized_img.unsqueeze(0), on_center_kernel, padding=size // 2, groups=3)
        #     preprocessed_img_on_center = preprocessed_img_on_center.squeeze(0)

        #     preprocessed_img_pil_on_center = transforms.functional.to_pil_image(preprocessed_img_on_center.cpu())
        #     axs[previous_rows_len+ i * 2, epoch-1].imshow(preprocessed_img_pil_on_center)
        #     axs[previous_rows_len+ i * 2, 0].set_title(f'{size}x{size} On-Center', size='xx-large')
        #     axs[previous_rows_len+ i * 2, epoch-1].axis('off')


        #     # Apply the off-center convolution kernel
        #     if not norm_flag:
        #         preprocessed_img_off_center = F.conv2d(rescaled_img.unsqueeze(0), off_center_kernel, padding=size // 2, groups=3)
        #     else:
        #         preprocessed_img_off_center = F.conv2d(normalized_img.unsqueeze(0), off_center_kernel, padding=size // 2, groups=3)
        #     preprocessed_img_off_center = preprocessed_img_off_center.squeeze(0)

        #     preprocessed_img_pil_off_center = transforms.functional.to_pil_image(preprocessed_img_off_center.cpu())
        #     axs[previous_rows_len+i * 2 + 1, epoch-1].imshow(preprocessed_img_pil_off_center)
        #     axs[previous_rows_len+ i * 2+1, 0].set_title(f'{size}x{size} Off-Center', size='xx-large')
        #     axs[previous_rows_len+ i * 2 + 1, epoch-1].axis('off')

    
    

    # Adjust layout
    plt.tight_layout()  # pad =3 Increase padding for better spacing

    # Save the figure
    plt.savefig(os.path.join(output_dir, f"{hyp['retina']['preprocess']}_norm_flag_{norm_flag}_enhance_factor_{enhance_factor}_{img_name}_comparison.png"))

    # Show the figure
    plt.show()

print("All images processed and saved!")

# print(f"kernels: {kernels}")
# print(f"kernels[3]: {kernels[3]}")
# print(f"kernels[5]: {kernels[5]}")
# print(f"kernels[7]: {kernels[7]}")

# Now you have a dictionary 'kernels' containing on-center and off-center kernels for different sizes
# Access them like kernels[7]["on_center"], kernels[9]["off_center"], etc.
