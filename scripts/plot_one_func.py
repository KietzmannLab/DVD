import numpy as np
import matplotlib.pyplot as plt

def custom_steep_double_sigmoid(x, x0=-1, k=0.05):

    # Apply the condition to each element of the array
    y = np.where(x < 0, 
                 255 / (1 + np.exp((x - x0) / k)), 
                 -2 / (1 + np.exp((-x - x0) / k)))
    return y

def custom_steep_positive_double_sigmoid(x, x0=-1, k=0.05):

    # Apply the condition to each element of the array
    y = np.where(x < 0, 
                 255 / (1 + np.exp((x - x0) / k)), 
                 2 / (1 + np.exp((-x - x0) / k)))
    return y

def custom_shifted_minus_1_steep_sigmoid(x, x0=0, k=0.05, left_shift=-1, down_shift=-2):
    # Adjust x0 by adding left_shift to move the sigmoid curve to the left by 1
    x0_adjusted = x0 + left_shift
    # Compute the sigmoid function and shift the result down by 3
    y = 255 / (1 + np.exp((x - x0_adjusted) / k)) - down_shift 
    # Ensure that y is within the valid range of image pixel values after the shift
    y = np.clip(y, 0, 255)
    return y

def custom_shifted_minus_1_not_steep_sigmoid(x, x0=0, k=0.2, left_shift=-1, down_shift=-2):
    # Adjust x0 by adding left_shift to move the sigmoid curve to the left by 1
    x0_adjusted = x0 + left_shift
    # Compute the sigmoid function and shift the result down by 3
    y = 255 / (1 + np.exp((x - x0_adjusted) / k)) - down_shift 
    # Ensure that y is within the valid range of image pixel values after the shift
    y = np.clip(y, 0, 255)
    return y

def custom_shifted_steep_sigmoid(x, x0=0, k=0.05, left_shift=0, down_shift=-2):
    # Adjust x0 by adding left_shift to move the sigmoid curve to the left by 1
    x0_adjusted = x0 + left_shift
    # Compute the sigmoid function and shift the result down by 3
    y = 255 / (1 + np.exp((x - x0_adjusted) / k)) - down_shift 
    # Ensure that y is within the valid range of image pixel values after the shift
    y = np.clip(y, 0, 255)
    return y

def custom_shifted_not_steep_sigmoid(x, x0=0, k=0.2, left_shift=0, down_shift=-2):
    # Adjust x0 by adding left_shift to move the sigmoid curve to the left by 1
    x0_adjusted = x0 + left_shift
    # Compute the sigmoid function and shift the result down by 3
    y = 255 / (1 + np.exp((x - x0_adjusted) / k)) - down_shift 
    # Ensure that y is within the valid range of image pixel values after the shift
    y = np.clip(y, 0, 255)
    return y

# Generate a range of values from -10 to 10, which will include our sigmoid changes
x_values = np.linspace(-2.55, 2.55, 1000)
y_values = custom_steep_positive_double_sigmoid(x_values)

plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, label=f"Custom Sigmoid, x0={-1/1}, k={0.05}")
plt.title("Custom Steep Sigmoid Function")
plt.xlabel("x")
plt.ylabel("Sigmoid Output")
plt.legend()
plt.grid(True)

# Save the plot as a file
plt.savefig('save_dir/actiation_funcs_plots/custom_steep_positive_double_sigmoid.png', format='png')

# Show the plot
plt.show()


# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# import torch

# # Define the custom steep sigmoid function
# def custom_steep_sigmoid(x, x0=-1, k=0.05):
#     y = np.where(x < 0, 
#                  255 / (1 + np.exp((x - x0) / k)), 
#                  -2 / (1 + np.exp((-x - x0) / k)))
#     return y

# # Load the image
# image_path = "/home/student/l/lzejin/codebase/blurring4texture2shape1file/save_dir/imgs/examples/ILSVRC2012_val_00000018.JPEG"
# image = Image.open(image_path)

# # Convert image to numpy array and normalize with ImageNet mean and std
# img_arr = np.array(image).astype(np.float32) / 255.0
# mean = np.array([0.485, 0.456, 0.406])
# std = np.array([0.229, 0.224, 0.225])
# normalized_img = (img_arr - mean) / std
# print(f"range of normalized_img: {np.min(normalized_img)} to {np.max(normalized_img)}")

# # Apply the custom steep sigmoid function
# # Ensure the input is in the expected range [-2.55, 2.55] before applying the function
# sigmoid_img = custom_steep_sigmoid(normalized_img)

# # Convert back to image format
# sigmoid_image = Image.fromarray(np.uint8(sigmoid_img))

# # Display the original and sigmoid-applied images
# plt.figure(figsize=(12, 6))

# # Original image
# plt.subplot(1, 2, 1)
# plt.imshow(image)
# plt.title('Original Image')
# plt.axis('off')

# # Sigmoid-applied image
# plt.subplot(1, 2, 2)
# plt.imshow(sigmoid_image)
# plt.title('After Custom Steep Sigmoid')
# plt.axis('off')

# plt.savefig('save_dir/actiation_funcs_plots/custom_steep_sigmoid_before_and_after.png', format='png')
# plt.show()


# ----


import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch 

# Normalize image function
def normalize_img(img_arr):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return (img_arr - mean) / std

# define images.byte() # tensor_imgs.to(torch.uint8) is equal to imgs.byte()
def byte(imgs):
    # Convert numpy array to a tensor
    tensor_imgs = torch.tensor(imgs)
    # Convert to byte type
    return tensor_imgs.to(torch.uint8)
# Function to process the images with byte()
def process_image_byte(image_arr):
    # Normalize and convert the numpy array to tensor
    tensor_img = torch.from_numpy(normalize_img(image_arr))
    # Convert to byte type
    byte_tensor_img = tensor_img.to(torch.uint8)
    # The `.half()` is not necessary for display, and would actually reduce the precision of the image.
    # If needed for computation, you would use it here.
    # half_tensor_img = byte_tensor_img.half()
    # Return as numpy array for plotting
    return byte_tensor_img.numpy()


# Function to process the image with strong_flatter_leaky_relu
def process_image_leaky_relu(image_arr):
    return strong_flatter_leaky_relu(normalize_img(image_arr))#.astype(np.int32)
# Define the strong_flatter_leaky_relu function
def strong_flatter_leaky_relu(imgs):
    imgs[imgs <= -1] = -100 * (imgs[imgs <= -1] + 1)
    imgs[(imgs >= 1)] *= 1
    imgs[(imgs > -1) & (imgs < 1)] = 0
    return imgs
 

def custom_steep_double_sigmoid(x, x0=-1, k=0.05):
    y = np.where(x < 0, 
                 255 / (1 + np.exp((x - x0) / k))+2, 
                 -2 / (1 + np.exp((-x - x0) / k))+2)
    y = np.clip(y, 0, 255)
    return y
# Function to process the image with custom_steep_sigmoid
def process_image_double_sigmoid(image_arr):
    return custom_steep_double_sigmoid(normalize_img(image_arr))#.astype(np.int32)

def custom_steep_positive_double_sigmoid(x, x0=-1, k=0.05):
    y = np.where(x < 0, 
                 255 / (1 + np.exp((x - x0) / k)), 
                 2 / (1 + np.exp((-x - x0) / k)))
    return y
# Function to process the image with custom_steep_positive_double_sigmoid
def process_custom_steep_positive_double_sigmoid(image_arr):
    return custom_steep_positive_double_sigmoid(normalize_img(image_arr))#.astype(np.int32)


def custom_shifted_minus_1_steep_sigmoid(x, x0=0, k=0.05, left_shift=-1, down_shift=-2):
    # Adjust x0 by adding left_shift to move the sigmoid curve to the left by 1
    x0_adjusted = x0 + left_shift
    # Compute the sigmoid function and shift the result down by 3
    y = 255 / (1 + np.exp((x - x0_adjusted) / k)) - down_shift 
    # Ensure that y is within the valid range of image pixel values after the shift
    y = np.clip(y, 0, 255)
    return y
def process_custom_shifted_minus_1_steep_sigmoid(image_arr):
    return custom_shifted_minus_1_steep_sigmoid(normalize_img(image_arr))#.astype(np.int32)

def custom_shifted_minus_1_not_steep_sigmoid(x, x0=0, k=0.2, left_shift=-1, down_shift=-2):
    # Adjust x0 by adding left_shift to move the sigmoid curve to the left by 1
    x0_adjusted = x0 + left_shift
    # Compute the sigmoid function and shift the result down by 3
    y = 255 / (1 + np.exp((x - x0_adjusted) / k)) - down_shift 
    # Ensure that y is within the valid range of image pixel values after the shift
    y = np.clip(y, 0, 255)
    return y
def process_custom_shifted_minus_1_not_steep_sigmoid(image_arr):
    return custom_shifted_minus_1_not_steep_sigmoid(normalize_img(image_arr))#.astype(np.int32)

def custom_shifted_steep_sigmoid(x, x0=0, k=0.05, left_shift=-1, down_shift=-2):
    # Adjust x0 by adding left_shift to move the sigmoid curve to the left by 1
    x0_adjusted = x0 + left_shift
    # Compute the sigmoid function and shift the result down by 3
    y = 255 / (1 + np.exp((x - x0_adjusted) / k)) - down_shift 
    # Ensure that y is within the valid range of image pixel values after the shift
    y = np.clip(y, 0, 255)
    return y
# Function to process the image with the shifted double sigmoid
def process_custom_shifted_steep_sigmoid(image_arr):
    return custom_shifted_steep_sigmoid(normalize_img(image_arr))#.astype(np.int32)



def custom_shifted_not_steep_sigmoid(x, x0=0, k=0.2, left_shift=0, down_shift=-2):
    # Adjust x0 by adding left_shift to move the sigmoid curve to the left by 1
    x0_adjusted = x0 + left_shift
    # Compute the sigmoid function and shift the result down by 3
    y = 255 / (1 + np.exp((x - x0_adjusted) / k)) - down_shift 
    # Ensure that y is within the valid range of image pixel values after the shift
    y = np.clip(y, 0, 255)
    return y
def process_custom_shifted_not_steep_sigmoid(image_arr):
    return custom_shifted_not_steep_sigmoid(normalize_img(image_arr))#.astype(np.int32)


# Directory with images
directory = "/home/student/l/lzejin/codebase/blurring4texture2shape1file/save_dir/imgs/examples"
image_paths = [os.path.join(directory, fname) for fname in os.listdir(directory) if fname.endswith('.JPEG')]

# Set up the plot
fig, axes = plt.subplots(len(image_paths), 9, figsize=(45, 5 * len(image_paths)))  # 3 columns for each processing type

# Process and plot each image
for idx, image_path in enumerate(image_paths):
    original_image = Image.open(image_path)
    img_arr = np.array(original_image).astype(np.float32) / 255.0
    
    # Process the images with both functions separately
    byte_image = Image.fromarray(np.uint8(process_image_byte(img_arr)))
    leaky_relu_image = Image.fromarray(np.uint8(process_image_leaky_relu(img_arr) ))
    sigmoid_image = Image.fromarray(np.uint8(process_image_double_sigmoid(img_arr) )) #! Not unit8, otherwise -1-2 to 255 , don't * 255 weird
    custom_steep_positive_double_sigmoid_image = Image.fromarray(np.uint8(process_custom_steep_positive_double_sigmoid(img_arr) ))
    custom_shifted_minus_1_steep_sigmoid_image = Image.fromarray(np.uint8(process_custom_shifted_minus_1_steep_sigmoid(img_arr) ))
    custom_shifted_minus_1_not_steep_sigmoid_image = Image.fromarray(np.uint8(process_custom_shifted_minus_1_not_steep_sigmoid(img_arr) ))
    custom_shifted_steep_sigmoid_image = Image.fromarray(np.uint8(process_custom_shifted_steep_sigmoid(img_arr) ))
    custom_shifted_not_steep_sigmoid_image = Image.fromarray(np.uint8(process_custom_shifted_not_steep_sigmoid(img_arr) ))
    
    # Plot original image
    axes[idx, 0].imshow(original_image)
    axes[idx, 0].set_title(f'Original - {os.path.basename(image_path)}')
    axes[idx, 0].axis('off')

    # Plot byte processed image
    axes[idx, 1].imshow(byte_image)
    axes[idx, 1].set_title('Byte Processed')
    axes[idx, 1].axis('off')

    # Plot leaky ReLU processed image
    axes[idx, 2].imshow(leaky_relu_image)
    axes[idx, 2].set_title('Strong Leaky ReLU Processed')
    axes[idx, 2].axis('off')

    # Plot sigmoid processed image
    axes[idx, 3].imshow(sigmoid_image)
    axes[idx, 3].set_title('Double Sigmoid Processed')
    axes[idx, 3].axis('off')

    # custom_steep_positive_double_sigmoid
    axes[idx, 4].imshow(custom_steep_positive_double_sigmoid_image)
    axes[idx, 4].set_title('Custom Steep Positive Double Sigmoid Processed')
    axes[idx, 4].axis('off')

    # Plot shifted minus 1 steep sigmoid processed image
    axes[idx, 5].imshow(custom_shifted_minus_1_steep_sigmoid_image)
    axes[idx, 5].set_title('Shifted Minus 1 Steep Sigmoid Processed')
    axes[idx, 5].axis('off')

    # Plot shifted minus 1 not steep sigmoid processed image
    axes[idx, 6].imshow(custom_shifted_minus_1_not_steep_sigmoid_image)
    axes[idx, 6].set_title('Shifted Minus 1 Not Steep Sigmoid Processed')
    axes[idx, 6].axis('off')

    # Plot shifted double sigmoid processed image
    axes[idx, 7].imshow(custom_shifted_steep_sigmoid_image)
    axes[idx, 7].set_title('Shifted Sigmoid Steep Processed')
    axes[idx, 7].axis('off')

    # Plot custom_shifted_not_steep_sigmoid processed image
    axes[idx, 8].imshow(custom_shifted_not_steep_sigmoid_image)
    axes[idx, 8].set_title('Shifted Not Steep Sigmoid Processed')
    axes[idx, 8].axis('off')

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig('all_examples_9type_before_and_after_processing.png', format='png')
plt.show()
