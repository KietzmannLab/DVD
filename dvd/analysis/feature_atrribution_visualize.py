import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import json

# Captum library for interpretability
from captum.attr import LayerLRP

################################################
# 1. Utility Functions
################################################

def load_image(image_path, image_size=256):
    """
    Loads and preprocesses the input image, returns a 1x3xHxW tensor.
    """
    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)  # [1,3,H,W]

def tensor_to_numpy_for_display(img_tensor):
    """
    Converts a normalized image tensor [1,3,H,W] back to float32 NumPy [H,W,3].
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    x = img_tensor.squeeze(0).cpu().detach().numpy().transpose(1,2,0)
    x = (x * std + mean).clip(0,1)
    return x

def upsample_to_img_size(relevance_map, target_size):
    """
    Upsample a relevance map from [1,1,h,w] or [1,C,h,w] to [1,1,H,W].
    `target_size` is a tuple (H, W).
    """
    if relevance_map.shape[2:] == target_size:
        return relevance_map
    
    # Bilinear upsample to match input image size
    return F.interpolate(
        relevance_map, 
        size=target_size, 
        mode='bilinear', 
        align_corners=False
    )

def compute_lrp_map(model, input_tensor, target_class, layer='layer4'):
    """
    Compute LRP for the specified model and layer, returning a single-channel
    relevance map at the same spatial size as that layer's output.
    """
    # Instead of model.layer4 (a Sequential), choose e.g. layer4[-1].conv3
    if layer == 'layer4':
        lrp_layer = model.layer4[-1].conv3
    elif layer == 'layer3':
        lrp_layer = model.layer3[-1].conv3
    elif layer == 'layer2':
        lrp_layer = model.layer2[-1].conv3
    elif layer == 'layer1':
        lrp_layer = model.layer1[-1].conv3
    elif layer == 'fc':
        lrp_layer = model.fc
    else:
        # fallback
        lrp_layer = model.layer4[-1].conv3

    captum_lrp = LayerLRP(model, lrp_layer)
    relevance = captum_lrp.attribute(input_tensor, target=target_class)

    # If multi‐channel (e.g. [1,C,H',W']), sum over channels -> [1,1,H',W']
    if relevance.shape[1] > 1:
        relevance = relevance.sum(dim=1, keepdim=True)
    return relevance

################################################
# 2. Visualization
################################################

def plot_lrp_overlay(orig_img_np, relevance_map, 
                     alpha=0.5, cmap='jet', 
                     subplot_ax=None, title_str=None):
    """
    Overlays a single-channel relevance_map (as a heatmap) on top of
    orig_img_np (H,W,3 in [0,1] range). 
    - alpha: transparency of the heatmap
    - If subplot_ax is given, draw into that subplot. Otherwise, create a new fig.
    """
    # Convert the relevance map to CPU/NumPy
    r = relevance_map.squeeze(0).squeeze(0).cpu().detach().numpy()
    # Normalize for better heatmap visibility
    r_min, r_max = r.min(), r.max()
    if r_max > r_min:
        r = (r - r_min) / (r_max - r_min)

    if subplot_ax is None:
        fig, subplot_ax = plt.subplots()
    
    subplot_ax.imshow(orig_img_np)            # Show original
    subplot_ax.imshow(r, cmap=cmap, alpha=alpha)  # Heatmap overlay
    subplot_ax.axis('off')
    
    if title_str is not None:
        subplot_ax.set_title(title_str)

################################################
# 3. Putting it all together with subplots
################################################

def visualize_lrp_comparison(
    model_dict,
    image_paths,
    image_size=256,
    layer='layer4',
    save_path = f"./results/plots/visualize_features/layer4/lrp_comparison.pdf",
    lookup_json_path = '/share/klab/datasets/optimized_datasets/lookup_ecoset_json.json', 
):
    """
    Visualize LRP comparisons across multiple models, with original images alongside.

    Args:
        model_dict (dict): Mapping of model names to checkpoint file paths.
        image_paths (list): List of image file paths to process.
        lookup_json_path (str): Path to the JSON lookup table mapping class IDs to names.
        image_size (int, optional): Size to which images are resized. Defaults to 256.
        layer (str, optional): Model layer at which to compute LRP. Defaults to 'layer4'.

    The grid will have rows = number of images, columns = number of models + 1.
    The first column shows the original image. Subsequent columns show the LRP overlay
    for each model, with titles indicating model name and predicted category.
    """
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load lookup table
    with open(lookup_json_path, 'r') as f:
        lookup = json.load(f)

    # Load all models
    loaded_models = model_dict

    n_images = len(image_paths)
    n_models = len(loaded_models)
    n_cols = n_models + 1  # extra column for original image

    fig, axes = plt.subplots(
        n_images,
        n_cols,
        figsize=(4 * n_cols, 4 * n_images),
        squeeze=False
    )

    for row_idx, img_path in enumerate(image_paths):
        # Load & preprocess image
        input_tensor = load_image(img_path, image_size).to(device)
        orig_img_np = tensor_to_numpy_for_display(input_tensor)

        # 1) Display original image in first column
        ax_orig = axes[row_idx, 0]
        ax_orig.imshow(orig_img_np)
        ax_orig.axis('off')
        ax_orig.set_title('Original')

        # 2) Compute and plot LRP overlays for each model
        for col_offset, (model_name, model_obj) in enumerate(loaded_models.items(), start=1):
            # Forward pass to get predicted class
            with torch.no_grad():
                logits = model_obj(input_tensor)
                probs = torch.softmax(logits, dim=1)
                pred_id = probs.argmax(dim=1).item()

            # Map prediction ID (0-based) to category name (lookup is 1-based)
            cat_key = f"{pred_id + 1:03d}"
            pred_label = lookup[cat_key]['category']


            # Compute LRP
            relevance = compute_lrp_map(model_obj, input_tensor, pred_id, layer=layer)
            upsampled = upsample_to_img_size(relevance, (image_size, image_size))

            # Plot overlay
            ax = axes[row_idx, col_offset]
            title_str = f"{model_name}\n(pred: {pred_label})"
            plot_lrp_overlay(
                orig_img_np,
                upsampled,
                alpha=0.6,
                cmap='jet',
                subplot_ax=ax,
                title_str=title_str
            )

    plt.tight_layout()
    # Save or show
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)

    print(f"Saved combined LRP figure to {save_path}")


########################################
# 4. Example Usage
########################################

if __name__ == "__main__":

    MODEL_NAME2PATH = {
        'DVD': '',
        'baseline': '',
    }
    
    # Suppose you have cue-conflict images: 
    TEST_IMAGES = [
        "./data/cue-conflict/cat/cat7-elephant1.png",
        "./data/cue-conflict/elephant/elephant6-bottle2.png",
    ]

    # Call the comparison function
    visualize_lrp_comparison(
        model_dict=MODEL_NAME2PATH,
        image_paths=TEST_IMAGES,
        image_size=256,
        layer='layer4',   # The last conv block. Could also do 'fc' or others.
        save_path = f"./results/plots/visualize_features/{layer}/lrp_comparison.pdf",
    )


