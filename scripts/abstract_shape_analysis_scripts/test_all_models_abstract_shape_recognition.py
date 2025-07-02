
import os
import io
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm

import torchvision.models as zoomodels
from neuroai.models import pytorch_model_zoo, tensorflow_model_zoo, list_models
# from neuroai.models.pytorch import model_zoo

# Load the IllusionBench dataset
dataset = load_dataset("arshiahemmat/IllusionBench")
illusion_in = dataset['Illusion_IN']

# Load ImageNet class labels
with open("./data/imagenet_classes.txt") as f:
    imagenet_labels = [line.strip() for line in f.readlines()]

# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
])

class IllusionDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image_name = item['image_name']
        image_data = item['image']

        try:
            shape, scene = image_name.split('-')[:2]
        except ValueError:
            return None

        if isinstance(image_data, dict):
            if 'array' in image_data:
                image = Image.fromarray(np.array(image_data['array']))
            elif 'bytes' in image_data:
                image = Image.open(io.BytesIO(image_data['bytes']))
            else:
                return None
        else:
            image = image_data

        if self.transform:
            image = self.transform(image)

        return image, shape, scene, image_name

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    images, shapes, scenes, names = zip(*batch)
    images = torch.stack(images)
    return images, shapes, scenes, names

def load_model(model_name, *args, **kwargs):

    if model_name in zoomodels.__dict__:
        model = eval("pytorch_model_zoo.model_pytorch")(model_name,  *args, **kwargs)
        framework = 'pytorch'
    elif model_name in list_models("pytorch"):
        model = eval(f"pytorch_model_zoo.{model_name}")(model_name, *args, **kwargs)
        framework = 'pytorch'
    elif model_name in list_models('tensorflow'):
        model = eval(f"tensorflow_model_zoo.{model_name}")(model_name, *args, **kwargs)
        framework = 'tensorflow'
    else:
        raise NameError(f"Model {model_name} is not supported.")
    return model, framework

# Create dataset and dataloader
batch_size = 32 #512 #1024 # 2048 #512
illusion_dataset = IllusionDataset(illusion_in, transform=preprocess)
illusion_dataloader = DataLoader(
    illusion_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    collate_fn=custom_collate_fn
)

# Define models to evaluate
network_names = [
    'resnet50_baseline_imagenet',
    'resnet50_DVD_S_imagenet',
    'resnet50_DVD_B_imagenet',
    'resnet50_DVD_P_imagenet',
    'resnet50_DVD_PP_imagenet',

    'resnet50_trained_on_SIN',
    'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN',
    'resnet50_trained_on_SIN_and_IN',


    # 'alexnet', 
    # 'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
    # 'squeezenet1_0', 'squeezenet1_1', 'densenet121', 'densenet169', 
    # 'densenet201',
    # 'inception_v3', 'resnet18', 'resnet34', 
    # 'resnet50', 
    # 'resnet101', 'resnet152',
    # 'shufflenet_v2_x0_5',
    #  'mobilenet_v2', 
    # 'resnext50_32x4d', 'resnext101_32x8d',
    # 'wide_resnet50_2', 'wide_resnet101_2', 'mnasnet0_5', 'mnasnet1_0',
    # "simclr_resnet50x1",
    # 'vit_small_patch16_224', 'vit_base_patch16_224', 'vit_large_patch16_224',
    # 'ResNeXt101_32x16d_swsl', 'resnet50_swsl',
    # 'BiTM_resnetv2_152x4', 'BiTM_resnetv2_152x2', 'BiTM_resnetv2_101x3',
    # 'BiTM_resnetv2_101x1', 'BiTM_resnetv2_50x3','BiTM_resnetv2_50x1',
    # 'transformer_L16_IN21K', 'transformer_B16_IN21K', 
    # 'efficientnet_l2_noisy_student_475', 

    # #* extra
    # 'MoCo',
    # 'PIRL',
    # 'MoCoV2',
    # 'InfoMin',
    # 'InsDis',
    
    # 'clipRN50',
    # 'clip',
    
    # 'clipRN50_CCM_12m_all_epochs',
    # 'clipRN50_CCM_12m_DVD',

    # 'transformer_L16_IN21K',
    # 'clip',
    # 'clipRN50',
    
    #* VLMs see other files in current dir


]

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for network_name in network_names:
    print(f"\nEvaluating model: {network_name}")
    # net = getattr(model_zoo, network_name)(model_name=network_name).model
    model, framework = load_model(network_name)
    # model.eval()
    # model.to(device)

    results = []
    shape_correct = 0
    scene_correct = 0
    total = 0

    for batch in tqdm(illusion_dataloader, desc=f"Processing with {network_name}"):
        if batch is None:
            continue
        images, shapes, scenes, names = batch
        images = images.to(device)

        with torch.no_grad():
            try:
                # import pdb;pdb.set_trace()
                logits = model.forward_batch(images) # for PytorchModel warpper
                softmax_output = torch.from_numpy(model.softmax(logits))  # convert to tensor
                _, predicted_indices = torch.max(softmax_output, 1)
            except:
                outputs = model(images)
                _, predicted_indices = torch.max(outputs, 1)
            predicted_labels = [imagenet_labels[idx] for idx in predicted_indices.cpu().numpy()]

        for predicted_label, shape, scene, name in zip(predicted_labels, shapes, scenes, names):
            predicted_label_lower = predicted_label.lower()
            result_entry = {
                "image_name": name,
                "predicted_label": predicted_label,
                "shape": shape,
                "scene": scene
            }
            results.append(result_entry)

            if shape and shape.lower() in predicted_label_lower:
                shape_correct += 1
            elif scene and scene.lower() in predicted_label_lower:
                scene_correct += 1
            total += 1

    # Save to CSV
    save_dir  = "./results/illusion_benchmark/raw_data/"
    csv_filename = save_dir + f"results_{network_name}.csv"
    with open(csv_filename, mode='w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "predicted_label", "shape", "scene"])
        writer.writeheader()
        writer.writerows(results)

    # Print summary
    if total > 0:
        shape_ratio = shape_correct / total
        scene_ratio = scene_correct / total
        print(f"Saved results to {csv_filename}")
        print(f"Total images processed: {total}")
        print(f"Shape-based predictions: {shape_correct} ({shape_ratio:.2%})")
        print(f"Scene-based predictions: {scene_correct} ({scene_ratio:.2%})")
    else:
        print("No images were processed.")
