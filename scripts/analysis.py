import os
import pickle
import datetime
import numpy as np
import pandas as pd
from itertools import product
import argparse
import torch
import torch.nn as nn

from neuroai.evaluation.shape_bias_eval import get_shape_bias

import main  # Note: If unused, you can safely remove this import.
from evd.analysis.save import (
    save_shape_biases_across_time_csv,
    save_shape_biases_across_categories_csv
)
from evd.models.eval import validate
import evd.utils
from evd.datasets.dataset_loader import get_test_loaders
from evd.analysis.plot import plot_shape_bias_across_time
from evd.analysis.adv_attack import adv_attack_analysis, ATTACKS, EPSILONS_DICT



################################################################################
#                           ANALYSIS CONFIGURATIONS
################################################################################

# Toggles for different evaluations/plots.
EVALUATE_ACC = 1                 # Fig1: Evaluate standard accuracy
EVALUATE_BIAS = 0                # Fig1: Evaluate shape bias
PLOT_SHAPE_BIAS_ACROSS_EPOCH = 0  # 0 = No, 1 = Generate shape bias plot across epochs

EVALUATE_ACC_ROBUSTNESS = 1       # Fig2: Evaluate robustness (not fully shown here)
PLOT_ACC_ROBUSTNESS = 0           # Fig2: Plot robustness results (not fully shown here)

# Mode for shape bias evaluation: 'single' uses a static set of epochs,
# 'traverse' scans through a range of epochs.
MODE = ['single', "traverse"][0]  # 0 = 'single', 1 = 'traverse'
if MODE == 'single':
    EPOCHS_list = ['last', 'best'][-1:]  # Example: pick 'best'
    EPOCHS = EPOCHS_list
else:
    EPOCHS = list(range(4, 150, 5))  # Traverse from epoch 4 to 145 in steps of 5


# Dataset / model training identifiers
dataset_id = 1  # 0=texture2shape_miniecoset, 1=ecoset, 2=imagenet, 3=imagenet_16, 4=facescrub
DATASET_TRAINED_ON = ['texture2shape_miniecoset', 'ecoset_square256', 'imagenet','imagenet_16', 'facescrub'][dataset_id]
IMAGE_SIZE = 256

# Choose model architecture by index
model_type = [ 'resnet50', 'alexnet', 'vgg16','resnet101','resnet152','resnext50_32x4d','blt_vs'
                ][0]  # i.e., 'resnet50'
# Hyperparameters for EVD
months_per_epochs = [1.0,2.0, 8.0, 4.0,  3.0, 0.5,  0.25][:0]
contrast_thresh = [0.8, 0.2, 0.1, 0.4, 0.05][:0] # Alpha for contrast
contrast_spd    = [150.0, 50.0,100.0, 300.0][:0] # Beta for contrast 

#! Setting Saving name for current analysis
today = datetime.date.today() # Get the current date
formatted_date = f"{today.day}th_{today.strftime('%B')}"# Format day and month: e.g., "8th_April"
analysis_id = f'{formatted_date}_{DATASET_TRAINED_ON}_{EPOCHS[0]}_'  # Prefix for output files

acc_save_dir = './results/acc/'
shape_bias_save_dir = './results/shape_bias/' 
shape_bias_per_class_save_dir = './results/shape_bias/shape_bias_per_class' 
plot_dir = './results/plots/'
os.makedirs(acc_save_dir, exist_ok=True)
os.makedirs(shape_bias_per_class_save_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

################################################################################
#                     MODEL NAME -> CHECKPOINT PATH MAPPINGS
################################################################################
# Option 1: add model name to path mannually
MODEL_NAME2PATH = {
    # mini
    # 'adult':f"resnet50_mpe2.0_alpha0.2_dn100.0_texture2shape_miniecoset256_0.0001_dev_adult_b1c1cs1_T_normal_seed_1",
    # 'mpe1.0_alpha1.6_dn150.0_':'resnet50_mpe1.0_alpha1.6_dn150.0_texture2shape_miniecoset256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',

    # extra
    # 'mpe1.0_alpha0.2_dn150.0_lr5e-5':'resnet50_mpe1.0_alpha0.2_dn150.0_ecoset_square256256_0.0005_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'mpe1.0_alpha0.8_dn50.0':'resnet50_mpe1.0_alpha0.8_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'mpe1.0_alpha0.8_dn150.0':"resnet50_mpe1.0_alpha0.8_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    # 'mpe1.0_alpha0.4_dn150.0':"resnet50_mpe1.0_alpha0.4_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    
    'adult':"resnet50_mpe2.0_alpha0.2_dn100.0_ecoset_square256256_0.0001_dev_adult_b1c1cs1_T_normal_seed_1",

    # "mpe1.0_alpha0.4_dn50.0": 'resnet50_mpe1.0_alpha0.4_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # "mpe1.0_alpha0.2_dn150.0": 'resnet50_mpe1.0_alpha0.2_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # "mpe1.0_alpha0.1_dn50.0": 'resnet50_mpe1.0_alpha0.1_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # "mpe1.0_alpha0.1_dn100.0": 'resnet50_mpe1.0_alpha0.1_dn100.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',

    # 'mpe2.0_alpha0.2_dn300.0':"resnet50_mpe2.0_alpha0.2_dn300.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    # 'mpe2.0_alpha0.2_dn150.0':"resnet50_mpe2.0_alpha0.2_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    # 'mpe2.0_alpha0.2_dn50.0':"resnet50_mpe2.0_alpha0.2_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    # 'mpe2.0_alpha0.2_dn100.0':"resnet50_mpe2.0_alpha0.2_dn100.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    # "mpe2.0_alpha0.1_dn150.0": 'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'mpe2.0_alpha0.1_dn100.0':'resnet50_mpe2.0_alpha0.1_dn100.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'mpe2.0_alpha0.1_dn50.0':'resnet50_mpe2.0_alpha0.1_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',

    # "mpe3.0_alpha0.4_dn50.0": 'resnet50_mpe3.0_alpha0.4_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # "mpe4.0_alpha0.4_dn50.0": 'resnet50_mpe4.0_alpha0.4_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # "mpe4.0_alpha0.4_dn150.0": 'resnet50_mpe4.0_alpha0.4_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # # "mpe4.0_alpha0.8_dn50.0": 'resnet50_mpe4.0_alpha0.8_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',

    # "mpe8.0_alpha0.05_dn150.0": 'resnet50_mpe8.0_alpha0.05_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # "mpe8.0_alpha0.2_dn50.0": 'resnet50_mpe8.0_alpha0.2_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # "mpe8.0_alpha0.8_dn50.0": 'resnet50_mpe8.0_alpha0.8_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',

    #* Debug 
    "resnet50_EVD_B1": 'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',

    }
# Convert into REAL PATH
MODEL_NAME2PATH = {model_name: f"logs/{model_path}/weights/checkpoint_{EPOCHS[0]}.pth" if model_path else None for model_name, model_path in MODEL_NAME2PATH.items()}

# Option 2: tarverse hyperparameters to include trade-off models
for mpe, alpha, beta in product(months_per_epochs, contrast_thresh, contrast_spd):
    model_save_name = f'mpe{mpe}_alpha{alpha}_dn{beta}' #* Setting name for models
    model_subdir = (
        f"{model_type}_mpe{mpe}_alpha{alpha}_dn{beta}_"
        f"{DATASET_TRAINED_ON}256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1"
    )
    MODEL_NAME2PATH[model_save_name] = f"logs/{model_subdir}/weights/checkpoint_{EPOCHS[0]}.pth"



################################################################################
#           DATAFRAMES FOR STORING ACCURACY AND SHAPE BIAS RESULTS
################################################################################

acc_df = pd.DataFrame(columns=['model_name', 'epoch', 'top1', 'top5'])
shape_bias_across_time_df = pd.DataFrame(
    columns=['model_name', 'epoch', 'top1', 'top5', 'shape_bias', 'timepoint']
)

################################################################################
#                           EVALUATE ACCURACY
################################################################################

if EVALUATE_ACC or EVALUATE_ACC_ROBUSTNESS:
    # Load general config shared across models
    config_file = '/share/klab/lzejin/lzejin/codebase/P001_evd_gpus/evd/models/config/config.yaml'
    config_dict = evd.utils.load_config(config_file)
    args = argparse.Namespace(**config_dict)
    args.dataset_name = DATASET_TRAINED_ON

    # Create test loader once for all models
    test_loader = get_test_loaders(args)

################################################################################
#                   LOOP OVER MODELS AND EPOCHS TO EVALUATE
################################################################################

for model_name, model_path in MODEL_NAME2PATH.items():

    # If we're traversing many epochs for certain models, override EPOCHS
    if 'mpe1' in model_name and MODE != 'single':
        EPOCHS = list(range(4, 300, 5))

    for epoch in EPOCHS:
        # try:
            # Replace the single epoch in model_path with the current epoch
            if model_path is not None:
                current_model_path = model_path.replace(
                    f"_{EPOCHS[0]}.pth",
                    f"_{epoch}.pth"
                )
            else:
                current_model_path = None

            ####################################################################
            #   1) Evaluate Standard Accuracy
            ####################################################################
            # Preload models for acc realted analysis
            if EVALUATE_ACC or EVALUATE_ACC_ROBUSTNESS:
                # Load model architecture from config
                args.arch = model_type
                model, linear_keyword = evd.models.loader.create_model(args)

                # Load checkpoint if available
                if current_model_path:
                    evd.models.loader.load_checkpoint(
                        model=model,
                        model_path=current_model_path,
                        log_dir=None,
                        args=None
                    )

            if EVALUATE_ACC:
                # Evaluate the model
                criterion = (
                    nn.CrossEntropyLoss().cuda()
                    if torch.cuda.is_available()
                    else nn.CrossEntropyLoss()
                )
                top1, top5 = validate(test_loader, model, criterion, epoch)
                print(f"Model {model_name}, Epoch {epoch} -> Top1: {top1:.2f}, Top5: {top5:.2f}\n")

                # Save results
                acc_df = acc_df.append({
                    'model_name': model_name,
                    'epoch': epoch,
                    'top1': float(top1),
                    'top5': float(top5)
                }, ignore_index=True)

                acc_df.to_csv(
                    os.path.join(acc_save_dir, f"{analysis_id}acc.csv"),
                    index=False
                )

            ####################################################################
            #  2) Evaluate Shape Bias
            ####################################################################
            if EVALUATE_BIAS:
                model_type_for_eval = 'resnet50_ours' #* can be replace by model_type + '_ours'
                evaluation = True
                load_checkpoint_path = current_model_path  # Use the same path as above
                dataset_trained_on = DATASET_TRAINED_ON
                data_normalization_type = "0-1"  # or "mean-std"
                matrixplot_savedir = None

                shape_bias, shape_bias_across_classes, classes = get_shape_bias(
                    model_type_for_eval,
                    evaluation=evaluation,
                    load_checkpoint_path=load_checkpoint_path,
                    dataset_trained_on=dataset_trained_on,
                    data_normalization_type=data_normalization_type,
                    matrixplot_savedir=matrixplot_savedir
                )
                print(f"Model {model_name}, Epoch {epoch} -> Shape bias: {shape_bias:.2f}\n")

                # Append overall shape bias to a time-based DataFrame
                new_row = {
                    'model_name': model_name,
                    'model_type': model_type_for_eval,
                    'epoch': epoch,
                    'shape_bias': shape_bias,
                    'timepoint': 0
                }
                if EVALUATE_ACC:
                    new_row['top1'] = float(top1)
                    new_row['top5'] = float(top5)

                shape_bias_across_time_df = shape_bias_across_time_df.append( new_row, ignore_index=True)
                shape_bias_csv_path = os.path.join(shape_bias_save_dir, f"{analysis_id}shape_bias.csv")
                shape_bias_across_time_df.to_csv(shape_bias_csv_path, index=False)
                print(f"Saved shape bias across time to {shape_bias_csv_path}")

                # Also save per-class shape bias
                shape_bias_classes_df = pd.DataFrame({
                    'class_name': classes,
                    'shape_bias_per_class': shape_bias_across_classes
                })
                class_bias_csv_path = os.path.join(
                    shape_bias_per_class_save_dir,
                    f"{model_name}",
                    f"shape_bias_per_class_epoch_{epoch}.csv"
                )
                os.makedirs(os.path.dirname(class_bias_csv_path), exist_ok=True)
                shape_bias_classes_df.to_csv(class_bias_csv_path, index=False)
                print(f"Saved per-class shape bias to {class_bias_csv_path}")

            ################################################################################
            #  OPTIONAL: PLOT SHAPE BIAS ACROSS EPOCHS
            ################################################################################

            if PLOT_SHAPE_BIAS_ACROSS_EPOCH:
                # Example usage; update paths as needed
                csv_file_path = (
                    '/home/student/l/lzejin/codebase/All-TNNs/P001_evd_gpus/'
                    'results/shape_bias/19th_Wed_traverse_4_shape_bais_plot.csv'
                )
                output_plot_path = (
                    '/home/student/l/lzejin/codebase/All-TNNs/P001_evd_gpus/'
                    'results/shape_bias/shape_bias_plot.pdf'
                )
                plot_shape_bias_across_time(csv_file_path, save_path=output_plot_path)

            ################################################################################
            #  3) ADVERSARIAL ATTACK EVALUATION (OPTIONAL ROBUSTNESS TEST)
            ################################################################################

            # If you wish to evaluate adversarial robustness, set EVALUATE_ACC_ROBUSTNESS = 1.
            # This code integrates seamlessly with the main script above. It assumes that
            # the model is already loaded and named `model`, and that a test DataLoader is
            # named `test_loader`. The rest is contained within these utility functions.

            if EVALUATE_ACC_ROBUSTNESS:
                selected_adv_attack_names = [
                    'L2AdditiveGaussianNoiseAttack',
                    'L2AdditiveUniformNoiseAttack',
                    # 'SaltAndPepperNoiseAttack'
                ]
                selected_attacks = {k: v for k, v in ATTACKS.items() if k in selected_adv_attack_names}
                
                adv_attack_df = adv_attack_analysis(model, test_loader, selected_attacks, EPSILONS_DICT, 
                                    model_name=model_name, epoch=epoch, device='cuda' if torch.cuda.is_available() else 'cpu')
                print(f"adv_attack_df:\n{adv_attack_df}")

        # except Exception as e:
        #     print(f"Skipping Model {model_name}, Epoch {epoch} due to error: {e}")
        #     pass


    

