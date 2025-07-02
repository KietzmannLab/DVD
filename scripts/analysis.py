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

# import main  # Note: If unused, you can safely remove this import.
from dvd.analysis.save import (
    save_shape_biases_across_time_csv,
    save_shape_biases_across_categories_csv
)
from dvd.models.eval import validate
from dvd.models.extract_acts import extract_acts_and_create_rdm
import dvd.utils
from dvd.datasets.dataset_loader import get_test_loaders, get_blur_loader, get_distortion_loader
from dvd.analysis.plot import plot_shape_bias_across_time, plot_shape_and_texture_accuracy_across_time
from dvd.analysis.adv_attack import AdversarialRobustnessEvaluator #, ATTACKS, EPSILONS_DICT
from dvd.analysis.feature_atrribution_visualize import visualize_lrp_comparison
from dvd.analysis.degradation_robustness import DegradationRobustnessEvaluator


################################################################################
#                           ANALYSIS CONFIGURATIONS
################################################################################

#* Figure 2 ACC & Shape bias
EVALUATE_ACC = 1              # Fig1: Evaluate standard accuracy
EVALUATE_BIAS = 1             # Fig1: Evaluate shape bias
PLOT_SHAPE_BIAS_ACROSS_EPOCH = 0  # 0 = No, 1 = Generate shape bias plot across epochs

#* Figure 3c | feature atrribution visulisation
FEATURE_ATRRIBUTION_VISUALIZE = 0
models_dict_for_feature_atrribution = {}

#* Figure 4a | Degradation robustness
EVALUATE_DEGRADATION_ROBUSTNESS = 0         # Fig2a: Evaluate degradation #* see shape texture project
PLOT_DEGRADATION_ROBUSTNESS = 0                 # Fig2a: Plot degradation results 
EVALUATE_VARIOUS_DEGRADTION = 0 #* evalute various degradations
EVALUATE_BLUR_DEGRADATIONN = 0 #* evalute blur degradation, need to set face or objects 
DEGRADATION_RESULTS_SAVE_DIR =  './results/degradation_robustness/' 
os.makedirs(DEGRADATION_RESULTS_SAVE_DIR, exist_ok=True)

#* Figure 4b | adv robustness
EVALUATE_ADV_ROBUSTNESS = 0     # Fig2b: Evaluate robustness (not fully shown here)
PLOT_ADV_ROBUSTNESS = 0          # Fig2b: Plot robustness results (not fully shown here)
ADV_ROBUSTNESS_RESULTS_SAVE_DIR =  './results/adv_robustness/' 
SELECTED_ADV_ATTACKS_NAMES = [

                    # "BoundaryAttack", #fail
                    # "PointwiseAttack", "HopSkipJumpAttack", # not supported


                    # "C&W_L2",  # too slow
                    

                    

                    #TODO slow one might just reduce the max_iter size e.g. go 5 -10
                    # "BIM",
                    # "Auto_PGD",  #! A
                    # "C&W_Linf",  # slow? 
                    # "EAD",   # not work

                    # black‑box
                    # "SimBA", # error
                    # "HopSkipJump",  # too slow, seems to be image level
                    # "ZOO", # too slow
                    # "PixelAttack",# too slow
                    # "UniversalPert", #! B #* 1h per model | OK | iterate quick, but still per iamge--> so slow
                    # "Square", # slow, even slow than below

                    # "MI_FGSM", #error

                    #* Blackbox | decision‑based adversarial attacks (Noise-Based Attacks )
                    'L2AdditiveGaussianNoiseAttack',
                    'L2AdditiveUniformNoiseAttack',
                    'SaltAndPepperNoiseAttack',

                    # white‑box
                    "FGSM", "FGM", "PGD", 
                    # "DeepFool",

                    # # #* Whitebox | Gradient-Based Attacks
                    # 'LinfFastGradientAttack', #* Fast Gradient Sign Method (FGSM, ℓ∞)
                    # 'L2FastGradientAttack', #* Fast Gradient Method (FGM, ℓ₂)
                    # 'LinfPGD', #* Projected Gradient Descent (PGD, ℓ∞)
                    # 'LinfAdamPGD',
                    # TODO decision‑based black‑box algorithms (e.g., Boundary Attack, HopSkipJump)
                    # #* Optimization-Based Attacks
                    # 'DDNAttack',
                    # #* Query-Based and Decision-Based Attacks
                    # 'HopSkipJumpAttack',
                    # 'GenAttack',
                    # #* Iterative Adaptive Attacks
                    # 'PointwiseAttack',
                    # # *Transformation-Based Attacks
                    # 'SpatialAttack_rotation',
                    # 'SpatialAttack_translation', 
                    # 'SpatialAttack_scaling', 
                    # 'InversionAttack',
                    # 'LinearSearchContrastReductionAttack',

                ]

#* extract activations for RDMs
EXTRACT_ACTS = 0
EXTRACT_ACTS_DATASETS = ["coco-split515", "skeleton_ayzenberg"][1]
ACTS_RDM_SAVE_DIR =  './results/acts/rdms/skeleton_ayzenberg/' 
os.makedirs(ACTS_RDM_SAVE_DIR, exist_ok=True)


# Mode for shape bias evaluation: 'single' uses a static set of epochs,
# 'traverse' scans through a range of epochs.
MODE = ['single', "traverse"][0]  # 0 = 'single', 1 = 'traverse'
if MODE == 'single':
    EPOCHS_list = ['last', 'best'][1:]  # Example: pick 'best'
    EPOCHS = EPOCHS_list
else:
    # EPOCHS = list(range(4, 300, 5))  # Traverse from epoch 4 to 145 in steps of 5
    EPOCHS =  list(range(4, 300, 5))  #list(range(4, 150, 5))  # Traverse from epoch 4 to 145 in steps of 5


# Dataset / model training identifiers
dataset_id = 1 # 0=texture2shape_miniecoset, 1=ecoset, 2=imagenet, 3=imagenet_16, 4=facescrub
DATASET_TRAINED_ON = ['texture2shape_miniecoset', 'ecoset_square256', 'imagenet','imagenet_16', 'facescrub'][dataset_id]
IMAGE_SIZE = [256,224,None][0] #! Default set as 256 x 256

# Choose model architecture by index
model_type = [ 'all', 'resnet50', 'resnet101', 'deit_base_patch16_224','vit_b_16', 'swin_b', 'alexnet', 'vgg16','resnet101','resnet152','resnext50_32x4d','blt_vs'
                ][1]  # i.e., 'resnet50'
# Hyperparameters for EVD
months_per_epochs = [1.0, 2.0, 4.0, 8.0, 3.0, 0.5,  0.25][:0]
contrast_thresh = [ 0.1, 0.2, 0.4, 0.8, 0.05][:0] # Alpha for contrast
contrast_spd    = [50.0,100.0, 150.0, 300.0][:0] # Beta for contrast 

#! Setting Saving name for current analysis
today = datetime.date.today() # Get the current date
formatted_date = f"{today.day}th_{today.strftime('%B')}"# Format day and month: e.g., "8th_April"
analysis_id = f'{formatted_date}_{DATASET_TRAINED_ON}_across_architecture_{MODE}_{EPOCHS[0]}_'  #! Prefix for output files

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
    #* mini-ecoset adult
    # 'adult':f"resnet50_texture2shape_miniecoset_256_0.0001_dev_adult_seed_1",
    
    # 'adult':f"resnet50_mpe2.0_alpha0.2_dn100.0_texture2shape_miniecoset256_0.0001_dev_adult_b1c1cs1_T_normal_seed_1",
    # 'mpe1.0_alpha1.6_dn150.0_':'resnet50_mpe1.0_alpha1.6_dn150.0_texture2shape_miniecoset256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',

    # extra
    # 'mpe1.0_alpha0.2_dn150.0_lr5e-5':'resnet50_mpe1.0_alpha0.2_dn150.0_ecoset_square256256_0.0005_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'mpe1.0_alpha0.8_dn50.0':'resnet50_mpe1.0_alpha0.8_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'mpe1.0_alpha0.8_dn150.0':"resnet50_mpe1.0_alpha0.8_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    # 'mpe1.0_alpha0.4_dn150.0':"resnet50_mpe1.0_alpha0.4_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    
    # 'adult':"resnet50_mpe2.0_alpha0.2_dn100.0_ecoset_square256256_0.0001_dev_adult_b1c1cs1_T_normal_seed_1",
    
    #? mpe1.0_alpha0.1_dn100 or mpe2.0_alpha0.1_dn100.0
    #TODO try mpe1.0_alpha0.2_dn100.0 mpe1.0_alpha0.2_dn50 
    ##* acc 47.47 | bias 0.80 or 42.77 | bias 0.89 -->  --> | TVD-S
    # "mpe1.0_alpha0.2_dn150.0": 'resnet50_mpe1.0_alpha0.2_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    ## acc 55.89 | bias 0.81
    # "mpe1.0_alpha0.1_dn50.0": 'resnet50_mpe1.0_alpha0.1_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    ##* acc 50.24 | bias 0.87 or 49.31 bias 0.88 | 55.20 | 0.878 -->  --> | TVD-S
    # "mpe1.0_alpha0.1_dn100.0": 'resnet50_mpe1.0_alpha0.1_dn100.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    ## acc 46.04 | bais 0.81
    # "mpe1.0_alpha0.4_dn50.0": 'resnet50_mpe1.0_alpha0.4_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',

    #? mpe2.0_alpha0.1_dn150.0 or mpe2.0_alpha0.2_dn100.0
    #* acc 56.70 | 0.81 | 149ep then go down | or 58.1 | 0.81  -->  --> | TVD-B
    # "mpe2.0_alpha0.1_dn150.0": 'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # #* acc 58.80 | 0.77 or 57.8 | 0.83
    # #! 55.20 | 0.878 -->  --> | TVD-S
    # 'mpe2.0_alpha0.1_dn100.0':'resnet50_mpe2.0_alpha0.1_dn100.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # acc 57.89 | 0.8009685230024213 -->  --> | TVD-B
    # 'mpe2.0_alpha0.2_dn50.0':"resnet50_mpe2.0_alpha0.2_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    #* acc 57.29 | 0.80  -->  --> | TVD-B (every stable and might can keep training, stable at end)
    # 'mpe2.0_alpha0.2_dn100.0':"resnet50_mpe2.0_alpha0.2_dn100.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    # acc 56. 41 | bias 0.77
    # 'mpe2.0_alpha0.2_dn150.0':"resnet50_mpe2.0_alpha0.2_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    # acc 55.81 | bias 0.79
    # 'mpe2.0_alpha0.2_dn300.0':"resnet50_mpe2.0_alpha0.2_dn300.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    # acc 60.32  | 0.79 or 58.37 | 0.81
    # 'mpe2.0_alpha0.1_dn50.0':'resnet50_mpe2.0_alpha0.1_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',

    # acc 61.44 | 0.71
    # "mpe3.0_alpha0.4_dn50.0": 'resnet50_mpe3.0_alpha0.4_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    #* 63.74 | 0.70 | --> good performance favored | TVD-P 
    # "mpe4.0_alpha0.4_dn50.0": 'resnet50_mpe4.0_alpha0.4_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    #* 62.76 | 0.85 or 62.25 0.69 | the best one might be good
    # "mpe4.0_alpha0.4_dn150.0": 'resnet50_mpe4.0_alpha0.4_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # # "mpe4.0_alpha0.8_dn50.0": 'resnet50_mpe4.0_alpha0.8_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',

    # 62.84 | 0.56
    # "mpe8.0_alpha0.05_dn150.0": 'resnet50_mpe8.0_alpha0.05_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    #* 64.15 | 0.58 or 64.44 | 0.54 or 64.81 | 0.48 --> good performance favored | TVD-P2
    # "mpe8.0_alpha0.2_dn50.0": 'resnet50_mpe8.0_alpha0.2_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    #* 64.88 | 0.52 | --> good performance favored | TVD-P2
    # "mpe8.0_alpha0.8_dn50.0": 'resnet50_mpe8.0_alpha0.8_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    
    
    # 'mpe1.0_alpha0.1_dn150.0':"resnet50_mpe1.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    # 'mpe1.0_alpha0.2_dn100.0':'resnet50_mpe1.0_alpha0.2_dn100.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'mpe1.0_alpha0.2_dn50.0':'resnet50_mpe1.0_alpha0.2_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'mpe2.0_alpha0.2_dn100.0':'resnet50_mpe2.0_alpha0.2_dn100.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'swin_b_mpe2.0_alpha0.1_dn150.0':'swin_b_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'swin_b_mpe1.0_alpha0.2_dn100.0':"swin_b_mpe1.0_alpha0.2_dn100.0_texture2shape_miniecoset256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    # 'swin_b_mpe2.0_alpha0.1_dn150.0':'swin_b_mpe2.0_alpha0.1_dn150.0_texture2shape_miniecoset256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'vit_b_16_mpe2.0_alpha0.1_dn150.0':'vit_b_16_mpe2.0_alpha0.1_dn150.0_texture2shape_miniecoset256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'deit_base_patch16_224_mpe2.0_alpha0.1_dn150.0':"deit_base_patch16_224_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    # '':'',
    # 'resnet101_mpe2.0_alpha0.2_dn150.0':'resnet101_mpe2.0_alpha0.2_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'resnet101_mpe2.0_alpha0.1_dn150.0':'resnet101_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    
    #* imagenet
    # 'adult':'resnet50_imagenet_256_0.0001_dev_adult_seed_1',
    # 'resnet50_mpe1.0_alpha0.1_dn100.0':'resnet50_mpe1.0_alpha0.1_dn100.0_imagenet256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'resnet50_mpe2.0_alpha0.1_dn150.0':'resnet50_mpe2.0_alpha0.1_dn150.0_imagenet256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'resnet50_mpe2.0_alpha0.2_dn100.0':'resnet50_mpe2.0_alpha0.2_dn100.0_imagenet256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'resnet50_mpe4.0_alpha0.4_dn50.0':'resnet50_mpe4.0_alpha0.4_dn50.0_imagenet256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'resnet50_mpe8.0_alpha0.8_dn50.0':'resnet50_mpe8.0_alpha0.8_dn50.0_imagenet256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    
    #* miniecoset
    # 'adult':'resnet50_texture2shape_miniecoset_256_0.0001_dev_adult_seed_1',
    # ##  ['mpe2.0_alpha0.1_dn100.0', 'mpe1.0_alpha0.2_dn150.0', 'mpe4.0_alpha0.8_dn50.0']
    # 'resnet50_mpe1.0_alpha0.2_dn150.0':'resnet50_mpe1.0_alpha0.2_dn150.0_texture2shape_miniecoset256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'resnet50_mpe2.0_alpha0.1_dn100.0':'resnet50_mpe2.0_alpha0.1_dn100.0_texture2shape_miniecoset256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'resnet50_mpe4.0_alpha0.8_dn50.0':'resnet50_mpe4.0_alpha0.8_dn50.0_texture2shape_miniecoset256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'resnet50_mpe8.0_alpha0.8_dn100.0':'resnet50_mpe8.0_alpha0.8_dn100.0_texture2shape_miniecoset256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    
    # 'resnet50_mpe1.0_alpha0.1_dn100.0':'resnet50_mpe1.0_alpha0.1_dn100.0_texture2shape_miniecoset256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'resnet50_mpe2.0_alpha0.1_dn150.0':'resnet50_mpe2.0_alpha0.1_dn150.0_texture2shape_miniecoset256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'resnet50_mpe2.0_alpha0.2_dn100.0':'resnet50_mpe2.0_alpha0.2_dn100.0_texture2shape_miniecoset256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'resnet50_mpe4.0_alpha0.4_dn50.0':'resnet50_mpe4.0_alpha0.4_dn50.0_texture2shape_miniecoset256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'resnet50_mpe8.0_alpha0.8_dn50.0':'resnet50_mpe8.0_alpha0.8_dn50.0_texture2shape_miniecoset256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # #? No grayscale aug
    # 'resnet50_mpe1.0_alpha0.2_dn150.0_no_grayscale':'resnet50_mpe1.0_alpha0.2_dn150.0_texture2shape_miniecoset256_0.0001_dev_evd_b1c1cs1_T_normal_seed_100', #0.73 (acc from 45% to 52%)
    # 'resnet50_mpe2.0_alpha0.2_dn150.0_no_grayscale':'resnet50_mpe2.0_alpha0.2_dn150.0_texture2shape_miniecoset256_0.0001_dev_evd_b1c1cs1_T_normal_seed_100', #0.68
    
     # No blurring aug for more fair robustness evaluation
    # 'ecoset_adult_no_blur', 'ecoset_DVD_B_no_blur', 'ecoset_DVD_S_no_blur', 'ecoset_DVD_SS_no_blur', 'ecoset_DVD_P_no_blur', 'ecoset_DVD_PP_no_blur'
    # 'ecoset_adult_no_blur':'resnet50_ecoset_square256_256_0.0001_dev_adult_seed_1_no_blur_aug',
    # 'ecoset_DVD_B_no_blur':'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1_no_blur_aug',
    # 'ecoset_DVD_S_no_blur':'resnet50_mpe1.0_alpha0.1_dn100.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1_no_blur_aug',
    # 'ecoset_DVD_SS_no_blur':'resnet50_mpe1.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1_no_blur_aug',
    # 'ecoset_DVD_P_no_blur':'resnet50_mpe4.0_alpha0.4_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1_no_blur_aug',
    # 'ecoset_DVD_PP_no_blur':'resnet50_mpe8.0_alpha0.2_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1_no_blur_aug',

    # # # #* ecoset
    # # # # Balanced
    # 'resnet50_TVD-B': 'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # ##Baseline
    # 'resnet50_baseline':'resnet50_mpe2.0_alpha0.2_dn100.0_ecoset_square256256_0.0001_dev_adult_b1c1cs1_T_normal_seed_1',
    # Shape bias favored
    # 'resnet50_DVD-SS':
    #     'resnet50_mpe1.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'resnet50_DVD-S':
    #     'resnet50_mpe1.0_alpha0.1_dn100.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # # # Performance favored
    # 'resnet50_DVD-P':
    #     'resnet50_mpe4.0_alpha0.4_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'resnet50_DVD-PP':
    #     'resnet50_mpe8.0_alpha0.2_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',

    #  # #* multiple seeds
    # 'adult_seed_1':'resnet50_mpe2.0_alpha0.2_dn100.0_ecoset_square256256_0.0001_dev_adult_b1c1cs1_T_normal_seed_1',
    # 'adult_seed_2':'resnet50_mpe2.0_alpha0.2_dn100.0_ecoset_square256256_0.0001_dev_adult_b1c1cs1_T_normal_seed_2',
    # 'adult_seed_3':'resnet50_mpe2.0_alpha0.2_dn100.0_ecoset_square256256_0.0001_dev_adult_b1c1cs1_T_normal_seed_3',
    # "mpe2.0_alpha0.1_dn150.0_seed_1": 'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # "mpe2.0_alpha0.1_dn150.0_seed_2": 'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_2',
    # "mpe2.0_alpha0.1_dn150.0_seed_3": 'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_3',

    # # # # # # #* ablation
    # 'resnet50_mpe2.0_alpha0.1_dn150.0_b1c1cs1':'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'resnet50_mpe2.0_alpha0.1_dn150.0_b1c0cs0':'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c0cs0_T_normal_seed_1',
    # 'resnet50_mpe2.0_alpha0.1_dn150.0_b0c1cs0':'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b0c1cs0_T_normal_seed_1',
    # 'resnet50_mpe2.0_alpha0.1_dn150.0_b0c0cs1':'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b0c0cs1_T_normal_seed_1',
    # 'resnet50_mpe2.0_alpha0.1_dn150.0_b1c1cs0':'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs0_T_normal_seed_1',
    # 'resnet50_mpe2.0_alpha0.1_dn150.0_b1c0cs1':'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c0cs1_T_normal_seed_1',
    # 'resnet50_mpe2.0_alpha0.1_dn150.0_b0c1cs1':'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b0c1cs1_T_normal_seed_1',
    # '':'',

    #* SimCLR
    # 'SimCLR_adult': "may_evd_adult_from_79ep_v2", #* 0.77 acc < 15%
    # 'SimCLR_DVD_B': "may_evd_mpe2_0.1_dn150_v2_from_299", #* 0.83 shape bas but acc only < 15%
    # 'SimCLR_DVD_B_249': "may_evd_mpe2_0.1_dn150_v2_from_249", #* 0.66, acc< 17%


    #* other models:
    
    # 'alexnet_DVD_B': "alexnet_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1", 
    # 'alexnet_baseline': "alexnet_ecoset_square256_256_0.0001_dev_adult_seed_1", 
    
    # # 'vgg16_DVD_B': "vgg16_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1", 
    # # 'vgg16_baseline': "vgg16_ecoset_square256_256_0.0001_dev_adult_seed_1", 

    # 'mobilenet_v3_small_baseline': "mobilenet_v3_small_ecoset_square256_256_0.0001_dev_adult_seed_1", 
    # 'mobilenet_v3_small_DVD_B': "mobilenet_v3_small_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1", 
    
    # 'shufflenet_v2_x1_0_baseline': "shufflenet_v2_x1_0_ecoset_square256_256_0.0001_dev_adult_seed_1", 
    # 'shufflenet_v2_x1_0_DVD_B': "shufflenet_v2_x1_0_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",  
    
    # # 'resnet18_baseline': "resnet18_ecoset_square256_256_0.0001_dev_adult_seed_1", 
    # # 'resnet18_DVD_B': "resnet18_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1", 
    
    # # 'resnet101_baseline': "resnet101_ecoset_square256_256_0.0001_dev_adult_seed_1", 
    # # 'resnet101_DVD_B': "resnet101_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1", 
    
    # 'resnext50_32x4d_baseline': "resnext50_32x4d_ecoset_square256_256_0.0001_dev_adult_seed_1", 
    # 'resnext50_32x4d_DVD_B': "resnext50_32x4d_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1", 
    
    # 'deit_base_patch16_224_baseline': "deit_base_patch16_224_ecoset_square256_256_0.0001_dev_adult_seed_1", 
    # 'deit_base_patch16_224_DVD_B': "deit_base_patch16_224_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1", 
    
    # 'vit_b_16_baseline': "vit_b_16_ecoset_square256_256_0.0001_dev_adult_seed_1", 
    # 'vit_b_16_DVD_B': "vit_b_16_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1"


    #* face trained only
    # 'Face_DVD-B_seed_1':'resnet50_mpe2.0_alpha0.1_dn150.0_facescrub256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'Face_DVD-B_seed_2':'resnet50_mpe2.0_alpha0.1_dn150.0_facescrub256_0.0001_dev_evd_b1c1cs1_T_normal_seed_2',
    # 'Face_DVD-B_seed_3':'resnet50_mpe2.0_alpha0.1_dn150.0_facescrub256_0.0001_dev_evd_b1c1cs1_T_normal_seed_3',
    # 'Face_DVD-S_seed_1':'resnet50_mpe1.0_alpha0.1_dn150.0_facescrub256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'Face_DVD-SS_seed_1':'resnet50_mpe1.0_alpha0.1_dn100.0_facescrub256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'Face_DVD-P_seed_1':'resnet50_mpe4.0_alpha0.4_dn50.0_facescrub256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'Face_DVD-PP_seed_1':'resnet50_mpe8.0_alpha0.2_dn50.0_facescrub256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',

    # 'Face_DVD-B_no_blur_seed_1':'resnet50_mpe2.0_alpha0.1_dn150.0_facescrub256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1_no_blur_aug',
    # 'Face_DVD-B_no_blur_seed_2':'resnet50_mpe2.0_alpha0.1_dn150.0_facescrub256_0.0001_dev_evd_b1c1cs1_T_normal_seed_2_no_blur_aug',
    # 'Face_DVD-B_no_blur_seed_3':'resnet50_mpe2.0_alpha0.1_dn150.0_facescrub256_0.0001_dev_evd_b1c1cs1_T_normal_seed_3_no_blur_aug',
    # 'Face_baseline_no_blur_seed_1':'resnet50_facescrub_256_0.0001_dev_adult_seed_1_no_blur_aug',
    # 'Face_baseline_no_blur_seed_2':'resnet50_facescrub_256_0.0001_dev_adult_seed_2_no_blur_aug',
    # 'Face_baseline_no_blur_seed_3':'resnet50_facescrub_256_0.0001_dev_adult_seed_3_no_blur_aug',
    # 'Face_DVD-SSS_mpe0-5_no_blur_seed_1':'resnet50_mpe0.5_alpha0.1_dn150.0_facescrub256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1_no_blur_aug',
    
    # 'Face_baseline_seed_1':'resnet50_facescrub_256_0.0001_dev_adult_seed_1',
    # 'Face_baseline_seed_2':'resnet50_facescrub_256_0.0001_dev_adult_seed_2',
    # 'Face_baseline_seed_3':'resnet50_facescrub_256_0.0001_dev_adult_seed_3',

    # cirtical period
    'DVD-B_random_order': 'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_random_seed_2',
    'DVD-B_fully_random': 'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_fully_random_seed_2',
    'DVD-B_mid_phase': 'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_mid_phase_seed_2',

   
    }
# Convert into REAL PATH
MODEL_NAME2PATH = {model_name: f"logs/{model_path}/weights/checkpoint_{EPOCHS[0]}.pth" if model_path else None for model_name, model_path in MODEL_NAME2PATH.items()}
# MODEL_NAME2PATH = {model_name: f"checkpoints/{model_path}/model_{EPOCHS[0]}.pth" if model_path else None for model_name, model_path in MODEL_NAME2PATH.items()} #* SimCLR

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
    columns=['model_name', 'epoch', 'top1', 'top5', 'shape_bias', 'timepoint',"median_shape_bias","model_type","shape_acc","texture_acc","shape_bias_per_class",]
)

################################################################################
#                           EVALUATE ACCURACY
################################################################################

#* create loader for activation extraction, then still use default test_loader
if EXTRACT_ACTS:
    # Load general config shared across models
    config_file_acts = '/share/klab/lzejin/lzejin/codebase/P001_dvd_gpus/dvd/models/config/config.yaml'
    config_dict_acts = dvd.utils.load_config(config_file_acts)
    args_acts = argparse.Namespace(**config_dict_acts)
    args_acts.dataset_name = EXTRACT_ACTS_DATASETS
    args_acts.batch_size_per_gpu = 25 #* need to assign basize size properly

    # Create test loader once for all models
    test_acts_loader = get_test_loaders(args_acts)
    # import pdb;pdb.set_trace()

if EVALUATE_ACC or EVALUATE_ADV_ROBUSTNESS or EVALUATE_DEGRADATION_ROBUSTNESS or EXTRACT_ACTS or FEATURE_ATRRIBUTION_VISUALIZE:
    # Load general config shared across models
    config_file = '/share/klab/lzejin/lzejin/codebase/P001_dvd_gpus/dvd/models/config/config.yaml'
    config_dict = dvd.utils.load_config(config_file)
    args = argparse.Namespace(**config_dict)
    args.dataset_name = DATASET_TRAINED_ON
    # args.batch_size_per_gpu = 512 #!1024 too large for adv

    # Create test loader once for all models
    test_loader = get_test_loaders(args)


################################################################################
#                   LOOP OVER MODELS AND EPOCHS TO EVALUATE
################################################################################

# for model_name, model_path in MODEL_NAME2PATH.items():
for model_idx, (model_name, model_path) in enumerate(MODEL_NAME2PATH.items()):
    print(f"Evaluating Model {model_name} start from Epoch {EPOCHS[0]}")


    for epoch in EPOCHS:
        # analysis_id = f'{formatted_date}_{DATASET_TRAINED_ON}_{MODE}_{epoch}_'  #! Prefix for output files
        try:
            # Replace the single epoch in model_path with the current epoch
            if model_path is not None:
                current_model_path = model_path.replace(
                    f"_{EPOCHS[0]}.pth",
                    f"_{epoch}.pth"
                )
            else:
                current_model_path = None
            
            #* across different model type
            if model_type == 'all':
                model_type = model_name.replace('_baseline', '').replace('_DVD_B', '') # if model_type == 'all' else model_name
                # assert 'baseline' not in model_type and 'DVD_B' not in model_type
                if 'vit' in model_type or 'deit' in model_type: 
                    IMAGE_SIZE = 224


            ####################################################################
            #   1) Evaluate Standard Accuracy
            ####################################################################
            # Preload models for acc realted analysis
            if EVALUATE_ACC or EVALUATE_ADV_ROBUSTNESS or EVALUATE_DEGRADATION_ROBUSTNESS or EXTRACT_ACTS or FEATURE_ATRRIBUTION_VISUALIZE:
                # Load model architecture from config
                args.arch = model_type
                model, linear_keyword = dvd.models.loader.create_model(args)
                
                # Load checkpoint if available
                if current_model_path:
                    dvd.models.loader.load_checkpoint(model=model, model_path=current_model_path, log_dir=None, args=None)
      

            if EVALUATE_ACC:
                # Evaluate the model
                criterion = (
                    nn.CrossEntropyLoss().cuda()
                    if torch.cuda.is_available()
                    else nn.CrossEntropyLoss()
                )
                top1, top5 = validate(test_loader, model, criterion, epoch,  image_size = IMAGE_SIZE)
                print(f"Model {model_name}, Epoch {epoch} -> Top1: {top1:.2f}, Top5: {top5:.2f}\n")

                # Save results
                new_acc_row = {
                    'model_name': model_name,
                    'epoch': epoch,
                    'top1': float(top1),
                    'top5': float(top5)
                }
                acc_df = pd.concat([acc_df, pd.DataFrame([new_acc_row])], ignore_index=True)


                acc_df.to_csv(
                    os.path.join(acc_save_dir, f"{analysis_id}acc.csv"),
                    index=False
                )

            ####################################################################
            #  2) Evaluate Shape Bias
            ####################################################################
            if EVALUATE_BIAS:
                model_type_for_eval = model_type + '_ours'#'resnet50_ours' #* can be replace by model_type + '_ours'
                evaluation = True
                load_checkpoint_path = current_model_path  # Use the same path as above
                dataset_trained_on = DATASET_TRAINED_ON
                data_normalization_type = "0-1"  # or "mean-std"
                matrixplot_savedir = None


                shape_bias, shape_bias_across_classes, classes, shape_accs, texture_accs = get_shape_bias(
                    model_type_for_eval,
                    evaluation=evaluation,
                    load_checkpoint_path=load_checkpoint_path,
                    dataset_trained_on=dataset_trained_on,
                    image_size=IMAGE_SIZE,
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
                    'median_shape_bias': np.median(shape_bias_across_classes),
                    'timepoint': 0,
                    # 'texture_acc': texture_accs,
                    # 'shape_acc': shape_accs,
                    # 'shape_bias_per_class': shape_bias_across_classes,
                }
                if EVALUATE_ACC:
                    new_row['top1'] = float(top1)
                    new_row['top5'] = float(top5)

                # shape_bias_across_time_df = shape_bias_across_time_df.append( new_row, ignore_index=True)
                shape_bias_across_time_df = pd.concat([shape_bias_across_time_df, pd.DataFrame([new_row])], ignore_index=True)
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

            #* 3) visulize the importance of features
            if FEATURE_ATRRIBUTION_VISUALIZE:
                model.to('cuda' if torch.cuda.is_available() else 'cpu')
                models_dict_for_feature_atrribution[model_name + f"_ep_{epoch}"] = model
                

            ################################################################################
            #  5) ADVERSARIAL ATTACK EVALUATION (OPTIONAL ROBUSTNESS TEST)
            ################################################################################

            # If you wish to evaluate adversarial robustness, set EVALUATE_ADV_ROBUSTNESS = 1.
            # This code integrates seamlessly with the main script above. It assumes that
            # the model is already loaded and named `model`, and that a test DataLoader is
            # named `test_loader`. The rest is contained within these utility functions.

            if EVALUATE_ADV_ROBUSTNESS:
                
                selected_attacks = SELECTED_ADV_ATTACKS_NAMES #{k: v for k, v in ATTACKS.items() if k in SELECTED_ADV_ATTACKS_NAMES}
                
                evaluator = AdversarialRobustnessEvaluator(test_loader=test_loader)
                adv_attack_results_df = evaluator.evaluate(model, attack_subset= selected_attacks, model_name= model_name, epoch=epoch, log_dir=ADV_ROBUSTNESS_RESULTS_SAVE_DIR,)
                print(f"adv_attack_results_df:\n{adv_attack_results_df}")

                # adv_attack_df = adv_attack_analysis(model, test_loader, selected_attacks, EPSILONS_DICT, 
                #                     model_name=model_name, epoch=epoch, device='cuda' if torch.cuda.is_available() else 'cpu')
                # print(f"adv_attack_df:\n{adv_attack_df}")

            if EXTRACT_ACTS:
                # EXTRACT_ACTS_DATASETS = "coco-split515"
                # import pdb;pdb.set_trace()
                rdms = extract_acts_and_create_rdm(
                        model,
                        test_acts_loader,
                        layers=['conv1', 'layer1', 'layer2', 'layer3', 'layer4'], #* Default layers for ResNet50
                        device='cuda',
                        distance_metric = 'euclidean',  # or  'correlation' 
                )
                    
                # ave each layer’s RDM
                for layer_name, rdm in rdms.items():
                    save_path = os.path.join(ACTS_RDM_SAVE_DIR, f'{model_name}_{layer_name}_rdm.npy')
                    np.save(save_path, rdm)
                    print(f"[INFO] Saved RDM for {model_name}' {layer_name} → {save_path}")
                
                # import pdb;pdb.set_trace()
            
            if EVALUATE_DEGRADATION_ROBUSTNESS:
                #* Degradation robustness
                evaluator = DegradationRobustnessEvaluator(
                    test_loader=test_loader,
                    blur_loader_fn= get_blur_loader,# get_blur_loader, get_distortion_loader
                    distortion_loader_fn= get_distortion_loader,
                    candidate_ids=None,     # optional
                )

                criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
                
                if EVALUATE_VARIOUS_DEGRADTION:
                    clean_result = evaluator.evaluate_clean(test_loader, model, criterion, epoch,  image_size = IMAGE_SIZE)
                    df_clean = evaluator.results_to_dataframe(clean_result, model_name, metric = "accuracy",)
                    distortions_results = evaluator.evaluate_distortions(model, args, criterion, epoch,  image_size = IMAGE_SIZE)
                    df_distortions_results = evaluator.results_to_dataframe(distortions_results, model_name, metric = "accuracy",)

                    # Merge clean and distortion results
                    df_merged = pd.concat([df_clean, df_distortions_results], ignore_index=True)
                    # Save merged DataFrame to CSV
                    image_degradtion_csv_path = os.path.join(DEGRADATION_RESULTS_SAVE_DIR, f"{model_name}_epoch{epoch}.csv")
                    df_merged.to_csv(image_degradtion_csv_path, index=False)
                    print(f"IMAGE DEGRADTION Results saved to {image_degradtion_csv_path}")

                if EVALUATE_BLUR_DEGRADATIONN:
                    clean_result = evaluator.evaluate_clean(test_loader, model, criterion, epoch,  image_size = IMAGE_SIZE, eval_subset_classes= True, test_type='face' if 'face' in args.dataset_name else 'object')
                    df_clean = evaluator.results_to_dataframe(clean_result, model_name, metric = "accuracy",)
                    blur_results = evaluator.evaluate_blur( model, args, criterion, epoch,  image_size = IMAGE_SIZE, eval_subset_classes = True, test_type='face' if 'face' in args.dataset_name else 'object')
                    df_blur = evaluator.results_to_dataframe(blur_results, model_name, metric = "accuracy",)
                    
                    df_blur_merged = pd.concat([df_clean, df_blur], ignore_index=True)
                    blur_degradtion_csv_path = os.path.join(DEGRADATION_RESULTS_SAVE_DIR, 'blur_degradtion', f"{args.dataset_name}_blur_degradtion_{model_name}_epoch{epoch}.csv")
                    os.makedirs(os.path.join(DEGRADATION_RESULTS_SAVE_DIR, 'blur_degradtion'), exist_ok=True)
                    df_blur_merged.to_csv(blur_degradtion_csv_path, index=False)
                
                # import pdb;pdb.set_trace()
                # blur_results       = evaluator.evaluate_blur(model, args, criterion, epoch,  image_size = IMAGE_SIZE) #! need update

                
                
        except Exception as e:
            print(f"Skipping Model {model_name}, Epoch {epoch} due to error: {e}")
            pass


if FEATURE_ATRRIBUTION_VISUALIZE:
    
    #*After getting models above
    # Suppose you have cue-conflict images: 
    # Define the directory
    # chossen_cat = 'None'
    categories = [
                    "airplane", "bear", "bicycle", "bird", "boat", "bottle", "car", "cat",
                    "chair", "clock", "dog", "elephant", "keyboard", "knife", "oven", "truck"
                ][:1]
    for chossen_cat in categories:
        visualzie_directory = f"/home/student/l/lzejin/codebase/P001_dvd_gpus/data/cue-conflict/{chossen_cat}"
        # List all image paths
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
        visualize_image_paths = []
        for root, _, files in os.walk(visualzie_directory):
            for file in files:
                if os.path.splitext(file)[1].lower() in image_extensions:
                    visualize_image_paths.append(os.path.join(root, file))

        # visualize_image_paths = [
        #     "/home/student/l/lzejin/codebase/P001_dvd_gpus/data/cue-conflict/cat/cat7-elephant1.png",
        #     "/home/student/l/lzejin/codebase/P001_dvd_gpus/data/cue-conflict/elephant/elephant6-bottle2.png",
        # ]

        #* Call the comparison function
        layer = 'layer4'
        visualize_lrp_comparison(
            model_dict=models_dict_for_feature_atrribution,
            image_paths=visualize_image_paths,
            image_size=256,
            layer=layer,   # The last conv block. Could also do 'fc' or others.
            save_path = f"./results/plots/visualize_features/{layer}/{chossen_cat}/lrp_comparison.pdf",
        )



################################################################################
#  OPTIONAL: PLOT SHAPE BIAS ACROSS EPOCHS
################################################################################

if PLOT_SHAPE_BIAS_ACROSS_EPOCH:
    # Example usage; update paths as needed
    csv_file_path = (
        '/home/student/l/lzejin/codebase/P001_dvd_gpus/'
        'results/shape_bias/ecoset_across_time.csv'
    )
    output_plot_path = (
        f'/home/student/l/lzejin/codebase/P001_dvd_gpus/results/plots/shape_bias/{formatted_date}_shape_bias_across_time_plot_with_human.pdf'
    )
    shape_texture_acc_output_plot_path = (
        f'/home/student/l/lzejin/codebase/P001_dvd_gpus/results/plots/shape_bias/{formatted_date}_shape_acc_texture_acc_across_time_plot_updated.pdf'
    )
    plot_shape_bias_across_time(csv_file_path,  colors = [ 'black', np.array([49, 162, 142]) / 255,], save_path=output_plot_path)
    plot_shape_and_texture_accuracy_across_time(csv_file_path,  colors = [ 'black', np.array([49, 162, 142]) / 255,], save_path=shape_texture_acc_output_plot_path)

if PLOT_ADV_ROBUSTNESS:
    from dvd.analysis.plot import curve_plot, bar_plot, plot_adv_attack_results_comparison, plot_adv_attack_results_comparison_multi

    # Load the adversarial attack results from the CSV file
    adv_attack_results_csv = f'results/adv_robustness/adv_attacks_all_results.csv'
    try:
        adv_attack_df = pd.read_csv(adv_attack_results_csv)
    except Exception as e:
        raise IOError(f"Unable to read CSV file at {adv_attack_results_csv}. Error: {e}")

    # Loop over each selected adversarial attack and plot the results
    # for attack_name in SELECTED_ADV_ATTACKS_NAMES:
    #     print(f"{attack_name} plotting")
    #     # plot_adv_attack_results_comparison
    #     plot_adv_attack_results_comparison(attack_name, adv_attack_df,
    #                          save_dir = plot_dir + '/adv_attacks', figsize=(3.54*2/3, 2))

    # plot togethear
    plot_adv_attack_results_comparison_multi(SELECTED_ADV_ATTACKS_NAMES, adv_attack_df,
                             save_dir = plot_dir + '/adv_attacks', figsize=(3.54*2, 2))

if PLOT_DEGRADATION_ROBUSTNESS:
    from dvd.analysis.plot import plot_degradation_grid
    # Load the degradation robustness results from the CSV file
    degradation_results_csv = f'results/degradation_robustness/degradation_robustness_all_results.csv'
    try:
        degradation_df = pd.read_csv(degradation_results_csv)
    except Exception as e:
        raise IOError(f"Unable to read CSV file at {degradation_results_csv}. Error: {e}")

    # Loop over each selected degradation type and plot the results
    # for degradation_type in DEGRADATION_TYPES:
    print(f"degradation robustness plotting")
    # import pdb;pdb.set_trace()
    plot_degradation_grid(degradation_df,
                            save_dir = plot_dir + '/degradation_robustness', 
                            save_name = "degradation_robustness_grid_v7_all.pdf", figsize=(3.54, 4),
                            relative=False)