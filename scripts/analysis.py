import os
import pickle
import numpy as np
import pandas as pd
from itertools import product
import argparse
import torch
import torch.nn as nn

from neuroai.bias_test import evaluate_shape_bias

import main
from evd.analysis.save import save_shape_biases_across_time_csv, save_shape_biases_across_categories_csv
from evd.models.eval import validate
import evd.utils
from evd.datasets.dataset_loader import get_test_loaders



############################################################################################################
## Set options for the analysis
############################################################################################################
EVALUATE_ACC = 1
EVALUATE_BIAS = 1
PLOT_SHAPE_BIAS_ACROSS_EPOCH = 1
EVALUATE_ACC_ROBUSTNESS = 0
PLOT_ACC_ROBUSTNESS = 0


############################################################################################################
## Set the parameters for the shape bias evaluation
############################################################################################################
EPOCHS = ['best'] #* '' means use the path directly, otherwise use the epoch number or 'best'
analysis_id = f'9th_Sun_{EPOCHS[0]}_extra_'#'27th_Thu1_Feb_' #!
save_dir_name = 'blur_sharpness'
plot_dir = f'./results/plots/'
dataset_id = 1
DATASET_TRAINED_ON =['texture2shape_miniecoset','ecoset_square256', 'imagenet_16','facescrub'][dataset_id]
DATASET_TRAINED_ON_Path = ['/home/student/l/lzejin/datasets/', '/home/student/l/lzejin/datasets/',
                        '/home/student/l/lzejin/codebase/All-TNNs/texture2shape_projects/generate_facescrub_dataset/'][dataset_id] 
NUM_CLASSES = [112,565,16,118][dataset_id]
DATASET_ANLYSIS_ON = ['cue-conflict'][:] #* Evaluation
IMAGE_SIZE = 149
model_type = ['resnet50', 'blt_vs'][0]

# Hyperparameters for EVD
# months_per_epochs = [2.0, 1.0, 4.0, 3.0, 0.5, 8.0, 0.25][:0]
months_per_epochs = [2.0, 1.0, 3.0, 4.0, 8.0,  0.5,  0.25][:0]
contrast_thresh = [0.2, 0.1, 0.4][:0] # Alpha for contrast
contrast_spd    = [100.0, 50.0, 150.0][:0] # Beta for contrast 

# Option 1: add model name to path mannually
MODEL_NAME2PATH = {
    # mini ecoset
    # 'adult':f"/home/student/l/lzejin/codebase/All-TNNs/evd_gpus/logs/resnet50_mpe2.0_alpha0.2_dn100.0_texture2shape_miniecoset256_0.0001_dev_adult_b1c1cs1_T_normal_seed_1/weights/checkpoint_{EPOCHS[0]}.pth",
    # 'full_ecoset_evd_mpe1_alpha0.1_dn100':f'/home/student/l/lzejin/codebase/All-TNNs/evd_gpus/logs/resnet50_mpe1.0_alpha0.1_dn100.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1/weights/checkpoint_149.pth',#_{EPOCHS[0]}.pth',
    # 'full_ecoset_evd_mpe1_alpha0.1_dn50':f'/home/student/l/lzejin/codebase/All-TNNs/evd_gpus/logs/resnet50_mpe1.0_alpha0.1_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1/weights/checkpoint_149.pth',#_{EPOCHS[0]}.pth',
    # 'full_ecoset_evd_mpe2_alpha0.1_dn100' :f"/home/student/l/lzejin/codebase/All-TNNs/evd_gpus/logs/resnet50_mpe2.0_alpha0.1_dn100.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1/weights/checkpoint_{EPOCHS[0]}.pth",
    # 'full_ecoset_evd_mpe2_alpha0.2_dn100':f'/home/student/l/lzejin/codebase/All-TNNs/evd_gpus/logs/resnet50_mpe2.0_alpha0.2_dn100.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1/weights/checkpoint_124.pth',#_{EPOCHS[0]}.pth',
    # 'full_ecoset_evd_mpe2_alpha0.2_dn50':f'/home/student/l/lzejin/codebase/All-TNNs/evd_gpus/logs/resnet50_mpe2.0_alpha0.2_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1/weights/checkpoint_124.pth',#_{EPOCHS[0]}.pth',
    # # 'full_ecoset_evd_mpe2_alpha0.2_dn150':f'/home/student/l/lzejin/codebase/All-TNNs/evd_gpus/logs/resnet50_mpe2.0_alpha0.2_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1/weights/checkpoint_{EPOCHS[0]}.pth',
    # # 'full_ecoset_evd_mpe2_alpha0.2_dn50_lr_5e-5' : f'/home/student/l/lzejin/codebase/All-TNNs/evd_gpus/logs/resnet50_mpe2.0_alpha0.2_dn50.0_ecoset_square256256_5e-05_dev_evd_b1c1cs1_T_normal_seed_1/weights/checkpoint_{EPOCHS[0]}.pth',
    # 'full_ecoset_evd_mpe3_alpha0.4_dn150':f'/home/student/l/lzejin/codebase/All-TNNs/evd_gpus/logs/resnet50_mpe3.0_alpha0.4_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1/weights/checkpoint_139.pth',#_{EPOCHS[0]}.pth',
    # 'full_ecoset_evd_mpe4_alpha0.4_dn150':f'/home/student/l/lzejin/codebase/All-TNNs/evd_gpus/logs/resnet50_mpe4.0_alpha0.4_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1/weights/checkpoint_{EPOCHS[0]}.pth',
    # 'full_ecoset_evd_mpe4_alpha0.4_dn50':f'/home/student/l/lzejin/codebase/All-TNNs/evd_gpus/logs/resnet50_mpe4.0_alpha0.4_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1/weights/checkpoint_{EPOCHS[0]}.pth',
    # 'full_ecoset_evd_mpe8_alpha0.4_dn150':f'/home/student/l/lzejin/codebase/All-TNNs/evd_gpus/logs/resnet50_mpe8.0_alpha0.4_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1/weights/checkpoint_{EPOCHS[0]}.pth',
    # 'full_ecoset_adult' :f"/home/student/l/lzejin/codebase/All-TNNs/evd_gpus/logs/resnet50_mpe2.0_alpha0.2_dn100.0_ecoset_square256256_0.0001_dev_adult_b1c1cs1_T_normal_seed_1/weights/checkpoint_{EPOCHS[0]}.pth",

    # 'full_ecoset_mpe3.0_alpha0.4_dn150':'resnet50_mpe3.0_alpha0.4_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'full_ecoset_mpe2.0_alpha0.2_dn50': 'resnet50_mpe2.0_alpha0.2_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    #! adult 63% + 0.45 shape bias, good acc model 63%+ 0.7 shape bias (boom), good trade-off 57-60% acc 0.8 shape bias, good shape bias model 53%-ish acc 0.88+ shape bias
    #TODO high acc models looks fine (mpe 4 -8 -3), mpe 2 dn 300 & 100 looks fine (somehow test is 3-5% lower than val), 150 need further train, dn50 might need retrain, mpe1 alpha 0.05-dn50 & alpha 0.1-dn100 on trainning, while alpha 0.1-dn50 need retrain (dn100 wait to see)
    #* p1 full_ecoset_evd_mpe2_alpha0.2_dn50, full_ecoset_evd_mpe1_alpha0.1_dn50
    # p2 full_ecoset_evd_mpe2_alpha0.1_dn100, full_ecoset_evd_mpe1_alpha0.05_dn50
    # p3 full_ecoset_evd_mpe2_alpha0.2_dn300, full_ecoset_evd_mpe1_alpha0.05 or 0.1 _dn300, so for mpe 4/8 alpha 0.4 dn 300
    # p4 mpe8 alpha 0.8 dn50 (higher acc higher shape bias)
    #TODO then dn300 seems fine? accuracy is even higher by without mapping in full ecoset, 
    #* fixed epoch 1 error | ideally shpae bias should be higher
    #! so retrain the model mpe2 alpha0.1 dn100 | 'full_ecoset_mpe1.0_alpha0.1_dn50', 'full_ecoset_mpe2.0_alpha0.1_dn100', 'full_ecoset_mpe8.0_alpha0.4_dn150', 'full_ecoset_mpe8.0_alpha0.8_dn150'
    # 'full_ecoset_mpe8.0_alpha0.4_dn150': 'resnet50_mpe8.0_alpha0.4_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'full_ecoset_mpe4.0_alpha0.4_dn50': 'resnet50_mpe4.0_alpha0.4_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'full_ecoset_mpe4.0_alpha0.4_dn150': 'resnet50_mpe4.0_alpha0.4_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'full_ecoset_mpe3.0_alpha0.4_dn50':'resnet50_mpe3.0_alpha0.4_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'full_ecoset_mpe2.0_alpha0.2_dn100':'resnet50_mpe2.0_alpha0.2_dn100.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'full_ecoset_mpe2.0_alpha0.2_dn300': 'resnet50_mpe2.0_alpha0.2_dn300.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'full_ecoset_mpe1.0_alpha0.1_dn50':'resnet50_mpe1.0_alpha0.1_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'full_ecoset_mpe1.0_alpha0.1_dn100':'resnet50_mpe1.0_alpha0.1_dn100.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    # 'full_ecoset_adult' : 'resnet50_mpe2.0_alpha0.2_dn100.0_ecoset_square256256_0.0001_dev_adult_b1c1cs1_T_normal_seed_1',
    }
# Convert into REAL PATH
MODEL_NAME2PATH = {model_name: f"/home/student/l/lzejin/codebase/All-TNNs/evd_gpus/logs/{model_path}/weights/checkpoint_{EPOCHS[0]}.pth" for model_name, model_path in MODEL_NAME2PATH.items()}

# Option 2: tarverse hyperparameters to include trade-off models
for mpe, alpha, beta in product(months_per_epochs, contrast_thresh, contrast_spd):
    MODEL_NAME2PATH[f'mpe{mpe}_alpha{alpha}_dn{beta}'] = f"/home/student/l/lzejin/codebase/All-TNNs/evd_gpus/logs/{model_type}_mpe{mpe}_alpha{alpha}_dn{beta}_{DATASET_TRAINED_ON}256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1/weights/checkpoint_{EPOCHS[0]}.pth"


############################################################################################################
## Evaluate shape bias of models across time and categories for each epoch
############################################################################################################

#Create acc df

acc_df = pd.DataFrame(columns=['model_name', 'epoch', 'top1', 'top5'])
shape_bias_across_time_df = pd.DataFrame(columns=['model_name', 'epoch', 'top1', 'top5', 'shape_bias', 'timepoint'])
acc_save_dir = f'./results/acc/'
shape_bais_save_dir =  f'./results/shape_bias/'

# preload
if EVALUATE_ACC:
    # 1) config commonly shared load
    config_file = '/home/student/l/lzejin/codebase/All-TNNs/evd_gpus/evd/models/config/config.yaml'
    config_dict = evd.utils.load_config(config_file)
    args = argparse.Namespace(**config_dict)
    args.dataset_name = DATASET_TRAINED_ON 

    # 2) Get data loaders
    test_loader = get_test_loaders(args) # Need to add init_distributed=True

# Traverse the models
for model_name, model_path in MODEL_NAME2PATH.items():
    for epoch in EPOCHS:
        
        #* 1 Evaluate accuracy
        if EVALUATE_ACC:
            # 1) update config 
            # config_file = '/home/student/l/lzejin/codebase/All-TNNs/evd_gpus/evd/models/config/config.yaml'
            # config_dict = evd.utils.load_config(config_file)
            # args = argparse.Namespace(**config_dict)
            # args.dataset_name = DATASET_TRAINED_ON #* Confirm the dataset name
            args.arch = model_type #* Confirm the model type

            # 2) Create and setup model
            model, linear_keyword = evd.models.loader.create_model(args)

            # 4) Load checkpoint
            #* if log_dir not None, then use the default best model (based on args, so need overwrite mpealpha beta)
            evd.models.loader.load_checkpoint( model, model_path=model_path, log_dir=None, args=None) 

            # 5) Evaluate the model on the test set
            # test_loader = get_test_loaders(args) # Need to add init_distributed=True

            criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
            top1, top5 = evd.models.eval.validate(test_loader, model, criterion, epoch)

            # 6) Save the results in to a csv file
            
            os.makedirs(acc_save_dir, exist_ok=True)
            acc_df = acc_df.append({'model_name': model_name, 'epoch': epoch, 'top1': float(top1), 'top5': float(top5)}, ignore_index=True)
            acc_df.to_csv(os.path.join(acc_save_dir, f"{analysis_id}acc.csv"), index=False)
            # import pdb;pdb.set_trace()

        if EVALUATE_BIAS:
            results_dict = evaluate_shape_bias(
                model_name= model_type,
                save_name = save_dir_name,
                model_path = model_path,
                epoch= epoch, #['best', 149, 99, 129, 144, 154][-1], # best or num | 144 get 0.78
                dataset_name= DATASET_TRAINED_ON,
                save_dir = os.path.join(shape_bais_save_dir, f"{analysis_id}{model_name}/"),
                evaluation_datasets=DATASET_ANLYSIS_ON,
                use_best=False,
                plot=False,
            )

            shape_biases_across_timesteps, shape_bias_cat_dict, shape_correct_dict, texture_correct_dict = results_dict[save_dir_name][model_type][epoch]
            print(f"shape bias across time: {results_dict[save_dir_name][model_type][epoch][0]}")

            os.makedirs(shape_bais_save_dir, exist_ok=True)
            shape_bias_across_time_df = shape_bias_across_time_df.append({'model_name': model_name, 'epoch': epoch, 'top1': float(top1), 'top5': float(top5), 'shape_bias': shape_biases_across_timesteps, 'timepoint': np.arange(0, len(shape_biases_across_timesteps))}, ignore_index=True)
            shape_bias_across_time_df.to_csv(os.path.join(shape_bais_save_dir, f"{analysis_id}shape_bais.csv"), index=False)
            # 
            #TODO the results slightly different from neuroai package, need to recheck
            #TODO save gehrios and our shape value 
            #* Save shape_biases_across_timesteps as csv table
            os.makedirs(os.path.join(acc_save_dir, f"{analysis_id}{model_name}/"), exist_ok=True)
            save_shape_biases_across_time_csv(shape_biases_across_timesteps, epoch, filename=os.path.join(acc_save_dir, f"{analysis_id}{model_name}/shape_biases_across_timesteps.csv"))
            #* Save shape_bias_cat_dict, shape_correct_dict, texture_correct_dict into one table
            save_shape_biases_across_categories_csv(shape_bias_cat_dict, shape_correct_dict, texture_correct_dict, epoch, filename=os.path.join(acc_save_dir, f"{analysis_id}{model_name}/shape_biases_across_categories.csv"))


        #*** And so on ...

        #TODO implement the analyese first