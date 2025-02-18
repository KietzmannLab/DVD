import os
import pickle
import numpy as np
import pandas as pd

from neuroai.bias_test import evaluate_shape_bias
from evd.utils.save import save_shape_biases_across_time_csv, save_shape_biases_across_categories_csv


############################################################################################################
## Set options for the analysis
############################################################################################################
EVALUATE_BIAS = [False, True][1]
PLOT_SHAPE_BIAS_ACROSS_EPOCH = [False, True][1]
EVALUATE_ACC_ROBUSTNESS = [False, True][0]
PLOT_ACC_ROBUSTNESS = [False, True][0]

PREPROCESSING_INFANTS2ADULT_EEG = [False, True][0]
ANALYSIS_INFANTS2ADULT_EEG_ANN_FUSION = [False, True][0]
PREPROCESSING_INFANTS2ADULT_fMRI = [False, True][0]
ANALYSIS_INFANTS2ADULT_fMRI_ANN_FUSION = [False, True][0]


############################################################################################################
## Set the parameters for the shape bias evaluation
############################################################################################################
analysis_id = '18th_feb_' #!
save_dir_name = 'blur_sharpness'
save_dir =  f'./results/shape_bias/'
DATASET_TRAINED_ON =['texture2shape_miniecoset','ecoset_square256'][1]
DATASET_ANLYSIS_ON = ['cue-conflict'][:]
EPOCHS = [''] #* '' means use the path directly, otherwise use the epoch number or 'best'
model_type = ['resnet50', 'blt_vs'][0]
MODEL_NAME2PATH = {
        # 'evd_dn100': '/home/student/l/lzejin/codebase/All-TNNs/evd/results/shared/for_alex/resnet50_evd.pth',
        'evd_dn150': '/home/student/l/lzejin/codebase/All-TNNs/EVD/logs/net_params/blur_sharpnessFeb_resnet50_linear_no_beta_dn150.0_2.0_mixgray1.0_mpe2.0_alpha0.2beta1.0_id_1_lr_0.0001_none_ecoset_square256_dev_retina_fft_development_leReLU0.01_colorspd1_egray0_cs_age50_57.6_T__rm1000.0_seed1/blur_sharpnessFeb_resnet50_linear_no_beta_dn150.0_2.0_mixgray1.0_mpe2.0_alpha0.2beta1.0_id_1_lr_0.0001_none_ecoset_square256_dev_retina_fft_development_leReLU0.01_colorspd1_egray0_cs_age50_57.6_T__rm1000.0_seed1_epoch_154.pth', #  ep149 [0.7964939024390244] but neuroai is lower
        'adult': '/home/student/l/lzejin/codebase/All-TNNs/evd/results/shared/for_alex/resnet50_adult.pth',
    }


############################################################################################################
## Evaluate shape bias of models across time and categories for each epoch
############################################################################################################
if EVALUATE_BIAS:
    for model_name, model_path in MODEL_NAME2PATH.items():
        for epoch in EPOCHS:
            results_dict = evaluate_shape_bias(
                model_name= model_type,
                save_name = save_dir_name,
                model_path = model_path,
                epoch= epoch, #['best', 149, 99, 129, 144, 154][-1], # best or num | 144 get 0.78
                dataset_name= DATASET_TRAINED_ON,
                save_dir = os.path.join(save_dir, f"{analysis_id}{model_name}/"),
                evaluation_datasets=DATASET_ANLYSIS_ON,
                use_best=False,
                plot=True
            )

            shape_biases_across_timesteps, shape_bias_cat_dict, shape_correct_dict, texture_correct_dict = results_dict[save_dir_name][model_type][epoch]
            print(f"shape bias across time: {results_dict[save_dir_name][model_type][epoch][0]}")

            #TODO the results slightly different from neuroai package, need to recheck
            #* Save shape_biases_across_timesteps as csv table
            save_shape_biases_across_time_csv(shape_biases_across_timesteps, epoch, filename=os.path.join(save_dir, f"{analysis_id}{model_name}/shape_biases_across_timesteps.csv"))
            #* Save shape_bias_cat_dict, shape_correct_dict, texture_correct_dict into one table
            save_shape_biases_across_categories_csv(shape_bias_cat_dict, shape_correct_dict, texture_correct_dict, epoch, filename=os.path.join(save_dir, f"{analysis_id}{model_name}/shape_biases_across_categories.csv"))
            
            # import pdb; pdb.set_trace()
            #* Save shape_biases_across_timesteps as pickle
            # save_pickle_dir = os.path.join(save_dir, f"{analysis_id}{model_name}/shape_bias.pickle"),
            # pickle.dump(results_dict, open(save_pickle_dir, 'wb+'))






