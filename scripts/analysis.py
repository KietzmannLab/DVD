# import os
# import pickle
# import datetime
# import numpy as np
# import pandas as pd
# from itertools import product
# import argparse
# import torch
# import torch.nn as nn

# from neuroai.evaluation.shape_bias_eval import get_shape_bias

# # import main  # Note: If unused, you can safely remove this import.
# from dvd.analysis.save import (
#     save_shape_biases_across_time_csv,
#     save_shape_biases_across_categories_csv
# )
# from dvd.models.eval import validate
# from dvd.models.extract_acts import extract_acts_and_create_rdm
# import dvd.utils
# from dvd.datasets.dataset_loader import get_test_loaders, get_blur_loader, get_distortion_loader
# from dvd.analysis.plot import plot_shape_bias_across_time, plot_shape_and_texture_accuracy_across_time
# from dvd.analysis.adv_attack import AdversarialRobustnessEvaluator #, ATTACKS, EPSILONS_DICT
# from dvd.analysis.feature_atrribution_visualize import visualize_lrp_comparison
# from dvd.analysis.degradation_robustness import DegradationRobustnessEvaluator


# ################################################################################
# #                           ANALYSIS CONFIGURATIONS
# ################################################################################
# #* Figure 1 is an illustration plot
# #* To plot development trajectory of different visual aspects, check dvd.dvd.development.EarlyVisualDevelopmentTransformer

# #* Figure 2 ACC & Shape bias
# # Figure 1A just cue-conflicting images
# # Fig 2C: to plot multiple models' shape bias in a benchmark, use code in another repo (develop base on Geirhos code):https://github.com/KietzmannLab/Neuro-vs-AI/blob/main/scripts/evaluate.py 
# EVALUATE_ACC = 1              # Fig 2B: Evaluate standard accuracy
# EVALUATE_BIAS = 1             # Fig 2B: Evaluate shape bias

# #* Figure 3 | generalisation across vision datasets and ANN architectures.
# # All shape bias evaluation mainly based on Geirhos code
# # Just check code in another repo (develop base on Geirhos code):https://github.com/KietzmannLab/Neuro-vs-AI/blob/main/scripts/evaluate.py 

# #* Figure 4 | In-depth investigation of the emergence of shape bias & feature atrribution visulisation
# PLOT_SHAPE_BIAS_ACROSS_EPOCH = 0  # Figure 4A | Generate shape bias plot across epochs
# FEATURE_ATRRIBUTION_VISUALIZE = 0 # Figure 4B feature atrribution visulisation
# models_dict_for_feature_atrribution = {} # initial model_dict for  FEATURE_ATRRIBUTION_VISUALIZE
# # Figure 4C control rearing, also check code in another repo (develop base on Geirhos code):https://github.com/KietzmannLab/Neuro-vs-AI/blob/main/scripts/evaluate.py 

# #* Figure 5 abstract shape recogntion
# #* Fig 5B just an illustration plot
# #* Fig 5B plot benchmark: just check scripts/abstract_shape_analysis_scripts/plot_illusion_benchmark.py
# #* Fig 5C t-SNE visualization: just check scripts/abstract_shape_analysis_scripts/tsne_visualization_abstract_shape.py

# #* Figure 6 
# #* Figure 6 A & B | Degradation robustness
# EVALUATE_DEGRADATION_ROBUSTNESS = 0         # Fig2a: Evaluate degradation #* see shape texture project
# PLOT_DEGRADATION_ROBUSTNESS = 0                 # Fig2a: Plot degradation results 
# EVALUATE_VARIOUS_DEGRADTION = 0 #* evalute various degradations
# EVALUATE_BLUR_DEGRADATIONN = 0 #* evalute blur degradation, need to set face or objects 
# DEGRADATION_RESULTS_SAVE_DIR =  './results/degradation_robustness/' 
# os.makedirs(DEGRADATION_RESULTS_SAVE_DIR, exist_ok=True)
# #* Figure 6C | adv robustness
# EVALUATE_ADV_ROBUSTNESS = 0     #  Evaluate adv robustness in SELECTED_ADV_ATTACKS_NAMES
# PLOT_ADV_ROBUSTNESS = 0          # Plot  adv robustness
# ADV_ROBUSTNESS_RESULTS_SAVE_DIR =  './results/adv_robustness/' 
# SELECTED_ADV_ATTACKS_NAMES = [
#                     'L2AdditiveGaussianNoiseAttack','L2AdditiveUniformNoiseAttack', 'SaltAndPepperNoiseAttack',
#                     "FGSM", "FGM", "PGD",  ] #* Blackbox /decision‑based adversarial attacks (Noise-Based Attacks) & # white‑box attacks

# #* extract activations for RDMs
# EXTRACT_ACTS = 0
# EXTRACT_ACTS_DATASETS = ["coco-split515", "skeleton_ayzenberg"][1]
# ACTS_RDM_SAVE_DIR =  './results/acts/rdms/skeleton_ayzenberg/' 
# os.makedirs(ACTS_RDM_SAVE_DIR, exist_ok=True)

# # Mode for shape bias evaluation: 'single' uses a static set of epochs,
# # 'traverse' scans through a range of epochs.
# MODE = ['single', "traverse"][0]  # 0 = 'single', 1 = 'traverse'
# if MODE == 'single':
#     EPOCHS_list = ['last', 'best'][1:]  # Example: pick 'best'
#     EPOCHS = EPOCHS_list
# else:
#     # EPOCHS = list(range(4, 300, 5))  # Traverse from epoch 4 to 145 in steps of 5
#     EPOCHS =  list(range(4, 300, 5))  #list(range(4, 150, 5))  # Traverse from epoch 4 to 145 in steps of 5


# # Dataset / model training identifiers
# dataset_id = 1 # 0=texture2shape_miniecoset, 1=ecoset, 2=imagenet, 3=imagenet_16, 4=facescrub
# DATASET_TRAINED_ON = ['texture2shape_miniecoset', 'ecoset_square256', 'imagenet','imagenet_16', 'facescrub'][dataset_id]
# IMAGE_SIZE = [256,224,None][0] #! Default set as 256 x 256

# # Choose model architecture by index
# model_type = [ 'all', 'resnet50', 'resnet101', 'deit_base_patch16_224','vit_b_16', 'swin_b', 'alexnet', 'vgg16','resnet101','resnet152','resnext50_32x4d','blt_vs'
#                 ][1]  # i.e., 'resnet50'
# # Hyperparameters for EVD
# months_per_epochs = [1.0, 2.0, 4.0, 8.0, 3.0, 0.5,  0.25][:0]
# contrast_thresh = [ 0.1, 0.2, 0.4, 0.8, 0.05][:0] # Alpha for contrast
# contrast_spd    = [50.0,100.0, 150.0, 300.0][:0] # Beta for contrast 

# #! Setting Saving name for current analysis
# today = datetime.date.today() # Get the current date
# formatted_date = f"{today.day}th_{today.strftime('%B')}"# Format day and month: e.g., "8th_April"
# analysis_id = f'{formatted_date}_{DATASET_TRAINED_ON}_across_architecture_{MODE}_{EPOCHS[0]}_'  #! Prefix for output files

# acc_save_dir = './results/acc/'
# shape_bias_save_dir = './results/shape_bias/' 
# shape_bias_per_class_save_dir = './results/shape_bias/shape_bias_per_class' 
# plot_dir = './results/plots/'
# os.makedirs(acc_save_dir, exist_ok=True)
# os.makedirs(shape_bias_per_class_save_dir, exist_ok=True)
# os.makedirs(plot_dir, exist_ok=True)



# ################################################################################
# #                     MODEL NAME -> CHECKPOINT PATH MAPPINGS
# ################################################################################
# # Option 1: add model name to path mannually

# MODEL_NAME2PATH = {

#     # cirtical period
#     'DVD-B_random_order': 'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_random_seed_2',
#     'DVD-B_fully_random': 'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_fully_random_seed_2',
#     'DVD-B_mid_phase': 'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_mid_phase_seed_2',

#     }
# # Convert into REAL PATH
# MODEL_NAME2PATH = {model_name: f"/share/klab/lzejin/lzejin/codebase//P001_dvd_gpus/logs/{model_path}/weights/checkpoint_{EPOCHS[0]}.pth" if model_path else None for model_name, model_path in MODEL_NAME2PATH.items()}
# # MODEL_NAME2PATH = {model_name: f"checkpoints/{model_path}/model_{EPOCHS[0]}.pth" if model_path else None for model_name, model_path in MODEL_NAME2PATH.items()} #* SimCLR

# # Option 2: tarverse hyperparameters to include trade-off models
# for mpe, alpha, beta in product(months_per_epochs, contrast_thresh, contrast_spd):
#     model_save_name = f'mpe{mpe}_alpha{alpha}_dn{beta}' #* Setting name for models
    
#     model_subdir = (
#         f"{model_type}_mpe{mpe}_alpha{alpha}_dn{beta}_"
#         f"{DATASET_TRAINED_ON}256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1"
#     )

#     MODEL_NAME2PATH[model_save_name] = f"logs/{model_subdir}/weights/checkpoint_{EPOCHS[0]}.pth"


# ################################################################################
# #           DATAFRAMES FOR STORING ACCURACY AND SHAPE BIAS RESULTS
# ################################################################################

# acc_df = pd.DataFrame(columns=['model_name', 'epoch', 'top1', 'top5'])
# shape_bias_across_time_df = pd.DataFrame(
#     columns=['model_name', 'epoch', 'top1', 'top5', 'shape_bias', 'timepoint',"median_shape_bias","model_type","shape_acc","texture_acc","shape_bias_per_class",]
# )

# ################################################################################
# #                           EVALUATE ACCURACY
# ################################################################################

# #* create loader for activation extraction, then still use default test_loader
# if EXTRACT_ACTS:
#     # Load general config shared across models
#     config_file_acts = '/share/klab/lzejin/lzejin/codebase/P001_dvd_gpus/dvd/models/config/config.yaml'
#     config_dict_acts = dvd.utils.load_config(config_file_acts)
#     args_acts = argparse.Namespace(**config_dict_acts)
#     args_acts.dataset_name = EXTRACT_ACTS_DATASETS
#     args_acts.batch_size_per_gpu = 25 #* need to assign basize size properly

#     # Create test loader once for all models
#     test_acts_loader = get_test_loaders(args_acts)
#     # import pdb;pdb.set_trace()

# if EVALUATE_ACC or EVALUATE_ADV_ROBUSTNESS or EVALUATE_DEGRADATION_ROBUSTNESS or EXTRACT_ACTS or FEATURE_ATRRIBUTION_VISUALIZE:
#     # Load general config shared across models
#     config_file = '/share/klab/lzejin/lzejin/codebase/P001_dvd_gpus/dvd/models/config/config.yaml'
#     config_dict = dvd.utils.load_config(config_file)
#     args = argparse.Namespace(**config_dict)
#     args.dataset_name = DATASET_TRAINED_ON
#     # args.batch_size_per_gpu = 512 #!1024 too large for adv

#     # Create test loader once for all models
#     test_loader = get_test_loaders(args)


# ################################################################################
# #                   LOOP OVER MODELS AND EPOCHS TO EVALUATE
# ################################################################################

# # for model_name, model_path in MODEL_NAME2PATH.items():
# for model_idx, (model_name, model_path) in enumerate(MODEL_NAME2PATH.items()):
#     print(f"Evaluating Model {model_name} start from Epoch {EPOCHS[0]}")


#     for epoch in EPOCHS:
#         # analysis_id = f'{formatted_date}_{DATASET_TRAINED_ON}_{MODE}_{epoch}_'  #! Prefix for output files
#         try:
#             # Replace the single epoch in model_path with the current epoch
#             if model_path is not None:
#                 current_model_path = model_path.replace(
#                     f"_{EPOCHS[0]}.pth",
#                     f"_{epoch}.pth"
#                 )
#             else:
#                 current_model_path = None
            
#             #* across different model type
#             if model_type == 'all':
#                 model_type = model_name.replace('_baseline', '').replace('_DVD_B', '') # if model_type == 'all' else model_name
#                 # assert 'baseline' not in model_type and 'DVD_B' not in model_type
#                 if 'vit' in model_type or 'deit' in model_type: 
#                     IMAGE_SIZE = 224


#             ####################################################################
#             #   1) Evaluate Standard Accuracy
#             ####################################################################
#             # Preload models for acc realted analysis
#             if EVALUATE_ACC or EVALUATE_ADV_ROBUSTNESS or EVALUATE_DEGRADATION_ROBUSTNESS or EXTRACT_ACTS or FEATURE_ATRRIBUTION_VISUALIZE:
#                 # Load model architecture from config
#                 args.arch = model_type
#                 model, linear_keyword = dvd.models.loader.create_model(args)
                
#                 # Load checkpoint if available
#                 if current_model_path:
#                     dvd.models.loader.load_checkpoint(model=model, model_path=current_model_path, log_dir=None, args=None)
      

#             if EVALUATE_ACC:
#                 # Evaluate the model
#                 criterion = (
#                     nn.CrossEntropyLoss().cuda()
#                     if torch.cuda.is_available()
#                     else nn.CrossEntropyLoss()
#                 )
#                 top1, top5 = validate(test_loader, model, criterion, epoch,  image_size = IMAGE_SIZE)
#                 print(f"Model {model_name}, Epoch {epoch} -> Top1: {top1:.2f}, Top5: {top5:.2f}\n")

#                 # Save results
#                 new_acc_row = {
#                     'model_name': model_name,
#                     'epoch': epoch,
#                     'top1': float(top1),
#                     'top5': float(top5)
#                 }
#                 acc_df = pd.concat([acc_df, pd.DataFrame([new_acc_row])], ignore_index=True)


#                 acc_df.to_csv(
#                     os.path.join(acc_save_dir, f"{analysis_id}acc.csv"),
#                     index=False
#                 )

#             ####################################################################
#             #  2) Evaluate Shape Bias
#             ####################################################################
#             if EVALUATE_BIAS:
#                 model_type_for_eval = model_type + '_ours'#'resnet50_ours' #* can be replace by model_type + '_ours'
#                 evaluation = True
#                 load_checkpoint_path = current_model_path  # Use the same path as above
#                 dataset_trained_on = DATASET_TRAINED_ON
#                 data_normalization_type = "0-1"  # or "mean-std"
#                 matrixplot_savedir = None


#                 shape_bias, shape_bias_across_classes, classes, shape_accs, texture_accs = get_shape_bias(
#                     model_type_for_eval,
#                     evaluation=evaluation,
#                     load_checkpoint_path=load_checkpoint_path,
#                     dataset_trained_on=dataset_trained_on,
#                     image_size=IMAGE_SIZE,
#                     data_normalization_type=data_normalization_type,
#                     matrixplot_savedir=matrixplot_savedir
#                 )
#                 print(f"Model {model_name}, Epoch {epoch} -> Shape bias: {shape_bias:.2f}\n")

#                 # Append overall shape bias to a time-based DataFrame
#                 new_row = {
#                     'model_name': model_name,
#                     'model_type': model_type_for_eval,
#                     'epoch': epoch,
#                     'shape_bias': shape_bias,
#                     'median_shape_bias': np.median(shape_bias_across_classes),
#                     'timepoint': 0,
#                     # 'texture_acc': texture_accs,
#                     # 'shape_acc': shape_accs,
#                     # 'shape_bias_per_class': shape_bias_across_classes,
#                 }
#                 if EVALUATE_ACC:
#                     new_row['top1'] = float(top1)
#                     new_row['top5'] = float(top5)

#                 # shape_bias_across_time_df = shape_bias_across_time_df.append( new_row, ignore_index=True)
#                 shape_bias_across_time_df = pd.concat([shape_bias_across_time_df, pd.DataFrame([new_row])], ignore_index=True)
#                 shape_bias_csv_path = os.path.join(shape_bias_save_dir, f"{analysis_id}shape_bias.csv")
#                 shape_bias_across_time_df.to_csv(shape_bias_csv_path, index=False)
#                 print(f"Saved shape bias across time to {shape_bias_csv_path}")

#                 # Also save per-class shape bias
#                 shape_bias_classes_df = pd.DataFrame({
#                     'class_name': classes,
#                     'shape_bias_per_class': shape_bias_across_classes
#                 })
#                 class_bias_csv_path = os.path.join(
#                     shape_bias_per_class_save_dir,
#                     f"{model_name}",
#                     f"shape_bias_per_class_epoch_{epoch}.csv"
#                 )
#                 os.makedirs(os.path.dirname(class_bias_csv_path), exist_ok=True)
#                 shape_bias_classes_df.to_csv(class_bias_csv_path, index=False)
#                 print(f"Saved per-class shape bias to {class_bias_csv_path}")

#             #* 3) visulize the importance of features
#             if FEATURE_ATRRIBUTION_VISUALIZE:
#                 model.to('cuda' if torch.cuda.is_available() else 'cpu')
#                 models_dict_for_feature_atrribution[model_name + f"_ep_{epoch}"] = model
                

#             ################################################################################
#             #  5) ADVERSARIAL ATTACK EVALUATION (OPTIONAL ROBUSTNESS TEST)
#             ################################################################################

#             # If you wish to evaluate adversarial robustness, set EVALUATE_ADV_ROBUSTNESS = 1.
#             # This code integrates seamlessly with the main script above. It assumes that
#             # the model is already loaded and named `model`, and that a test DataLoader is
#             # named `test_loader`. The rest is contained within these utility functions.

#             if EVALUATE_ADV_ROBUSTNESS:
                
#                 selected_attacks = SELECTED_ADV_ATTACKS_NAMES #{k: v for k, v in ATTACKS.items() if k in SELECTED_ADV_ATTACKS_NAMES}
                
#                 evaluator = AdversarialRobustnessEvaluator(test_loader=test_loader)
#                 adv_attack_results_df = evaluator.evaluate(model, attack_subset= selected_attacks, model_name= model_name, epoch=epoch, log_dir=ADV_ROBUSTNESS_RESULTS_SAVE_DIR,)
#                 print(f"adv_attack_results_df:\n{adv_attack_results_df}")

#                 # adv_attack_df = adv_attack_analysis(model, test_loader, selected_attacks, EPSILONS_DICT, 
#                 #                     model_name=model_name, epoch=epoch, device='cuda' if torch.cuda.is_available() else 'cpu')
#                 # print(f"adv_attack_df:\n{adv_attack_df}")

#             if EXTRACT_ACTS:
#                 # EXTRACT_ACTS_DATASETS = "coco-split515"
#                 # import pdb;pdb.set_trace()
#                 rdms = extract_acts_and_create_rdm(
#                         model,
#                         test_acts_loader,
#                         layers=['conv1', 'layer1', 'layer2', 'layer3', 'layer4'], #* Default layers for ResNet50
#                         device='cuda',
#                         distance_metric = 'euclidean',  # or  'correlation' 
#                 )
                    
#                 # ave each layer’s RDM
#                 for layer_name, rdm in rdms.items():
#                     save_path = os.path.join(ACTS_RDM_SAVE_DIR, f'{model_name}_{layer_name}_rdm.npy')
#                     np.save(save_path, rdm)
#                     print(f"[INFO] Saved RDM for {model_name}' {layer_name} → {save_path}")
                
#                 # import pdb;pdb.set_trace()
            
#             if EVALUATE_DEGRADATION_ROBUSTNESS:
#                 #* Degradation robustness
#                 evaluator = DegradationRobustnessEvaluator(
#                     test_loader=test_loader,
#                     blur_loader_fn= get_blur_loader,# get_blur_loader, get_distortion_loader
#                     distortion_loader_fn= get_distortion_loader,
#                     candidate_ids=None,     # optional
#                 )

#                 criterion = nn.CrossEntropyLoss().cuda() if torch.cuda.is_available() else nn.CrossEntropyLoss()
                
#                 if EVALUATE_VARIOUS_DEGRADTION:
#                     clean_result = evaluator.evaluate_clean(test_loader, model, criterion, epoch,  image_size = IMAGE_SIZE)
#                     df_clean = evaluator.results_to_dataframe(clean_result, model_name, metric = "accuracy",)
#                     distortions_results = evaluator.evaluate_distortions(model, args, criterion, epoch,  image_size = IMAGE_SIZE)
#                     df_distortions_results = evaluator.results_to_dataframe(distortions_results, model_name, metric = "accuracy",)

#                     # Merge clean and distortion results
#                     df_merged = pd.concat([df_clean, df_distortions_results], ignore_index=True)
#                     # Save merged DataFrame to CSV
#                     image_degradtion_csv_path = os.path.join(DEGRADATION_RESULTS_SAVE_DIR, f"{model_name}_epoch{epoch}.csv")
#                     df_merged.to_csv(image_degradtion_csv_path, index=False)
#                     print(f"IMAGE DEGRADTION Results saved to {image_degradtion_csv_path}")

#                 if EVALUATE_BLUR_DEGRADATIONN:
#                     clean_result = evaluator.evaluate_clean(test_loader, model, criterion, epoch,  image_size = IMAGE_SIZE, eval_subset_classes= True, test_type='face' if 'face' in args.dataset_name else 'object')
#                     df_clean = evaluator.results_to_dataframe(clean_result, model_name, metric = "accuracy",)
#                     blur_results = evaluator.evaluate_blur( model, args, criterion, epoch,  image_size = IMAGE_SIZE, eval_subset_classes = True, test_type='face' if 'face' in args.dataset_name else 'object')
#                     df_blur = evaluator.results_to_dataframe(blur_results, model_name, metric = "accuracy",)
                    
#                     df_blur_merged = pd.concat([df_clean, df_blur], ignore_index=True)
#                     blur_degradtion_csv_path = os.path.join(DEGRADATION_RESULTS_SAVE_DIR, 'blur_degradtion', f"{args.dataset_name}_blur_degradtion_{model_name}_epoch{epoch}.csv")
#                     os.makedirs(os.path.join(DEGRADATION_RESULTS_SAVE_DIR, 'blur_degradtion'), exist_ok=True)
#                     df_blur_merged.to_csv(blur_degradtion_csv_path, index=False)
                
#                 # import pdb;pdb.set_trace()
#                 # blur_results       = evaluator.evaluate_blur(model, args, criterion, epoch,  image_size = IMAGE_SIZE) #! need update

                
                
#         except Exception as e:
#             print(f"Skipping Model {model_name}, Epoch {epoch} due to error: {e}")
#             pass


# if FEATURE_ATRRIBUTION_VISUALIZE:
    
#     #*After getting models above
#     # Suppose you have cue-conflict images: 
#     # Define the directory
#     # chossen_cat = 'None'
#     categories = [
#                     "airplane", "bear", "bicycle", "bird", "boat", "bottle", "car", "cat",
#                     "chair", "clock", "dog", "elephant", "keyboard", "knife", "oven", "truck"
#                 ][:1]
#     for chossen_cat in categories:
#         visualzie_directory = f"/share/klab/lzejin/lzejin/codebase//P001_dvd_gpus/data/cue-conflict/{chossen_cat}"
#         # List all image paths
#         image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}
#         visualize_image_paths = []
#         for root, _, files in os.walk(visualzie_directory):
#             for file in files:
#                 if os.path.splitext(file)[1].lower() in image_extensions:
#                     visualize_image_paths.append(os.path.join(root, file))

#         # visualize_image_paths = [
#         #     "/share/klab/lzejin/lzejin/codebase//P001_dvd_gpus/data/cue-conflict/cat/cat7-elephant1.png",
#         #     "/share/klab/lzejin/lzejin/codebase//P001_dvd_gpus/data/cue-conflict/elephant/elephant6-bottle2.png",
#         # ]

#         #* Call the comparison function
#         layer = 'layer4'
#         visualize_lrp_comparison(
#             model_dict=models_dict_for_feature_atrribution,
#             image_paths=visualize_image_paths,
#             image_size=256,
#             layer=layer,   # The last conv block. Could also do 'fc' or others.
#             save_path = f"./results/plots/visualize_features/{layer}/{chossen_cat}/lrp_comparison.pdf",
#         )



# ################################################################################
# #  OPTIONAL: PLOT SHAPE BIAS ACROSS EPOCHS
# ################################################################################

# if PLOT_SHAPE_BIAS_ACROSS_EPOCH:
#     # Example usage; update paths as needed
#     csv_file_path = (
#         '/share/klab/lzejin/lzejin/codebase//P001_dvd_gpus/'
#         'results/shape_bias/ecoset_across_time.csv'
#     )
#     output_plot_path = (
#         f'/share/klab/lzejin/lzejin/codebase//P001_dvd_gpus/results/plots/shape_bias/{formatted_date}_shape_bias_across_time_plot_with_human.pdf'
#     )
#     shape_texture_acc_output_plot_path = (
#         f'/share/klab/lzejin/lzejin/codebase//P001_dvd_gpus/results/plots/shape_bias/{formatted_date}_shape_acc_texture_acc_across_time_plot_updated.pdf'
#     )
#     plot_shape_bias_across_time(csv_file_path,  colors = [ 'black', np.array([49, 162, 142]) / 255,], save_path=output_plot_path)
#     plot_shape_and_texture_accuracy_across_time(csv_file_path,  colors = [ 'black', np.array([49, 162, 142]) / 255,], save_path=shape_texture_acc_output_plot_path)

# if PLOT_ADV_ROBUSTNESS:
#     from dvd.analysis.plot import curve_plot, bar_plot, plot_adv_attack_results_comparison, plot_adv_attack_results_comparison_multi

#     # Load the adversarial attack results from the CSV file
#     adv_attack_results_csv = f'results/adv_robustness/adv_attacks_all_results.csv'
#     try:
#         adv_attack_df = pd.read_csv(adv_attack_results_csv)
#     except Exception as e:
#         raise IOError(f"Unable to read CSV file at {adv_attack_results_csv}. Error: {e}")

#     # Loop over each selected adversarial attack and plot the results
#     # for attack_name in SELECTED_ADV_ATTACKS_NAMES:
#     #     print(f"{attack_name} plotting")
#     #     # plot_adv_attack_results_comparison
#     #     plot_adv_attack_results_comparison(attack_name, adv_attack_df,
#     #                          save_dir = plot_dir + '/adv_attacks', figsize=(3.54*2/3, 2))

#     # plot togethear
#     plot_adv_attack_results_comparison_multi(SELECTED_ADV_ATTACKS_NAMES, adv_attack_df,
#                              save_dir = plot_dir + '/adv_attacks', figsize=(3.54*2, 2))

# if PLOT_DEGRADATION_ROBUSTNESS:
#     from dvd.analysis.plot import plot_degradation_grid
#     # Load the degradation robustness results from the CSV file
#     degradation_results_csv = f'results/degradation_robustness/degradation_robustness_all_results.csv'
#     try:
#         degradation_df = pd.read_csv(degradation_results_csv)
#     except Exception as e:
#         raise IOError(f"Unable to read CSV file at {degradation_results_csv}. Error: {e}")

#     # Loop over each selected degradation type and plot the results
#     # for degradation_type in DEGRADATION_TYPES:
#     print(f"degradation robustness plotting")
#     # import pdb;pdb.set_trace()
#     plot_degradation_grid(degradation_df,
#                             save_dir = plot_dir + '/degradation_robustness', 
#                             save_name = "degradation_robustness_grid_v7_all.pdf", figsize=(3.54, 4),
#                             relative=False)
    
#!/usr/bin/env python3
"""analysis_main_figure_rewrite.py
------------------------------------------------------
A clearer, more modular rewrite of the original analysis
script used to reproduce the main figures of the Nature
paper. **All original annotations (Figure references,    
dataset notes, etc.) have been retained** so the paper can
be replicated one-to-one, while the code structure has
been reorganised for readability and ease-of-use.

Usage
-----
```bash
python analysis.py \
    --dataset-id 1 \
    --model-type resnet50 \
    --epochs best
```
All flags default to the values that reproduce the paper
figures. See ``python analysi.py -h``
for full CLI options.
"""
from __future__ import annotations

# ============================================================================
# Standard library imports
# ============================================================================
import argparse
import datetime as _dt
import itertools as _it
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

# ============================================================================
# Third-party imports
# ============================================================================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# ============================================================================
# Project-specific imports
# ============================================================================
from neuroai.evaluation.shape_bias_eval import get_shape_bias
from dvd.analysis.save import (
    save_shape_biases_across_time_csv,
    save_shape_biases_across_categories_csv,
)
from dvd.analysis.adv_attack import AdversarialRobustnessEvaluator
from dvd.analysis.degradation_robustness import DegradationRobustnessEvaluator
from dvd.analysis.feature_atrribution_visualize import (
    visualize_lrp_comparison,
)

from dvd.analysis.plot import (
    plot_shape_bias_across_time,
    plot_shape_and_texture_accuracy_across_time,
    ModelAccuracyPlotter, 
    plot_sweeped_shape_bias_tradeoff_scatter_colormap, 
    plot_shape_bias_tradeoff_scatter_colormap, 
    plot_shape_bias_tradeoff_scatter, plot_shape_bias_data_size_tradeoff, 
)
from dvd.datasets.dataset_loader import (
    get_blur_loader,
    get_distortion_loader,
    get_test_loaders,
)
from dvd.models.eval import validate
from dvd.models.extract_acts import extract_acts_and_create_rdm
import dvd.models.loader  # noqa: E402 keep after torch import
import dvd.utils

# ============================================================================
#                           ANALYSIS CONFIGURATIONS
# ============================================================================
# * Figure 1 is an illustration plot
# * To plot development trajectory of different visual aspects, check
#   dvd.dvd.development.EarlyVisualDevelopmentTransformer

# * Figure 2 ACC & Shape bias
#   Figure 1A just cue-conflicting images
#   Fig 2C: to plot multiple models' shape bias in a benchmark, use code in
#   another repo based on Geirhos code:
#   https://github.com/KietzmannLab/Neuro-vs-AI/blob/main/scripts/evaluate.py
EVALUATE_ACC: int = 1              # Fig 2B: Evaluate standard accuracy
EVALUATE_BIAS: int = 1             # Fig 2B: Evaluate shape bias
PLOT_ACC_BIAS_TRADEOFF = 0 

# * Figure 3 | generalisation across vision datasets and ANN architectures.
#   All shape bias evaluation mainly based on Geirhos code
#   Just check code in another repo:
#   https://github.com/KietzmannLab/Neuro-vs-AI/blob/main/scripts/evaluate.py

# * Figure 4 | In-depth investigation of the emergence of shape bias & feature
#   attribution visualisation
PLOT_SHAPE_BIAS_ACROSS_EPOCH: int = 0  # Figure 4A | Generate shape-bias plot
FEATURE_ATRRIBUTION_VISUALIZE: int = 0  # Figure 4B feature-attribution vis.
models_dict_for_feature_atrribution: Dict[str, torch.nn.Module] = {}
# Figure 4C control rearing, see Geirhos repo above

# * Figure 5 abstract shape recognition
#   Fig 5B just an illustration plot
#   Fig 5B plot benchmark:
#     scripts/abstract_shape_analysis_scripts/plot_illusion_benchmark.py
#   Fig 5C t-SNE visualisation:
#     scripts/abstract_shape_analysis_scripts/tsne_visualization_abstract_shape.py

# * Figure 6
#   Figure 6 A & B | Degradation robustness
EVALUATE_DEGRADATION_ROBUSTNESS: int = 0  # Evaluate degradation
EVALUATE_BLUR_DEGRADATIONN: int = 0
PLOT_BLUR_DEGRADATION_ROBUSTNESS : int = 0          # Plot blur degradation results
EVALUATE_VARIOUS_DEGRADTION: int = 0
PLOT_DEGRADATION_ROBUSTNESS: int = 0      # Plot various degradation results
DEGRADATION_RESULTS_SAVE_DIR: Path = Path("./results/degradation_robustness")
DEGRADATION_RESULTS_SAVE_DIR.mkdir(parents=True, exist_ok=True)

#   Figure 6C | adversarial robustness
EVALUATE_ADV_ROBUSTNESS: int = 0          # Evaluate selected attacks
PLOT_ADV_ROBUSTNESS: int = 0              # Plot adversarial robustness
ADV_ROBUSTNESS_RESULTS_SAVE_DIR: Path = Path("./results/adv_robustness")
SELECTED_ADV_ATTACKS_NAMES: List[str] = [
    "L2AdditiveGaussianNoiseAttack",
    "L2AdditiveUniformNoiseAttack",
    "SaltAndPepperNoiseAttack",
    "FGSM",
    "FGM",
    "PGD",
]  # Black-box (noise-based) & white-box attacks

# ---------------------------------------------------------------------------
# Mode for shape-bias evaluation
#   'single'   – evaluate specific epochs (default)
#   'traverse' – scan a range of epochs
# ---------------------------------------------------------------------------
MODE: str = ["single", "traverse"][0]
if MODE == "single":
    EPOCHS: Sequence[str | int] = ["last", "best"][1:]  # Example: choose 'best'
else:
    EPOCHS = list(range(4, 300, 5))

# ---------------------------------------------------------------------------
# Dataset / model configuration
# ---------------------------------------------------------------------------
dataset_id: int = 1  # 0=texture2shape_miniecoset, 1=ecoset, 2=imagenet, 3=imagenet_16, 4=facescrub
DATASET_TRAINED_ON: str = [
    "texture2shape_miniecoset",
    "ecoset_square256",
    "imagenet",
    "imagenet_16",
    "facescrub",
][dataset_id]
IMAGE_SIZE: int | None = [256, 224, None][0]  # Default 256×256; ViT models switch to 224

# Choose model architecture by index
model_type: str = [
    "all",
    "resnet50",
    "resnet101",
    "deit_base_patch16_224",
    "vit_b_16",
    "swin_b",
    "alexnet",
    "vgg16",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "blt_vs",
][1]

# Hyper-parameters for EVD (Early Visual Development)
months_per_epochs: Sequence[float] = [1.0, 2.0, 4.0, 8.0, 3.0, 0.5, 0.25][:0]
contrast_thresh: Sequence[float] = [0.1, 0.2, 0.4, 0.8, 0.05][:0]
contrast_spd: Sequence[float] = [50.0, 100.0, 150.0, 300.0][:0]

# ---------------------------------------------------------------------------
# Output-directory bookkeeping
# ---------------------------------------------------------------------------
_today = _dt.date.today()
formatted_date: str = f"{_today.day}th_{_today.strftime('%B')}"
analysis_id: str = (
    f"{formatted_date}_{DATASET_TRAINED_ON}_across_architecture_"
    f"{MODE}_{EPOCHS[0]}_"
)

ACC_SAVE_DIR = Path("./results/acc")
SHAPE_BIAS_SAVE_DIR = Path("./results/shape_bias")
SHAPE_BIAS_PER_CLASS_SAVE_DIR = SHAPE_BIAS_SAVE_DIR / "shape_bias_per_class"
PLOT_DIR = Path("./results/plots")
for _p in (ACC_SAVE_DIR, SHAPE_BIAS_PER_CLASS_SAVE_DIR, PLOT_DIR):
    _p.mkdir(parents=True, exist_ok=True)

# ============================================================================
#                MODEL NAME ➔ CHECKPOINT-PATH MAPPING HELPERS
# ============================================================================

# Option 1 – manually specify name → checkpoint directory mapping
_MODEL_NAME2PATH_RAW: Dict[str, str] = {
    #* imagenet
    "adult": "resnet50_imagenet_256_0.0001_dev_adult_seed_1",
    "resnet50_mpe1.0_alpha0.1_dn100.0": "resnet50_mpe1.0_alpha0.1_dn100.0_imagenet256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    "resnet50_mpe2.0_alpha0.1_dn150.0": "resnet50_mpe2.0_alpha0.1_dn150.0_imagenet256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    "resnet50_mpe2.0_alpha0.2_dn100.0": "resnet50_mpe2.0_alpha0.2_dn100.0_imagenet256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    "resnet50_mpe4.0_alpha0.4_dn50.0": "resnet50_mpe4.0_alpha0.4_dn50.0_imagenet256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",
    "resnet50_mpe8.0_alpha0.8_dn50.0": "resnet50_mpe8.0_alpha0.8_dn50.0_imagenet256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",

    # #* ecoset
    'resnet50_baseline':'resnet50_mpe2.0_alpha0.2_dn100.0_ecoset_square256256_0.0001_dev_adult_b1c1cs1_T_normal_seed_1',
    'resnet50_DVD-PP':'resnet50_mpe8.0_alpha0.2_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    'resnet50_DVD-P':'resnet50_mpe4.0_alpha0.4_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    'resnet50_TVD-B': 'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    'resnet50_DVD-S':'resnet50_mpe1.0_alpha0.1_dn100.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    'resnet50_DVD-SS':'resnet50_mpe1.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',

    #* miniecoset
    'adult':'resnet50_texture2shape_miniecoset_256_0.0001_dev_adult_seed_1',
    'resnet50_mpe1.0_alpha0.2_dn150.0':'resnet50_mpe1.0_alpha0.2_dn150.0_texture2shape_miniecoset256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    'resnet50_mpe2.0_alpha0.1_dn100.0':'resnet50_mpe2.0_alpha0.1_dn100.0_texture2shape_miniecoset256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    'resnet50_mpe4.0_alpha0.8_dn50.0':'resnet50_mpe4.0_alpha0.8_dn50.0_texture2shape_miniecoset256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    'resnet50_mpe8.0_alpha0.8_dn100.0':'resnet50_mpe8.0_alpha0.8_dn100.0_texture2shape_miniecoset256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    
    'resnet50_mpe1.0_alpha0.1_dn100.0':'resnet50_mpe1.0_alpha0.1_dn100.0_texture2shape_miniecoset256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    'resnet50_mpe2.0_alpha0.1_dn150.0':'resnet50_mpe2.0_alpha0.1_dn150.0_texture2shape_miniecoset256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    'resnet50_mpe2.0_alpha0.2_dn100.0':'resnet50_mpe2.0_alpha0.2_dn100.0_texture2shape_miniecoset256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    'resnet50_mpe4.0_alpha0.4_dn50.0':'resnet50_mpe4.0_alpha0.4_dn50.0_texture2shape_miniecoset256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    'resnet50_mpe8.0_alpha0.8_dn50.0':'resnet50_mpe8.0_alpha0.8_dn50.0_texture2shape_miniecoset256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    
    #* multiple seeds | TODO | Not finished
    'adult_seed_1':'resnet50_mpe2.0_alpha0.2_dn100.0_ecoset_square256256_0.0001_dev_adult_b1c1cs1_T_normal_seed_1',
    'adult_seed_2':'resnet50_mpe2.0_alpha0.2_dn100.0_ecoset_square256256_0.0001_dev_adult_b1c1cs1_T_normal_seed_2',
    'adult_seed_3':'resnet50_mpe2.0_alpha0.2_dn100.0_ecoset_square256256_0.0001_dev_adult_b1c1cs1_T_normal_seed_3',
    "mpe2.0_alpha0.1_dn150.0_seed_1": 'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    "mpe2.0_alpha0.1_dn150.0_seed_2": 'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_2',
    "mpe2.0_alpha0.1_dn150.0_seed_3": 'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_3',

    ##* ablation | based on DVD-B
    'resnet50_mpe2.0_alpha0.1_dn150.0_b1c1cs1':'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1',
    'resnet50_mpe2.0_alpha0.1_dn150.0_b1c0cs0':'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c0cs0_T_normal_seed_1',
    'resnet50_mpe2.0_alpha0.1_dn150.0_b0c1cs0':'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b0c1cs0_T_normal_seed_1',
    'resnet50_mpe2.0_alpha0.1_dn150.0_b0c0cs1':'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b0c0cs1_T_normal_seed_1',
    'resnet50_mpe2.0_alpha0.1_dn150.0_b1c1cs0':'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs0_T_normal_seed_1',
    'resnet50_mpe2.0_alpha0.1_dn150.0_b1c0cs1':'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c0cs1_T_normal_seed_1',
    'resnet50_mpe2.0_alpha0.1_dn150.0_b0c1cs1':'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b0c1cs1_T_normal_seed_1',

    #* other models:
    'alexnet_DVD_B': "alexnet_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1", 
    'alexnet_baseline': "alexnet_ecoset_square256_256_0.0001_dev_adult_seed_1", 
    'vgg16_DVD_B': "vgg16_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1", 
    'vgg16_baseline': "vgg16_ecoset_square256_256_0.0001_dev_adult_seed_1", 
    'mobilenet_v3_small_baseline': "mobilenet_v3_small_ecoset_square256_256_0.0001_dev_adult_seed_1", 
    'mobilenet_v3_small_DVD_B': "mobilenet_v3_small_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1", 
    'resnet18_baseline': "resnet18_ecoset_square256_256_0.0001_dev_adult_seed_1", 
    'resnet18_DVD_B': "resnet18_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1", 
    'resnet101_baseline': "resnet101_ecoset_square256_256_0.0001_dev_adult_seed_1", 
    'resnet101_DVD_B': "resnet101_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1", 
    'resnext50_32x4d_baseline': "resnext50_32x4d_ecoset_square256_256_0.0001_dev_adult_seed_1", 
    'resnext50_32x4d_DVD_B': "resnext50_32x4d_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1", 
    'deit_base_patch16_224_baseline': "deit_base_patch16_224_ecoset_square256_256_0.0001_dev_adult_seed_1", 
    'deit_base_patch16_224_DVD_B': "deit_base_patch16_224_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1", 
    'vit_b_16_baseline': "vit_b_16_ecoset_square256_256_0.0001_dev_adult_seed_1", 
    'vit_b_16_DVD_B': "vit_b_16_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1",

    #* No blurring aug for more fair robustness evaluation
    'ecoset_adult_no_blur':'resnet50_ecoset_square256_256_0.0001_dev_adult_seed_1_no_blur_aug',
    'ecoset_DVD_B_no_blur':'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1_no_blur_aug',
    'ecoset_DVD_S_no_blur':'resnet50_mpe1.0_alpha0.1_dn100.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1_no_blur_aug',
    'ecoset_DVD_SS_no_blur':'resnet50_mpe1.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1_no_blur_aug',
    'ecoset_DVD_P_no_blur':'resnet50_mpe4.0_alpha0.4_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1_no_blur_aug',
    'ecoset_DVD_PP_no_blur':'resnet50_mpe8.0_alpha0.2_dn50.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1_no_blur_aug',

    #* Faces trained only
    'Face_DVD-B_no_blur_seed_1':'resnet50_mpe2.0_alpha0.1_dn150.0_facescrub256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1_no_blur_aug',
    'Face_baseline_no_blur_seed_1':'resnet50_facescrub_256_0.0001_dev_adult_seed_1_no_blur_aug',
    
    # cirtical period
    'DVD-B_random_order': 'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_random_seed_2',
    'DVD-B_fully_random': 'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_fully_random_seed_2',
    'DVD-B_mid_phase': 'resnet50_mpe2.0_alpha0.1_dn150.0_ecoset_square256256_0.0001_dev_evd_b1c1cs1_T_mid_phase_seed_2',

    #* No grayscale aug
    'resnet50_mpe1.0_alpha0.2_dn150.0_no_grayscale':'resnet50_mpe1.0_alpha0.2_dn150.0_texture2shape_miniecoset256_0.0001_dev_evd_b1c1cs1_T_normal_seed_100', #0.73 (acc from 45% to 52%)
    'resnet50_mpe2.0_alpha0.2_dn150.0_no_grayscale':'resnet50_mpe2.0_alpha0.2_dn150.0_texture2shape_miniecoset256_0.0001_dev_evd_b1c1cs1_T_normal_seed_100', #0.68
    
}

# Convert symbolic directory names → full checkpoint paths
MODEL_NAME2PATH: Dict[str, str | None] = {
    name: f"/share/klab/lzejin/lzejin/codebase//P001_dvd_gpus/logs/{dir_}/weights/checkpoint_{EPOCHS[0]}.pth" if dir_ else None
    for name, dir_ in _MODEL_NAME2PATH_RAW.items()
}

# Option 2 – generate trade-off models by traversing hyper-parameters
for mpe, alpha, beta in _it.product(months_per_epochs, contrast_thresh, contrast_spd):
    model_name = f"mpe{mpe}_alpha{alpha}_dn{beta}"
    model_subdir = (
        f"{model_type}_mpe{mpe}_alpha{alpha}_dn{beta}_"
        f"{DATASET_TRAINED_ON}256_0.0001_dev_evd_b1c1cs1_T_normal_seed_1"
    )
    MODEL_NAME2PATH[model_name] = (
        f"logs/{model_subdir}/weights/checkpoint_{EPOCHS[0]}.pth"
    )

# ============================================================================
#                           DATAFRAME DECLARATIONS
# ============================================================================
acc_df = pd.DataFrame(columns=["model_name", "epoch", "top1", "top5"])
shape_bias_across_time_df = pd.DataFrame(
    columns=[
        "model_name",
        "epoch",
        "top1",
        "top5",
        "shape_bias",
        "timepoint",
        "median_shape_bias",
        "model_type",
        "shape_acc",
        "texture_acc",
        "shape_bias_per_class",
    ]
)

# ============================================================================
# Helper functions
# ============================================================================

def _load_common_config() -> argparse.Namespace:
    """Load YAML config and convert to argparse.Namespace."""
    cfg_path = Path("/share/klab/lzejin/lzejin/codebase/P001_dvd_gpus/dvd/models/config/config.yaml")
    cfg_dict = dvd.utils.load_config(cfg_path)
    return argparse.Namespace(**cfg_dict)


def _build_test_loader(dataset_name: str, batch_size: int | None = None):
    args = _load_common_config()
    args.dataset_name = dataset_name
    if batch_size is not None:
        args.batch_size_per_gpu = batch_size
    return get_test_loaders(args)


def _prepare_model(arch: str, ckpt_path: str , device: torch.device) -> torch.nn.Module:
    """Instantiate model architecture and (optionally) load checkpoint."""
    args = _load_common_config()
    args.arch = arch
    args.dataset_name = DATASET_TRAINED_ON
    model, _ = dvd.models.loader.create_model(args)
    if ckpt_path and Path(ckpt_path).is_file():
        dvd.models.loader.load_checkpoint(model, model_path=ckpt_path, log_dir=None, args=args)
    model.to(device)
    model.eval()
    return model


# ============================================================================
# Main evaluation routine
# ============================================================================

def evaluate_models() -> None:
    """Loop over all model/epoch combinations and perform requested analyses."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------------------------------------------------
    # Test loader is shared by *most* evaluations
    # -----------------------------------------------------------------------
    test_loader = _build_test_loader(DATASET_TRAINED_ON)

    for model_name, base_ckpt in MODEL_NAME2PATH.items():
        print(f"\n[INFO] Evaluating model '{model_name}' (starting epoch: {EPOCHS[0]})")

        for epoch in EPOCHS:
            # Replace epoch token in checkpoint path (if path provided)
            ckpt_path = (
                base_ckpt.replace(f"_{EPOCHS[0]}.pth", f"_{epoch}.pth")
                if base_ckpt is not None
                else None
            )

            # When model_type == 'all' we infer architecture from model_name
            arch_name = model_type
            if model_type == "all":
                arch_name = model_name.replace("_baseline", "").replace("_DVD_B", "")
                if any(s in arch_name for s in ("vit", "deit")):
                    global IMAGE_SIZE  # noqa: PLW0603 – explicit global update
                    IMAGE_SIZE = 224

            try:
                # ----------------------------------------------------------
                # 1) Standard accuracy (Top-1 / Top-5)
                # ----------------------------------------------------------
                if EVALUATE_ACC or any(
                    (
                        EVALUATE_ADV_ROBUSTNESS,
                        EVALUATE_DEGRADATION_ROBUSTNESS,
                        FEATURE_ATRRIBUTION_VISUALIZE,
                    )
                ):
                    model = _prepare_model(arch_name, ckpt_path, device)

                if EVALUATE_ACC:
                    criterion = (
                        nn.CrossEntropyLoss().to(device)
                        if device.type == "cuda"
                        else nn.CrossEntropyLoss()
                    )
                    top1, top5 = validate(test_loader, model, criterion, epoch, IMAGE_SIZE)
                    print(f"  ↳ Epoch {epoch}: Top-1 {top1:.2f} | Top-5 {top5:.2f}")

                    # Save to dataframe and CSV
                    acc_df.loc[len(acc_df)] = [model_name, epoch, float(top1), float(top5)]
                    acc_df.to_csv(ACC_SAVE_DIR / f"{analysis_id}acc.csv", index=False)

                # ----------------------------------------------------------
                # 2) Shape bias
                # ----------------------------------------------------------
                if EVALUATE_BIAS:
                    model_type_for_eval = f"{arch_name}_ours"
                    shape_bias, sb_per_class, classes, shape_accs, texture_accs = get_shape_bias(
                        model_type_for_eval,
                        evaluation=True,
                        load_checkpoint_path=ckpt_path,
                        dataset_trained_on=DATASET_TRAINED_ON,
                        image_size=IMAGE_SIZE,
                        data_normalization_type="0-1",
                        matrixplot_savedir=None,
                    )
                    print(f"  ↳ Epoch {epoch}: Shape-bias {shape_bias:.2f}")

                    row = {
                        "model_name": model_name,
                        "model_type": model_type_for_eval,
                        "epoch": epoch,
                        "shape_bias": shape_bias,
                        "median_shape_bias": np.median(sb_per_class),
                        "timepoint": 0,
                        "shape_acc": shape_accs,
                        "texture_acc": texture_accs,
                        "shape_bias_per_class": sb_per_class,
                    }
                    if EVALUATE_ACC:
                        row.update({"top1": float(top1), "top5": float(top5)})

                    shape_bias_across_time_df.loc[len(shape_bias_across_time_df)] = row
                    shape_bias_csv = SHAPE_BIAS_SAVE_DIR / f"{analysis_id}shape_bias.csv"
                    shape_bias_across_time_df.to_csv(shape_bias_csv, index=False)

                    # Save per-class bias
                    per_class_df = pd.DataFrame(
                        {
                            "class_name": classes,
                            "shape_bias_per_class": sb_per_class,
                        }
                    )
                    pc_dir = SHAPE_BIAS_PER_CLASS_SAVE_DIR / model_name
                    pc_dir.mkdir(parents=True, exist_ok=True)
                    per_class_df.to_csv(
                        pc_dir / f"shape_bias_per_class_epoch_{epoch}.csv", index=False
                    )

                # ----------------------------------------------------------
                # 3) Feature-attribution visualisation
                # ----------------------------------------------------------
                if FEATURE_ATRRIBUTION_VISUALIZE:
                    models_dict_for_feature_atrribution[f"{model_name}_ep{epoch}"] = model

                # ----------------------------------------------------------
                # 4) Adversarial robustness evaluation
                # ----------------------------------------------------------
                if EVALUATE_ADV_ROBUSTNESS:
                    evaluator = AdversarialRobustnessEvaluator(test_loader)
                    evaluator.evaluate(
                        model,
                        attack_subset=SELECTED_ADV_ATTACKS_NAMES,
                        model_name=model_name,
                        epoch=epoch,
                        log_dir=ADV_ROBUSTNESS_RESULTS_SAVE_DIR,
                    )

                # ----------------------------------------------------------
                # 5) Degradation robustness evaluation
                # ----------------------------------------------------------
                if EVALUATE_DEGRADATION_ROBUSTNESS:
                    degr_eval = DegradationRobustnessEvaluator(
                        test_loader,
                        blur_loader_fn=get_blur_loader,
                        distortion_loader_fn=get_distortion_loader,
                    )
                    criterion = nn.CrossEntropyLoss().to(device)

                    if EVALUATE_VARIOUS_DEGRADTION:
                        clean_result = degr_eval.evaluate_clean(
                            test_loader, model, criterion, epoch, IMAGE_SIZE
                        )
                        distortions = degr_eval.evaluate_distortions(
                            model, _load_common_config(), criterion, epoch, IMAGE_SIZE
                        )
                        df = pd.concat(
                            [
                                degr_eval.results_to_dataframe(clean_result, model_name, "accuracy"),
                                degr_eval.results_to_dataframe(distortions, model_name, "accuracy"),
                            ],
                            ignore_index=True,
                        )
                        df.to_csv(
                            DEGRADATION_RESULTS_SAVE_DIR / f"{model_name}_epoch{epoch}.csv",
                            index=False,
                        )

                    if EVALUATE_BLUR_DEGRADATIONN:
                        subtype = "face" if "face" in DATASET_TRAINED_ON else "object"
                        clean = degr_eval.evaluate_clean(
                            test_loader,
                            model,
                            criterion,
                            epoch,
                            IMAGE_SIZE,
                            eval_subset_classes=True,
                            test_type=subtype,
                        )
                        blur = degr_eval.evaluate_blur(
                            model,
                            _load_common_config(),
                            criterion,
                            epoch,
                            IMAGE_SIZE,
                            eval_subset_classes=True,
                            test_type=subtype,
                        )
                        df_blur = pd.concat(
                            [
                                degr_eval.results_to_dataframe(clean, model_name, "accuracy"),
                                degr_eval.results_to_dataframe(blur, model_name, "accuracy"),
                            ],
                            ignore_index=True,
                        )
                        blur_dir = DEGRADATION_RESULTS_SAVE_DIR / "blur_degradation"
                        blur_dir.mkdir(exist_ok=True)
                        df_blur.to_csv(
                            blur_dir / f"{DATASET_TRAINED_ON}_blur_{model_name}_epoch{epoch}.csv",
                            index=False,
                        )

                # ----------------------------------------------------------
                # 6) Activation extraction (optional)
                # ----------------------------------------------------------
                # To extract activations, set EXTRACT_ACTS flag in config-section
                # and adapt EXTRACT_ACTS_DATASETS / ACTS_RDM_SAVE_DIR.

            except Exception as exc:  # pylint: disable=broad-except
                import pdb;pdb.set_trace()
                print(f"  [WARN] Skipping {model_name} @ epoch {epoch}: {exc}")

    # -----------------------------------------------------------------------
    # Post-processing – plotting and visualisation
    # -----------------------------------------------------------------------
    if PLOT_ACC_BIAS_TRADEOFF:
        _plot_sweeped_and_ecoset_tradeoff(colormap=["magma", "viridis", "plasma"][1])  

    if PLOT_SHAPE_BIAS_ACROSS_EPOCH:
        csv_path = (
            Path("/share/klab/lzejin/lzejin/codebase//P001_dvd_gpus/results/shape_bias/")
            / "ecoset_across_time.csv"
        )
        sb_plot_path = (
            PLOT_DIR
            / "shape_bias"
            / f"{formatted_date}_shape_bias_across_time_plot_with_human.pdf"
        )
        st_plot_path = (
            PLOT_DIR
            / "shape_bias"
            / f"{formatted_date}_shape_acc_texture_acc_across_time_plot.pdf"
        )
        plot_shape_bias_across_time(csv_path, colors=["black", np.array([49, 162, 142]) / 255], save_path=sb_plot_path)
        plot_shape_and_texture_accuracy_across_time(csv_path, colors=["black", np.array([49, 162, 142]) / 255], save_path=st_plot_path)

    if FEATURE_ATRRIBUTION_VISUALIZE:
        _run_feature_visualisation()

    if PLOT_ADV_ROBUSTNESS:
        _plot_adv_robustness()

    if PLOT_DEGRADATION_ROBUSTNESS:
        _plot_degradation_robustness()

    if PLOT_BLUR_DEGRADATION_ROBUSTNESS:
        model_names = ['Face_baseline_no_blur_seed_1',  
               'Face_DVD-B_no_blur_seed_1', 
               'ecoset_adult_no_blur',  'ecoset_DVD_B_no_blur',  'ecoset_DVD_S_no_blur', 'ecoset_DVD_P_no_blur', 
               'ecoset_DVD_SS_no_blur', 'ecoset_DVD_PP_no_blur'][:2]#[3:5]
        dataset_name = ['facescrub','ecoset_square256'][0]
        _plot_blur_degradation_robustness(model_names=model_names, dataset_name=dataset_name, epoch='best')

# ============================================================================
# Auxiliary top-level functions (kept separate for clarity)
# ============================================================================
def _plot_sweeped_and_ecoset_tradeoff(colormap):
    # Plot sweeped shape bias tradeoff
    plot_sweeped_shape_bias_tradeoff_scatter_colormap(
        csv_path=".results/shape_bias/across_time_miniecoset_sweeped_shape_bias_acc_tradeoff_best_epoch.csv",
        analysis_name=f"miniecoset_sweeped_shape_bias_acc_trade_off_colormap_{colormap}_v2",
        save_dir="./results/plots/trade_off",
        highlight_models=['mpe2.0_alpha0.1_dn100.0', 'mpe1.0_alpha0.2_dn150.0', 'mpe4.0_alpha0.8_dn50.0'],
        xlim=(45, 74),
        ylim=(0.3, 1),
        figsize=(3.5 * 0.75, 2),
        cmap_name=colormap,
        reverse=False,
        use_fixed_range=True,
        fixed_range=(0.6, 1),
    )

    # Plot shape bias tradeoff for ecoset
    plot_shape_bias_tradeoff_scatter_colormap(
        csv_path=".results/shape_bias/ecoset_shape_bias_acc_trade_off.csv",
        analysis_name=f"ecoset_shape_bias_acc_trade_off_colormap_{colormap}",
        save_dir="./results/plots/trade_off",
        highlight_models=['DVD-B', 'DVD-S', 'DVD-P'],
        xlim=(47, 67),
        ylim=(0.3, 1),
        figsize=(3.5 * 0.75, 2),
        cmap_name=colormap,
        reverse=False,
        use_fixed_range=True,
        fixed_range=(0.6, 1),
    )


def _run_feature_visualisation() -> None:
    """Run LRP comparison on cue-conflict images for selected categories."""
    categories = [
        "airplane",
        "bear",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "car",
        "cat",
        "chair",
        "clock",
        "dog",
        "elephant",
        "keyboard",
        "knife",
        "oven",
        "truck",
    ][:1]
    for cat in categories:
        img_dir = Path("/share/klab/lzejin/lzejin/codebase//P001_dvd_gpus/data/cue-conflict") / cat
        image_paths: List[Path] = [
            p for p in img_dir.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ]
        layer = "layer4"
        save_path = PLOT_DIR / "visualize_features" / layer / cat / "lrp_comparison.pdf"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        visualize_lrp_comparison(
            model_dict=models_dict_for_feature_atrribution,
            image_paths=[str(p) for p in image_paths],
            image_size=256,
            layer=layer,
            save_path=str(save_path),
        )


def _plot_adv_robustness() -> None:
    from dvd.analysis.plot import plot_adv_attack_results_comparison_multi

    csv_file = Path("results/adv_robustness/adv_attacks_all_results.csv")
    if not csv_file.exists():
        print(f"[WARN] Could not find {csv_file} for adversarial plotting.")
        return
    df = pd.read_csv(csv_file)
    out_dir = PLOT_DIR / "adv_attacks"
    plot_adv_attack_results_comparison_multi(
        SELECTED_ADV_ATTACKS_NAMES, df, save_dir=out_dir, figsize=(3.54 * 2, 2)
    )


def _plot_blur_degradation_robustness(model_names, dataset_name, epoch='best'):
    degradation_results_dir = './results/degradation_robustness/blur_degradtion/'
    model_acc_dict = {}

    for model_name in model_names:
        csv_path = os.path.join(degradation_results_dir, f"{dataset_name}_blur_degradtion_{model_name}_epoch{epoch}.csv")
        df = pd.read_csv(csv_path)
        model_acc_dict[model_name] = df['accuracy'].tolist()[1:]  # Skip sigma = 0 (assume sigma starts from 1)

    plotter = ModelAccuracyPlotter(
        save_dir='./results/sigma_blur_attack',
        human_data_path='./data/blur_degradation_human_experiment/human_accuracy.mat'
    )
    plotter.plot_model_accuracy(
        new_models=model_acc_dict,
        dataset_name=dataset_name,
        absolute_acc=True,
        save_name=f'human_and_models_{dataset_name}_vs_sigma_blur_attack_1st_July_all'
    )

def _plot_degradation_robustness() -> None:
    from dvd.analysis.plot import plot_degradation_grid

    csv_file = Path("results/degradation_robustness/degradation_robustness_all_results.csv")
    if not csv_file.exists():
        print(f"[WARN] Could not find {csv_file} for degradation plotting.")
        return
    df = pd.read_csv(csv_file)
    out_dir = PLOT_DIR / "degradation_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_degradation_grid(
        df,
        save_dir=out_dir,
        save_name="degradation_robustness_grid_v7_all.pdf",
        figsize=(3.54, 4),
        relative=False,
    )


# ============================================================================
# CLI entry-point
# ============================================================================

def _parse_cli() -> argparse.Namespace:
    """Parse command-line arguments to override common settings."""
    parser = argparse.ArgumentParser(description="Evaluate models for Nature paper figures")
    parser.add_argument("--dataset-id", type=int, default=dataset_id, help="Dataset identifier index (see script for mapping)")
    parser.add_argument("--model-type", type=str, default=model_type, help="Model architecture identifier")
    parser.add_argument("--epochs", nargs="*", default=EPOCHS, help="Epoch(s) to evaluate, e.g. best last or 0 5 10 …")
    return parser.parse_args()


def main() -> None:
    # CLI overrides (optional)
    cli_args = _parse_cli()
    globals()["dataset_id"] = cli_args.dataset_id  # noqa: PLW0603
    globals()["model_type"] = cli_args.model_type
    globals()["EPOCHS"] = [str(e) for e in cli_args.epochs]

    evaluate_models()


if __name__ == "__main__":
    main()

    



