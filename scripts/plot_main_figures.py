import pandas as pd
import os

# from dvd.analysis.plot import plot_shape_bias_across_time, plot_sweeped_shape_bias_tradeoff_scatter, plot_shape_bias_tradeoff_scatter 
from dvd.analysis.plot import ModelAccuracyPlotter
from dvd.analysis.plot import plot_sweeped_shape_bias_tradeoff_scatter_colormap, plot_shape_bias_tradeoff_scatter_colormap, plot_shape_bias_tradeoff_scatter, plot_shape_bias_data_size_tradeoff 

################################################################################
## Figure 2 Training with TVD brings model with shape bias
################################################################################

# Figure 2A is just illustration panel
#* Figure 2B | Shpae bias - Accuracy trade off
# sweeped shape bias - accuracy trade off across models in mini-ecoset
# plot_sweeped_shape_bias_tradeoff_scatter(
# colormap = ["magma", "viridis", "plasma" ][1]
# plot_sweeped_shape_bias_tradeoff_scatter_colormap(
#     csv_path="results/shape_bias/across_time_miniecoset_sweeped_shape_bias_acc_tradeoff_best_epoch.csv",
#     analysis_name=f"miniecoset_sweeped_shape_bias_acc_trade_off_colormap_{colormap}_v2",
#     save_dir="./results/plots/trade_off",
#     highlight_models = ['mpe2.0_alpha0.1_dn100.0', 'mpe1.0_alpha0.2_dn150.0', 'mpe4.0_alpha0.8_dn50.0'],
#     xlim = (45, 74),
#     ylim = (0.3, 1),
#     figsize = (3.5*0.75, 2),
#     cmap_name = colormap,
#     reverse =  False,           # whether to invert cmap_name
#     use_fixed_range = True,   # switch normalization of colormap mode
#     fixed_range = (0.6, 1), # only used if use_fixed_range=True
# )

# # # trade off across models in ecoset
# # plot_shape_bias_tradeoff_scatter(
# plot_shape_bias_tradeoff_scatter_colormap(
#     csv_path="results/shape_bias/ecoset_shape_bias_acc_trade_off.csv",
#     analysis_name=f"ecoset_shape_bias_acc_trade_off_colormap_{colormap}",
#     save_dir="./results/plots/trade_off",
#     highlight_models = ['DVD-B', 'DVD-S', 'DVD-P'],
#     xlim = (47, 67),
#     ylim = (0.3, 1),
#     figsize = (3.5*0.75, 2),
#     cmap_name = colormap,
#     reverse =  False,           # whether to invert cmap_name
#     use_fixed_range = True,   # switch normalization of colormap mode
#     fixed_range = (0.6, 1), # only used if use_fixed_range=True
# )

#* training data size vs shape bais
# shape_bias_data_size_csv_path = '/home/student/l/lzejin/codebase/P001_dvd_gpus/results/shape_bias/all_models_shape-bias_data_size_model_size.csv'
# plot_shape_bias_data_size_tradeoff(shape_bias_data_size_csv_path, save_path="./results/plots/trade_off/data_size_shape_bias_tradeoff.pdf")

#* Figure 2C & 2D | benchmarks acorss models and categories
# Go Neuro-vs-AI repo to test the all benchmark with others (includes evaluation and plotting)
# cd /home/student/l/lzejin/codebase/neuro-vs-ai
# python scripts/evaluate.py

#* Figure 2E | examples

################################################################################
## Figure 3 Understanding how the model become shape bias
################################################################################
#TODO
#* Figure 3A | Shape bias and shap/ texture choices changes across time


#* Figure 3B | Ablation studies


#* Figure 3C | Visualization of feature attribution
#? just use the models under /share/klab/datasets/texture2shape_projects/share/TVD/main_models

################################################################################
## Figure 4 Degradation robustness and adversarial robustness
################################################################################
#* Figure 4A | Degradation robustness
# Face & objects under blur degradations
model_names = ['Face_baseline_no_blur_seed_1',  
               'Face_DVD-B_no_blur_seed_2', 'Face_DVD-SSS_mpe0-5_no_blur_seed_1', #'Face_baseline_no_blur_seed_2',  'Face_DVD-B_no_blur_seed_2', 'Face_baseline_no_blur_seed_3',  'Face_DVD-B_no_blur_seed_3', #'Face_DVD-B_seed_1', 'Face_DVD-SS_seed_1',
               'ecoset_adult_no_blur',  'ecoset_DVD_B_no_blur',  'ecoset_DVD_S_no_blur', 'ecoset_DVD_P_no_blur', 
               'ecoset_DVD_SS_no_blur', 'ecoset_DVD_PP_no_blur'][:2]#[3:5]
dataset_name = ['facescrub','ecoset_square256'][0]
epoch = 'best'
degradation_results_dir = './results/degradation_robustness/blur_degradtion/' #  os.path.join(DEGRADATION_RESULTS_SAVE_DIR, 'blur_degradtion',
model_acc_dict = {}
for model_name in model_names:
    csv_path = os.path.join(degradation_results_dir, f"{dataset_name}_blur_degradtion_{model_name}_epoch{epoch}.csv")
    df = pd.read_csv(csv_path)
    # Assuming the CSV has a column named 'accuracy'
    # import pdb;pdb.set_trace()
    model_acc_dict[model_name] = df['accuracy'].tolist()[1:] # start from signma 1

# import pdb;pdb.set_trace()

plotter = ModelAccuracyPlotter(
    save_dir='./results/sigma_blur_attack',
    human_data_path='./data/blur_degradation_human_experiment/human_accuracy.mat'
)
plotter.plot_model_accuracy(new_models=model_acc_dict, dataset_name= dataset_name, absolute_acc=True,
                            save_name = f'human_and_models_{dataset_name}_vs_sigma_blur_attack_1st_July_all')


#* Figure 4B | Adversarial robustness

################################################################################
## Figure 5 Generalization to different models and datasets
################################################################################
#* Figure 5A | Generalization to different models
#* Figure 5B | Generalization to different datasets
# Done: directly run in neuro-vs-ai repo

################################################################################
## Figure X Extra analyses
################################################################################
#* First extract the representations
# Done

#* Ayzen skeleton RSA

#* Martin Herbat Triple judgement