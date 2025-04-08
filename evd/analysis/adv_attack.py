################################################################################
#              ADVERSARIAL ATTACK EVALUATION (OPTIONAL ROBUSTNESS TEST)
################################################################################

import os
import csv
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

import torch
import foolbox
from foolbox import PyTorchModel
from foolbox.attacks import (
    LinfFastGradientAttack,
    L2FastGradientAttack,
    LinfPGD,
    L2AdditiveGaussianNoiseAttack,
    L2AdditiveUniformNoiseAttack,
    SaltAndPepperNoiseAttack,
    LinearSearchContrastReductionAttack,
    HopSkipJumpAttack,
    EADAttack,
    InversionAttack,
    PointwiseAttack,
    dataset_attack,
    GenAttack,
    LinfAdamPGD,
    SpatialAttack,
    DDNAttack
)
from foolbox.distances import linf

from evd.analysis.plot import curve_plot, bar_plot

############################################################################
#                   ADVERSARIAL ATTACK HELPER FUNCTIONS
############################################################################

def adv_attack_analysis(net, test_loader, attacks, epsilons_dict, 
                        model_name='default_model_name', epoch='default', device='cpu'):
    """
    Evaluates adversarial robustness using Foolbox.
    The function computes the clean (baseline) accuracy once over the test set,
    then for each attack computes adversarial accuracies (for each epsilon).
    The results are collected into a Pandas DataFrame (and saved as CSV),
    and both bar and curve plots are generated.
    The clean accuracy is treated as epsilon=0.

    Args:
        net (torch.nn.Module): The model to evaluate.
        test_loader (DataLoader): DataLoader with test/validation images.
        attacks (dict): Mapping from attack_name -> attack_function.
        epsilons_dict (dict): Mapping from attack_name -> list of epsilons/parameters.
        model_name (str): Descriptor for the model (used for file naming).
        epoch (int or str): Current epoch (used for logging).
        device (str): 'cpu' or 'cuda'.
        
    Returns:
        pandas.DataFrame: DataFrame containing adversarial accuracies for each attack.
    """
    # Ensure model is on the proper device and in evaluation mode.
    net = net.to(device)
    net.eval()
    
    results_list = []  # Will hold a dictionary for each result.
    log_path = os.path.join('./results', 'adv_robustness', model_name)
    os.makedirs(log_path, exist_ok=True)
    
    # Convert test_loader to a list so that we can iterate over it multiple times.
    test_data = list(test_loader)
    
    # --- Compute global clean accuracy (treated as epsilon=0) ---
    print(f'Starting clean accuracy evaluation...')
    total_samples, clean_correct = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(test_data):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            batch_size = images.size(0)
            total_samples += batch_size
            if device == 'cuda':
                with torch.cuda.amp.autocast():
                    preds = net(images).argmax(dim=1)
            else:
                preds = net(images).argmax(dim=1)
            clean_correct += (preds == labels).sum().item()
    global_clean_accuracy = clean_correct / total_samples
    print(f"Global Clean Accuracy: {global_clean_accuracy:.4f}")
    
    # --- Wrap the model with Foolbox for attack evaluation ---
    fmodel = PyTorchModel(net, bounds=(0, 1), preprocessing={})
    
    # --- Evaluate each adversarial attack ---
    for attack_name, attack_fn in attacks.items():
        print(f'[Adversarial] Attack: {attack_name}')
        epsilons = epsilons_dict.get(attack_name, [None])
        adv_correct = np.zeros(len(epsilons))
        
        # Loop over test batches.
        with torch.no_grad():
            for images, labels in tqdm(test_data):
                # import pdb;pdb.set_trace()
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long()
                
                # Run the attack function; if epsilons are provided, pass them.
                if epsilons[0] is not None:
                    if device == 'cuda':
                        with torch.cuda.amp.autocast():
                            adv_results = attack_fn(fmodel, images, labels, epsilons=epsilons)
                    else:
                        adv_results = attack_fn(fmodel, images, labels, epsilons=epsilons)
                else:
                    adv_results = attack_fn(fmodel, images, labels)
                    
                # The third element is a Boolean tensor indicating adversarial examples.
                is_adv = adv_results[2]
                is_adv_np = is_adv.cpu().numpy()
                if is_adv_np.ndim == 1:
                    is_adv_np = is_adv_np[:, np.newaxis]
                # Count the number of non-adversarial (i.e., correct) predictions.
                adv_correct += (~is_adv_np).astype(np.float32).sum(axis=1)
        
        adv_accuracies = adv_correct / total_samples
        
        # Save the clean baseline as epsilon=0 if not already provided.
        if 0 not in epsilons and 0.0 not in epsilons:
            results_list.append({
                'Model': model_name,
                'Attack': attack_name,
                'Epsilon': 0,
                'Accuracy': global_clean_accuracy,
                'Epoch': epoch
            })
        print(f'Clean accuracy (epsilon=0): {global_clean_accuracy:.4f}')
        
        for eps, acc in zip(epsilons, adv_accuracies):
            print(f'  Epsilon: {eps}, Adv Accuracy: {acc:.4f}')
            results_list.append({
                'Model': model_name,
                'Attack': attack_name,
                'Epsilon': eps,
                'Accuracy': acc,
                'Epoch': epoch
            })
        
        # Prepare data for plotting. If the clean baseline is missing, add it.
        plot_epsilons = list(epsilons)
        plot_accuracies = list(adv_accuracies)
        if 0 not in epsilons and 0.0 not in epsilons:
            plot_epsilons = [0] + plot_epsilons
            plot_accuracies = [global_clean_accuracy] + plot_accuracies
        tick_labels = [str(e) for e in plot_epsilons]
        plot_title = f'{attack_name} Adv Accuracy Ep{epoch}'
        
        bar_plot(plot_accuracies, tick_labels, 'Epsilon', 'Accuracy', plot_title,
                 os.path.join(log_path, f'{attack_name}_bar_ep{epoch}'))
        curve_plot(plot_accuracies, tick_labels, 'Epsilon', 'Accuracy', plot_title,
                    os.path.join(log_path, f'{attack_name}_curve_ep{epoch}'))
    
    # Combine all results into a DataFrame and save as CSV.
    df_results = pd.DataFrame(results_list)
    csv_save_path = os.path.join(log_path, f'adv_robustness_ep{epoch}.csv')
    df_results.to_csv(csv_save_path, index=False)
    print(f"Combined adversarial robustness results saved to {csv_save_path}")
    
    return df_results


def adv_spatial_attacks(net, hparams, test_loader, epoch, model_name, blurring_strategy,
                         attacks, spatial_attack_params, device='cpu'):
    """
    Evaluates adversarial robustness against spatial transformations.
    It iterates through different parameter values for each spatial attack,
    logs accuracies, saves CSV logs, and generates bar and curve plots.
    
    Args:
        net (torch.nn.Module): The model to evaluate.
        hparams (dict): Contains hyperparameters/metadata (e.g., dataset name).
        test_loader (DataLoader): DataLoader with test/validation images.
        epoch (int or str): Current epoch for logging.
        model_name (str): Model descriptor (used for file naming).
        blurring_strategy (str): Used for log folder naming.
        attacks (dict): Mapping from attack name to a spatial attack constructor.
        spatial_attack_params (dict): Mapping from attack name to list of parameter values.
        device (str): 'cpu' or 'cuda'.
        
    Returns:
        dict: A dictionary containing spatial adversarial accuracies for each attack.
    """
    net.eval()
    output = {}
    log_path = os.path.join('./results', 'adv_robustness', model_name)
    os.makedirs(log_path, exist_ok=True)

    # Wrap the network with Foolbox
    fmodel = PyTorchModel(net, bounds=(0, 1), preprocessing={})
    
    # Loop over spatial attacks that have parameter variations
    for attack_name, attack_constructor in attacks.items():
        if attack_name not in spatial_attack_params:
            continue  # Skip non-spatial attacks
        
        print(f'[Adversarial: Spatial] Attack: {attack_name}')
        param_values = spatial_attack_params[attack_name]
        total_samples, clean_correct = 0, 0
        adv_correct = np.zeros(len(param_values))
        
        # Evaluate over the test set
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device).long()
            batch_size = images.size(0)
            total_samples += batch_size
            
            # Compute clean accuracy
            with torch.no_grad():
                preds = net(images).argmax(dim=1)
                clean_correct += (preds == labels).sum().item()
            
            # Apply the spatial attack for each parameter value
            for i, param in enumerate(param_values):
                # Instantiate a new attack instance based on parameter type
                if attack_name == 'SpatialAttack_rotation':
                    current_attack = attack_constructor(max_rotation=param)
                elif attack_name == 'SpatialAttack_translation':
                    current_attack = attack_constructor(max_translation=param)
                elif attack_name == 'SpatialAttack_scaling':
                    current_attack = attack_constructor(max_rotation=0, max_translation=0)
                    current_attack.scaling = param  # Adjust scaling factor if supported
                else:
                    current_attack = attack_constructor  # Fallback: use the passed instance
                
                _, _, is_adv = current_attack(fmodel, images, labels)
                is_adv_np = is_adv.cpu().numpy()
                # Sum correct (i.e., non-adversarial) predictions over the batch
                adv_correct[i] += (~is_adv_np).sum()
        
        clean_accuracy = clean_correct / total_samples
        adv_accuracies = adv_correct / total_samples
        
        print(f'Clean accuracy: {clean_accuracy:.4f}')
        for param, acc in zip(param_values, adv_accuracies):
            print(f'  Param: {param}, Adv Accuracy: {acc:.4f}')
        
        # Save CSV log for this spatial attack
        csv_path = os.path.join(log_path, f'{attack_name}_adv_robustness_log_ep{epoch}.csv')
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['Model', 'Attack', 'Parameter', 'Accuracy'])
            writer.writeheader()
            for param, acc in zip(param_values, adv_accuracies):
                writer.writerow({
                    'Model': model_name,
                    'Attack': attack_name,
                    'Parameter': param,
                    'Accuracy': acc
                })
        
        # Generate both bar and curve plots for the adversarial accuracies
        tick_labels = [str(p) for p in param_values]
        plot_title = f'{attack_name} Adv Accuracy Ep{epoch}'
        bar_plot(adv_accuracies, tick_labels, 'Parameter', 'Accuracy', plot_title,
                 os.path.join(log_path, f'{attack_name}_bar_ep{epoch}'))
        curve_plot(adv_accuracies, tick_labels, 'Parameter', 'Accuracy', plot_title,
                   os.path.join(log_path, f'{attack_name}_curve_ep{epoch}'))
        
        output[f'adversarial_accuracies_{attack_name}'] = {
            str(param): float(acc) for param, acc in zip(param_values, adv_accuracies)
        }
    
    np.save(os.path.join(log_path, f'adv_spatial_attacks_ep{epoch}.npy'), output)
    return {'adversarial_attacks': output}

############################################################################
#           EXAMPLE ATTACK/EPSILON DEFINITIONS (CAN BE CUSTOMIZED)
############################################################################

EPSILONS_DICT = {
    'LinfFastGradientAttack': [0.0, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 0.8],
    'L2FastGradientAttack': [0.0, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 0.8],
    'LinfPGD': [0.0, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 0.8],
    'LinfAdamPGD': [0.0, 0.0001, 0.001, 0.01, 0.03, 0.1, 0.3, 0.5, 0.8],
    'L2AdditiveGaussianNoiseAttack': [0.0, 10.0, 20.0, 50.0, 80.0, 100.0, 150, 200],
    'L2AdditiveUniformNoiseAttack': [0.0, 10.0, 20.0, 50.0, 80.0, 100.0, 150, 200],
    'SaltAndPepperNoiseAttack': [0.0, 10.0, 20.0, 50.0, 80.0, 100.0, 150, 200],
    'SpatialAttack_rotation': [0, 15, 30, 45, 60],
    'SpatialAttack_translation': [0, 15, 30, 45, 60],
    'SpatialAttack_scaling': [1.0, 1.2, 1.5, 1.8, 2.0],
    'InversionAttack': [None],
    'LinearSearchContrastReductionAttack': [None],
    'DDNAttack': [None],
    'HopSkipJumpAttack': [10, 50, 100, 500, 1000],
    'GenAttack': [None],
    'PointwiseAttack': [None],
}

ATTACKS = {
            # #* Gradient-Based Attacks
            'LinfFastGradientAttack': LinfFastGradientAttack(),
            'L2FastGradientAttack': L2FastGradientAttack(),
            'LinfPGD': LinfPGD(),
            'LinfAdamPGD': LinfAdamPGD(),
            # 'CarliniWagnerL2': CarliniWagnerL2Attack(), # not available in curret verion foolbox
            # 'DeepFool': DeepFoolAttack(), # not available in curret verion foolbox

            #* Noise-Based Attacks
            'L2AdditiveGaussianNoiseAttack': L2AdditiveGaussianNoiseAttack(),
            'L2AdditiveUniformNoiseAttack': L2AdditiveUniformNoiseAttack(),
            'SaltAndPepperNoiseAttack': SaltAndPepperNoiseAttack(),

            # Transformation-Based Attacks
            'SpatialAttack_rotation': SpatialAttack(),
            'SpatialAttack_translation': SpatialAttack(), 
            'SpatialAttack_scaling': SpatialAttack(), 
            'InversionAttack': InversionAttack(),
            'LinearSearchContrastReductionAttack': LinearSearchContrastReductionAttack(),

            #* Optimization-Based Attacks
            #'EADAttack': EADAttack(),
            'DDNAttack': DDNAttack(),
            # 'UniversalPerturbation': UniversalPerturbation(),  not available in curret verion foolbox # May require custom implementation

            #* Query-Based and Decision-Based Attacks
            'HopSkipJumpAttack': HopSkipJumpAttack(),
            #'BoundaryAttack': BoundaryAttack(),
            #'LocalSearchAttack': LocalSearchAttack(),
            'GenAttack': GenAttack(),

            #* Iterative Adaptive Attacks
            'PointwiseAttack': PointwiseAttack(),
        }


if __name__ == '__main__':
    selected_adv_attack_names = [
        'L2AdditiveGaussianNoiseAttack',
        'L2AdditiveUniformNoiseAttack',
        'SaltAndPepperNoiseAttack'
    ]
    selected_attacks = {k: v for k, v in ATTACKS.items() if k in selected_adv_attack_names}

    #
    # Then, in your main epoch loop:
    adv_attack_analysis(model, your_hparams_dict, test_loader, epoch,
                        model_name='resnet50',
                        blurring_strategy='none',
                        attacks=attacks,
                        epsilons_dict=epsilons_dict,
                        device=device)
    
    # For spatial attacks specifically:
    adv_spatial_attacks(model, your_hparams_dict, test_loader, epoch,
                        model_name='resnet50',
                        blurring_strategy='none',
                        attacks=attacks,
                        spatial_attack_params={'SpatialAttack_rotation': [0, 15, 30]},
                        device=device)