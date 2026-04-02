import torch
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from typing import List, Optional, Dict
from torch.utils.data import DataLoader
from tqdm import tqdm

def extract_acts_and_create_rdm(
    model: torch.nn.Module,
    loader: DataLoader,               # yields batches (img, ...) in fixed order
    layers: Optional[List[str]] = None,
    device: str = 'cuda',
    distance_metric: str = 'correlation',  # 'correlation' or 'euclidean'
) -> Dict[str, np.ndarray]:
    """
    Run all images in `loader` through `model`, hook the specified layers,
    and compute a Representational Dissimilarity Matrix (RDM) for each layer
    using correlation distance (dropping any unit with zero variance).

    Args:
        model:      a torch Module (e.g. ResNet50) already on CPU/GPU
        loader:     DataLoader yielding (images, ...) with shuffle=False
        layers:     list of named_modules() keys to hook; defaults to
                    ['layer1','layer2','layer3','layer4'] for ResNet50
        device:     'cuda' or 'cpu'
        distance_metric: 'correlation' or 'euclidean'

    Returns:
        { layer_name: rdm_matrix } where each rdm_matrix is an (N×N) ndarray
    """
    if layers is None:
        layers = ['layer1', 'layer2', 'layer3', 'layer4']

    # move model to device, set eval
    model.to(device).eval()

    # prepare storage for each layer's activations
    activations: Dict[str, List[torch.Tensor]] = {ln: [] for ln in layers}

    # register hooks
    hooks = []
    modules = dict(model.named_modules())
    for ln in layers:
        if ln not in modules:
            raise ValueError(f"Layer '{ln}' not found in model.")
        def make_hook(name):
            def hook(module, inp, out):
                activations[name].append(out.detach().cpu())
            return hook
        hooks.append(modules[ln].register_forward_hook(make_hook(ln)))

    # forward all batches
    with torch.no_grad():
        for batch in loader:
            imgs = batch[0].to(device)
            model(imgs)

    # import pdb; pdb.set_trace()

    # remove hooks
    for h in hooks:
        h.remove()

    # compute RDM per layer
    rdms: Dict[str, np.ndarray] = {}
    for ln, acts_list in tqdm(activations.items()):
        feats = torch.cat(acts_list, dim=0)      # [N, ...]
        N = feats.shape[0]
        feats = feats.view(N, -1)               # flatten
        stds = feats.std(dim=0)
        feats = feats[:, stds > 0]              # drop constant channels
        arr = feats.numpy()
        d = pdist(arr, metric=distance_metric)
        rdms[ln] = squareform(d)                # (N×N) RDM

    return rdms



# def extract_acts_and_create_rdm(
#     model: torch.nn.Module,
#     loader: DataLoader,               # yields (imgs, …) with shuffle=False
#     layers: Optional[List[str]] = None,
#     device: str = 'cuda',
#     pca_dim: int = 1024                # target PCA dimensionality
# ) -> Dict[str, np.ndarray]:
#     """
#     Run the loader through model, hook `layers`, compute an RDM per layer in two steps:
#       1) flatten & drop constant units
#       2) reduce to `pca_dim` via PCA
#       3) compute correlation-distance RDM
    
#     Args:
#         model:    a torch Module (e.g. ResNet50)
#         loader:   DataLoader yielding (images, …), shuffle=False
#         layers:   list of named_modules() keys to hook; defaults to
#                   ['layer1','layer2','layer3','layer4']
#         device:   'cuda' or 'cpu'
#         pca_dim:  number of PCA components to keep before distance
        
#     Returns:
#         { layer_name: (N×N) numpy RDM }
#     """
#     if layers is None:
#         layers = ['layer1', 'layer2', 'layer3', 'layer4']

#     model = model.to(device).eval()
#     activations: Dict[str, List[torch.Tensor]] = {ln: [] for ln in layers}

#     # Register hooks
#     hooks = []
#     modules = dict(model.named_modules())
#     for ln in layers:
#         if ln not in modules:
#             raise ValueError(f"Layer '{ln}' not found in model.")
#         def make_hook(name):
#             def hook(module, inp, out):
#                 activations[name].append(out.detach().cpu())
#             return hook
#         hooks.append(modules[ln].register_forward_hook(make_hook(ln)))

#     # Forward pass
#     with torch.no_grad():
#         for batch in loader:
#             imgs = batch[0].to(device)
#             model(imgs)

#     # Remove hooks
#     for h in hooks:
#         h.remove()

#     # Compute and return RDMs
#     rdms: Dict[str, np.ndarray] = {}
#     for ln, acts_list in tqdm(activations.items(), desc="Layers"):
#         # 1) Concatenate & flatten
#         feats = torch.cat(acts_list, dim=0)    # [N, C, H, W] or [N, features]
#         N = feats.shape[0]
#         feats = feats.view(N, -1)             # [N, D]
#         stds = feats.std(dim=0)
#         feats = feats[:, stds > 0]            # drop constant dims
#         feats_np = feats.numpy()

#         # 2) PCA to reduce to pca_dim
#         n_components = min(pca_dim, feats_np.shape[1])
#         pca = PCA(n_components=n_components, svd_solver='randomized', whiten=False)
#         feats_reduced = pca.fit_transform(feats_np)  # [N, pca_dim]

#         # 3) Correlation-distance RDM
#         d = pdist(feats_reduced, metric='correlation')
#         rdms[ln] = squareform(d)               # (N×N)

#     return rdms