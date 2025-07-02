#!/usr/bin/env python
"""
IllusionBench – t-SNE visualisation pipeline
Author: <you>
Date: 2025-06-08
"""

# ───────────────────── imports ─────────────────────
from __future__ import annotations
import os, io, csv, json, argparse, itertools, collections, datetime as dt
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
# --- helper: load model & safe feature extractor ---------------------------
from contextlib import suppress
import torch.nn as nn
from torchvision.models.feature_extraction import create_feature_extractor


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datasets import load_dataset                                   

      
# import torchvision.models as zoomodels
# from neuroai.models import pytorch_model_zoo, tensorflow_model_zoo, list_models
# ── robust loader ───────────────────────────────────────────────────────────
import torchvision.models as tvm
from torchvision.models.feature_extraction import create_feature_extractor
import torch.nn as nn, torch, contextlib
from neuroai.models import pytorch_model_zoo, tensorflow_model_zoo, list_models

# ───────────────────── CLI args ─────────────────────
parser = argparse.ArgumentParser(description="Run t-SNE over IllusionBench embeddings.")
parser.add_argument("--batch", type=int, default=64)
parser.add_argument("--perplexity", type=float, default=30.0)
parser.add_argument("--out", type=str, default="./results/plots/results_tsne")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
args = parser.parse_args() if __name__ == "__main__" else argparse.Namespace(**{})

# ───────────────────── paths ─────────────────────
ROOT = Path(args.out)
EMB_DIR = ROOT / "embeddings"
PLOT_DIR = ROOT / "plots"
CSV_DIR  = ROOT / "csv"

for d in (EMB_DIR, PLOT_DIR, CSV_DIR): d.mkdir(parents=True, exist_ok=True)

# ───────────────────── dataset ─────────────────────
RESOLUTION = {
    "transformer_L16_IN21K": 384,
    "transformer_B16_IN21K": 224,
    # add more if needed
}
preprocess = transforms.Compose([                                        
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.Resize(int(384*256/224)),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
])


class IllusionDataset(Dataset):
    """Returns (tensor, shape, scene, filename)."""
    def __init__(self, hf_split, transform=None):
        self.data = hf_split
        self.transform = transform
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        rec   = self.data[idx]
        fname = rec["image_name"]
        image = rec["image"]
        shape, scene, *_ = fname.split("-") + [""]
        if isinstance(image, dict):       # array or bytes
            arr = image.get("array")
            bys = image.get("bytes")
            image = Image.fromarray(np.array(arr)) if arr is not None else Image.open(io.BytesIO(bys))
        if self.transform: image = self.transform(image)
        return image, shape, scene, fname

def collate(batch):
    ims, shapes, scenes, names = zip(*batch)
    return torch.stack(ims), list(shapes), list(scenes), list(names)

print("Downloading IllusionBench…")
ds = load_dataset("arshiahemmat/IllusionBench")["Illusion_IN"]        #  [oai_citation:9‡huggingface.co](https://huggingface.co/datasets/arshiahemmat/IllusionBench?utm_source=chatgpt.com)
loader = DataLoader(IllusionDataset(ds, preprocess),
                    batch_size=args.batch,
                    shuffle=False,
                    num_workers=4,
                    collate_fn=collate)

def _find_target_layer(model: nn.Module):
    """Heuristic: last pool/flatten layer, else the penultimate Linear."""
    candidates = [n for n, _ in model.named_modules()
                  if n.endswith(("avgpool", "flatten", "pre_logits", "globalpool"))]
    return candidates[-1] if candidates else list(model.named_modules())[-2][0]


# ---------- helpers ---------------------------------------------------------
def _pick_node(net: nn.Module) -> str:                # where to grab features
    keys = [n for n,_ in net.named_modules()
            if n.endswith(("pre_logits","avgpool","flatten","globalpool","fc_norm","norm"))]
    return keys[-1] if keys else list(net.named_modules())[-2][0]

def _hookify(net: nn.Module, node: str) -> nn.Module: # forward-hook wrapper
    buf = {}
    dict(net.named_modules())[node].register_forward_hook(
        lambda _,__,out: buf.setdefault("feat", out))

    class Wrap(nn.Module):
        def __init__(self,m): super().__init__(); self.m=m
        def forward(self,x): buf.clear(); _=self.m(x); return {"feat": buf["feat"]}
    return Wrap(net)


def load_model(name: str, device="cuda", **kw):
    low = name.lower()

    # 0 CLIP family – use encode_image(), skip FX entirely
    if "clip" in low:
        net = (tvm.__dict__.get(name) or
               getattr(pytorch_model_zoo, name)(model_name=name, **kw).model)
        if isinstance(net, nn.DataParallel): net = net.module
        net.eval().to(device)
        class ClipWrap(nn.Module):
            def __init__(self,n): super().__init__(); self.n=n
            def forward(self,x): return {"feat": self.n.encode_image(x)}
        return ClipWrap(net), "pytorch"

    # 1 TorchVision plain
    if name in tvm.__dict__:
        net, fw = tvm.__dict__[name](**kw), "pytorch"

    # 2 NeuroAI Zoo – Pytorch
    elif name in list_models("pytorch"):
        net, fw = getattr(pytorch_model_zoo, name)(model_name=name, **kw).model, "pytorch"

    # 3 NeuroAI Zoo – TensorFlow
    elif name in list_models("tensorflow"):
        net, fw = getattr(tensorflow_model_zoo, name)(model_name=name, **kw).model, "tensorflow"
    else:
        raise NameError(f"{name} not supported by TorchVision or NeuroAI-Zoo.")

    # unwrap DP & move to device
    if isinstance(net, nn.DataParallel): net = net.module
    net.eval().to(device)

    # 4 Vision-Transformers & other dynamic nets → go straight to hook
    if any(k in low for k in ["vit", "transformer_", "swin", "dino_vit"]):
        return _hookify(net, _pick_node(net)), fw

    # 5 default: try FX, else fall back to hook
    try:
        return create_feature_extractor(
            net, {_pick_node(net): "feat"},
            tracer_kwargs={"leaf_modules": [nn.DataParallel]}  # must be list!
        ), fw
    except (torch.fx.proxy.TraceError, NotImplementedError, TypeError, ValueError):
        return _hookify(net, _pick_node(net)), fw

# ───────────────────── helper: feature writer ─────────────────────
def compute_embeddings(model_name: str):
    """Forward pass entire dataset once, dump features + meta to disk."""
    npy_file = EMB_DIR / f"{model_name}.npy"
    meta_file = EMB_DIR / f"{model_name}_meta.json"
    if npy_file.exists() and meta_file.exists():
        print(f"Found cached embeddings for {model_name}")
        return np.load(npy_file), json.load(open(meta_file))
    extractor, _ = load_model(model_name)
    feats, shapes, scenes, names = [], [], [], []
    for imgs, shp, scn, nms in tqdm(loader, desc=f"{model_name}"):
        imgs = imgs.to(args.device)
        with torch.no_grad():
            out = extractor(imgs)["feat"]
        out = out.flatten(1).cpu()
        feats.append(out)
        shapes += shp
        scenes += scn
        names  += nms
    feats = torch.cat(feats).numpy()
    np.save(npy_file, feats)
    json.dump({"shape": shapes, "scene": scenes, "name": names}, open(meta_file, "w"))
    return feats, {"shape": shapes, "scene": scenes, "name": names}

# ───────────────────── helper: TSNE + scatter ─────────────────────
def tsne_and_plot(model_name: str):
    vecs, meta = compute_embeddings(model_name)
    tsne = TSNE(n_components=2,
                init="pca",
                perplexity=args.perplexity,
                learning_rate="auto",
                random_state=42,
                n_jobs=4,
                verbose=1)
    embed = tsne.fit_transform(vecs)                      # scikit-learn TSNE   [oai_citation:10‡scikit-learn.org](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html?utm_source=chatgpt.com)
    # Plot by shape and by scene
    for tag in ("shape", "scene"):
        fig, ax = plt.subplots(figsize=(7,7))
        labels = meta[tag]
        uniq   = sorted(set(labels))
        palette = plt.get_cmap("tab20").colors            # uses Matplotlib categorical palette   [oai_citation:11‡matplotlib.org](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.scatter.html?utm_source=chatgpt.com)
        color_map = {lab: palette[i % len(palette)] for i, lab in enumerate(uniq)}
        for lab in uniq:
            pts = embed[np.array(labels) == lab]
            ax.scatter(pts[:,0], pts[:,1],
                       label=lab,
                       s=14,
                       alpha=.7,
                       linewidths=.3)
        ax.set_title(f"{model_name} – t-SNE coloured by {tag}")
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(ncol=2, fontsize=8, markerscale=0.6, frameon=False)
        fig.tight_layout()
        out = PLOT_DIR / f"{model_name}_{tag}.pdf"
        fig.savefig(out, dpi=300)
        out_png = PLOT_DIR / f"{model_name}_{tag}.png"
        fig.savefig(out_png, dpi=300)
        plt.close(fig)
        print("✓", out)

# ───────────────────── helper: CSV summary ─────────────────────
def summarise(model_name: str, meta: dict):
    shapes = meta["shape"]; scenes = meta["scene"]
    shape_hits = sum(1 for s,p in zip(shapes, meta["name"]) if s and s.lower() in p.lower())
    scene_hits = sum(1 for s,p in zip(scenes, meta["name"]) if s and s.lower() in p.lower())
    tot = len(shapes)
    row = [model_name, tot, shape_hits, scene_hits,
           f"{shape_hits/tot:.2%}", f"{scene_hits/tot:.2%}"]
    csv_path = CSV_DIR / "summary.csv"
    hdr = ["model", "N", "shape_hits", "scene_hits", "shape%", "scene%"]
    write_hdr = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_hdr: w.writerow(hdr)
        w.writerow(row)

# ───────────────────── main loop ─────────────────────
NETWORKS = [
    'resnet50_baseline_imagenet',
    'resnet50_DVD_S_imagenet',
    'resnet50_DVD_B_imagenet',
    'resnet50_DVD_P_imagenet',
    'resnet50_DVD_PP_imagenet',

    'resnet50_trained_on_SIN',
    'resnet50_trained_on_SIN_and_IN',
    'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN',

    "alexnet", "vgg11_bn", 
    "clipRN50",
    "clip",
    "vgg13_bn", "vgg16_bn", "vgg19_bn",
    "squeezenet1_0", "squeezenet1_1", "densenet121", "densenet169",
    "inception_v3", "resnet18", "resnet34", "resnet50", "resnet101",
    "resnet152", "shufflenet_v2_x0_5", "mobilenet_v2",
    "resnext50_32x4d", "resnext101_32x8d",
    "wide_resnet50_2", "wide_resnet101_2",
    "mnasnet0_5", "mnasnet1_0",
    "simclr_resnet50x1", "vit_small_patch16_224",
    "vit_base_patch16_224", "vit_large_patch16_224",
    "ResNeXt101_32x16d_swsl", "resnet50_swsl",
    "BiTM_resnetv2_152x4", "BiTM_resnetv2_152x2", "BiTM_resnetv2_101x3",
    "BiTM_resnetv2_101x1", "BiTM_resnetv2_50x3", "BiTM_resnetv2_50x1",
    "efficientnet_l2_noisy_student_475",
    "transformer_L16_IN21K", # resize to 384
    "transformer_B16_IN21K",
    "InsDis",
    

    # "MoCo", "PIRL", "MoCoV2",  # "InfoMin", #* bug for running InfoMin somehow 
]

if __name__ == "__main__":
    for net in NETWORKS:
        print(f"\n—— {net} ——")
        tsne_and_plot(net)
        _, meta = compute_embeddings(net)   # already cached
        summarise(net, meta)

    # optional: build an HTML gallery once everything finishes
    gallery = ROOT / "index.html"
    with open(gallery, "w") as g:
        g.write("<h1>IllusionBench – t-SNE gallery</h1>\n")
        for png in sorted(PLOT_DIR.glob("*.png")):
            g.write(f'<div style="display:inline-block;text-align:center;margin:4px">'
                    f'<img src="plots/{png.name}" width="340"><br>{png.stem}</div>\n')
    # print("Open", gallery.relative_to(Path.cwd()))
    print("Open", gallery.resolve().relative_to(Path.cwd().resolve()))