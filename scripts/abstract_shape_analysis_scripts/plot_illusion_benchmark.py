
import os
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
from pathlib import Path
from typing import Dict, Iterable, List, Union
from collections import defaultdict

from matplotlib.ticker import FuncFormatter
# -- helper that turns any tick value into its absolute value string ----
abs_fmt = FuncFormatter(lambda y, _: f"{abs(int(y))}")   # e.g. -40 → "40"

from neuroai.plotting.colors import *  # uses black, bright_green, teal_green, deep_blue

# Style setup
plt.style.use('nature')
plt.rcParams.update({
    'font.size': 6,
    'axes.titlesize': 7,
    'axes.labelsize': 6,
    'xtick.labelsize': 5,
    'ytick.labelsize': 5,
    'legend.fontsize': 5
})

# ----------------------------------------------------------------------
# 1. Load the 1 000 ImageNet class names
# ----------------------------------------------------------------------

IMAGENET_LABELS_TXT = Path("./data/imagenet_classes.txt")

with IMAGENET_LABELS_TXT.open() as f:
    imagenet_labels: List[str] = [line.strip() for line in f]

NUM_IMAGENET_CLASSES = len(imagenet_labels)
assert NUM_IMAGENET_CLASSES in (1000, 100), (
    f"Expected 1000 (or 100) labels, found {NUM_IMAGENET_CLASSES}"
)

# ----------------------------------------------------------------------
# 2. Small helper – converts any iterable that may mix ints and str
# ----------------------------------------------------------------------

def to_labels(seq: Iterable[Union[int, str]]) -> List[str]:
    """
    Replace every *int* with its ImageNet name; leave strings untouched.

    The order and duplicates of the incoming sequence are preserved.
    """
    out: List[str] = []
    for item in seq:
        if isinstance(item, int):
            if not (0 <= item < NUM_IMAGENET_CLASSES):
                raise ValueError(f"ID {item} is outside the 0–{NUM_IMAGENET_CLASSES-1} range.")
            out.append(imagenet_labels[item])
        else:  # already a string (some lists mix IDs and literal words)
            out.append(str(item))
    return out

# ----------------------------------------------------------------------
# 3. Raw ID collections (verbatim copy-paste from the question)
# ----------------------------------------------------------------------

super_shape_classes_ids = {'airplane': [404, 895, 405, 812, 908, 327, 723, 913, 877, 744, 769, 421, 777, 146, 667, 718, 427, 726],
 'bicycle': [778, 605, 915, 919, 605, 610, 548, 915, 610, 986, 716, 883, 902, 493, 723, 363, 'bicycle'],
 'bird': [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 'snipe', 114, 115, 79, 327, 124, 145, 667, 87],
 'bottle': [681, 997, 988, 998, 730, 148, 742, 1001, 999, 989, 671, 148, 998, 823, 509, 120, 132, 148, 960, 918, 500, 675, 679, 983, 712, 987],
 'car': [901, 182, 903, 902, 738, 590, 145, 118, 783, 609, 708, 897, 704, 736, 712, 148, 722, 781, 808, 889, 873, 821, 873],
 'cat': [367, 368, 369, 370, 371, 372, 373, 374, 375, 376, 377, 378, 379, 518, 371, 476, 518, 117, 145, 47, 110, 500],
 'dog': [123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 147, 23, 161, 137, 119, 47],
 'sailboat': [763, 914, 728, 803, 756, 712, 995, 812, 886, 642, 890, 934, 974, 837, 969, 771, 399, 893, 990, 911, 887, 712],
 'dolphin': [187, 88, 168, 0, 1, 2, 3, 4, 5, 6, 63, 64, 65, 66, 67, 68, 69, 70, 71, 101, 102, 46, 173, 108, 135],
 'fork': [893, 661, 893, 784, 900, 909, 562, 593, 984, 548, 909, 991, 597, 691, 900, 421],
 'guitar': [323, 486, 46, 417, 405, 473, 323, 486],
 'mug': [505, 958, 247, 997, 984, 994, 539, 999, 671, 990, 917, 772, 365, 835, 395, 648, 988],
 'panda': [387, 386, 295, 294, 293, 292, 296, 653, 604, 971, 387, 509],
 'paper_clip': [760, 764, 790, 141, 633, 548, 421, 694, 704, 421, 548, 777, 694, 548, 624, 790],
 'scooter': [972, 610, 897, 650, 884, 915, 891, 650, 888, 440, 661],
 'teapot': [968, 301, 761, 285, 671, 839, 848, 982, 964, 1001, 152, 365, 917, 998, 985, 120, 535, 829, 712, 365, 917, 577, 190]}

super_scene_classes_ids = {'Bazaar_market': [950, 951, 953, 954, 949, 952, 955, 956, 957, 948, 936, 937, 938, 945, 943, 939, 941, 942, 940, 944, 959, 940, 984, 621, 988, 968, 989, 997, 961, 962, 963, 964, 766, 951, 952, 994, 995, 963, 884, 228, 585, 321, 696, 665, 742, 509, 764, 954, 509, 821, 785, 966, 968, 985, 919, 985, 874, 951, 102, 991, 305, 324, 946, 957, 970, 971, 972, 973, 990, 991, 992, 993, 993, 994, 995, 996, 138, 997, 998, 999, 358, 631, 134, 598, 598, 887, 990, 872, 948, 862, 843, 475, 862, 884],
 'City': [994, 998, 797, 746, 354, 835, 625, 629, 779, 984, 992, 883, 661, 893, 780, 300, 874, 959, 726, 735, 797, 723, 777, 786, 243, 74, 647, 783, 743, 792, 794, 812, 545, 607, 610, 961, 747, 845, 108, 359, 987, 970, 995, 820, 563, 554, 613, 850, 619, 691, 579, 712, 873, 899, 900, 921, 930, 940, 942, 975, 976, 979, 980, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 783, 784],
 'Medieval_Village': [826, 829, 830, 814, 824, 823, 833, 100, 523, 883, 633, 997, 282, 585, 845, 250, 204, 903, 760, 782, 726, 723, 777, 545, 521, 537, 153, 759, 437, 271, 818, 820, 144, 285, 434, 435, 820, 633, 575, 819, 957, 839, 901, 948],
 'Museum': [50, 68, 167, 777, 544, 600, 585, 473, 602, 416, 420, 757, 731, 745, 173, 105, 288, 421, 674, 784, 423, 739, 675, 651, 770, 679, 774, 421, 546, 544, 745, 679, 786, 789, 775, 768, 773, 772, 732, 542, 545, 672, 722, 721, 870, 777, 751, 769],
 'Time_square': [777, 723, 797, 726, 735, 741, 742, 746, 773, 780, 784, 789, 793, 797, 802, 803, 806, 807, 811, 819, 820, 819, 590, 462, 417, 623, 626, 664, 684, 685, 686, 723, 816, 821, 823, 873, 878, 885, 899, 900, 914, 923, 934],
 'Underwater_ruins': [2, 3, 4, 5, 6, 47, 48, 49, 327, 328, 329, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 44, 45, 46, 64, 65, 66, 67, 63, 64, 68, 69, 70, 71, 72, 63, 64, 65, 66, 67, 71, 72, 73, 74, 75, 76, 77],
 'Cloud': [210, 475, 272, 799, 725, 949, 780, 782, 792, 595, 598, 908],
 'Forest': [293, 294, 296, 300, 301, 302, 303, 304, 305, 306, 307, 314, 315, 316, 318, 319, 320, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344, 345, 346, 347, 348, 349, 350, 551, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 584, 585, 586, 587, 588, 589, 590, 591, 592, 593, 594, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637],
 'Ocean': [74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 64, 65, 74, 73, 59, 60, 62, 67, 88, 90, 94, 95, 96, 97, 99, 100, 109, 110, 111, 112, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 126],
 'Origami': [68, 69, 70, 71, 72, 73, 74, 462, 463, 474, 475, 452, 453, 574, 690, 228, 305, 347, 384, 387, 389, 407, 410, 411, 412, 413, 414, 416, 420, 421, 425, 427, 429, 433, 441, 442, 444, 448, 451, 458],
 'Sand_dune': [82, 63, 64, 55, 68, 74, 75, 76, 78, 79, 80, 81, 182, 183, 184, 186, 187, 188, 189, 190, 191, 192]}

original_scene_classes_ids = [ 970, 972, 973, 974, 975, 976, 977, 978, 979, 980, 415, 424, 425, 449, 454, 467, 483, 491, 492, 498, 521, 549, 550, 608, 634, 653, 668, 712, 721, 728, 743, 782, 865, 718, 460, 523, 830, 839, 888, 900, 873, 832, 708, 671, 829, 863, 495, 662, 912, 489, 717, 448, 410, 437, 442, 540, 634, 853, 832, 975, 858, 860, 669, 794, 861, 453, 564, 913, 819, 406, 736, 916, 854]

# ----------------------------------------------------------------------
# 4. Convert every ID collection into its label counterpart
# ----------------------------------------------------------------------
super_shape_class_labels: Dict[str, List[str]] = {  name: to_labels(ids) for name, ids in super_shape_classes_ids.items()}
super_scene_class_labels: Dict[str, List[str]] = { name: to_labels(ids) for name, ids in super_scene_classes_ids.items() }
original_scene_class_labels: List[str] = to_labels(original_scene_classes_ids)

# Some VLMs just say the scene name in their own way
VLMs_scene_classes = [ 'market','city', 'museum', 'skyscraper', 'forest', 'movie theater', 'cityscape']  

eleven_scene_class = ['Bazaar_market', 'City', 'Medieval_Village', 'Museum', 'Time_square', 'Underwater_ruins', 'Cloud', 'Forest', 'Ocean', 'Origami', 'Sand_dune']
sixteen_shape_categories = ['airplane', 'bicycle', 'bird', 'bottle', 'car', 'cat', 'dog', 'dolphin', 'fork', 'guitar', 'mug', 'panda', 'paper_clip', 'sailboat', 'scooter', 'teapot']



# ───────────────────────────  Scoring helpers  ─────────────────────────────── #

def is_shape_correct(pred: str, shape: str) -> bool:
    """True iff *pred* is a valid cue for *shape*."""
    return pred == shape or pred in super_shape_class_labels.get(shape, [])

def is_scene_correct(pred: str, scene: str) -> bool:
    """True iff *pred* is a valid cue for *scene*."""
    return (
        pred == scene
        or pred in original_scene_class_labels
        or pred in VLMs_scene_classes
        or pred in super_scene_class_labels.get(scene, [])
    )

def compute_recall(df: pd.DataFrame) -> tuple[float, float]:
    """Return (shape_recall %, scene_recall %)."""
    shape_hits = 0
    scene_hits = 0
    for _, row in df.iterrows():
        pred = row["predicted_label"]
        shape = row["shape"]
        scene = row["scene"]

        if is_shape_correct(pred, shape):
            shape_hits += 1
        if is_scene_correct(pred, scene):
            scene_hits += 1

    total = len(df)
    return (
        0.0 if total == 0 else 100 * shape_hits / total,
        0.0 if total == 0 else 100 * scene_hits / total,
    )

def compute_shape_accuracy(df: pd.DataFrame) -> dict[str, float]:
    """Return a dict mapping each shape category to its accuracy %."""
    correct_counts = defaultdict(int)
    total_counts = defaultdict(int)

    for _, row in df.iterrows():
        shape = row["shape"]
        pred = row["predicted_label"]

        if shape in sixteen_shape_categories:
            total_counts[shape] += 1
            if is_shape_correct(pred, shape):
                correct_counts[shape] += 1

    # Compute accuracy for each shape category
    accuracy_per_shape = {
        shape: 100 * correct_counts[shape] / total_counts[shape]
        if total_counts[shape] > 0 else 0.0
        for shape in sixteen_shape_categories
    }

    return accuracy_per_shape

# ───────────────────────────  Configuration  ──────────────────────────────── #
high_gray = "#A9A9A9"
middle_gray = "#C0C0C0"
COLOR_MAP = {
    "baseline": black,
    "DVD_S": bright_green,
    "DVD_B": teal_green,
    "DVD_P": deep_blue,
}

FILES: dict[str, str] = {
   
}
# List of model names to auto-fill
model_names = [

    #* CNN supervised
    "alexnet", 
    "vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn",
    "squeezenet1_0", "squeezenet1_1", 
    "densenet121", "densenet169", "densenet201",
    "inception_v3", "resnet18", "resnet34", 
    "resnet50", "resnet101", "resnet152",
    "shufflenet_v2_x0_5", "mobilenet_v2", 
    "resnext50_32x4d", "resnext101_32x8d",
    "wide_resnet50_2", "wide_resnet101_2", 
    "mnasnet0_5", "mnasnet1_0",

    'resnet50_swsl',
    'BiTM_resnetv2_152x4', 'BiTM_resnetv2_152x2', 'BiTM_resnetv2_101x3',
    'BiTM_resnetv2_101x1', 'BiTM_resnetv2_50x3','BiTM_resnetv2_50x1',


    #* ViT
    'vit_small_patch16_224', 'vit_base_patch16_224', 'vit_large_patch16_224',
    # 'ResNeXt101_32x16d_swsl', 
    'transformer_L16_IN21K', 'transformer_B16_IN21K', 

    #* SSL
    "simclr_resnet50x1",
    'MoCo',
    'PIRL',
    'MoCoV2',
    'InfoMin',
    'InsDis',
    
    #* SIN
    "resnet50_trained_on_SIN",
    "resnet50_trained_on_SIN_and_IN",

    #* multimodal/VLMs
    'clipRN50',
    'clip',
    'Llama-4-Scout-17B-16E-Instruct',
    'gemini-2.0-flash',
    'gpt-4o',

]
    
# ViTs (assigned middle gray)
vit_models = {"DINOv2"}

# Add entries to FILES and COLOR_MAP
save_dir = "/home/student/l/lzejin/codebase/P001_evd_gpus/results/illusion_benchmark/raw_data"
for model_name in model_names:
    FILES[model_name] = f"{save_dir}/results_{model_name}.csv"
    COLOR_MAP[model_name] = middle_gray if model_name in vit_models else high_gray
    if model_name == "resnet50":
        FILES[ "baseline"] =  "./results/illusion_benchmark/resnet50_baseline_imagenet.csv"

# our models
our_models_dict= {
#  "baseline": "./results/illusion_benchmark/resnet50_baseline_imagenet.csv",
    "DVD_P": "./results/illusion_benchmark/resnet50_DVD-P_imagenet.csv",
    "DVD_B": "./results/illusion_benchmark/resnet50_DVD-B_imagenet.csv",
    "DVD_S": "./results/illusion_benchmark/resnet50_DVD-S_imagenet.csv",
}
FILES.update(our_models_dict)

# ────────────────────────────────  Main  ──────────────────────────────────── #

def main() -> None:
    recall_rows: list[dict[str, float | str]] = []

    for model, csv_path in FILES.items():
        if not os.path.isfile(csv_path):
            print(f"[warning] CSV not found for {model!r}: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        shape_rec, scene_rec = compute_recall(df)
        recall_rows.append(
            {"model": model, "shape_recall": shape_rec, "scene_recall": scene_rec}
        )
        print(f"model: {model}, shape_recall: {shape_rec}, scene_recall: {scene_rec}")

        #* calculate per shape recall if needed
        # shape_accuracy = compute_shape_accuracy(df)
        # print(f"Shape accuracy for {model}:\n{shape_accuracy}")


    if not recall_rows:
        raise SystemExit("No CSVs loaded, nothing to plot.")

    recall_df = (
        pd.DataFrame(recall_rows)
        .set_index("model")
        .loc[FILES.keys()]  # maintain order
        .reset_index()
    )

    # ───────────────  Plot  ──────────────── #
    fig, ax = plt.subplots(figsize=(3.54*2, 4))
    x = range(len(recall_df))
    bar_w = 0.5

    ax.bar(
        x,
        recall_df["shape_recall"],
        width=bar_w,
        color=[COLOR_MAP[m] for m in recall_df["model"]],
        label="Shape",
    )
    ax.bar(
        x,
        -recall_df["scene_recall"],
        width=bar_w,
        color="grey",
        label="Scene",
    )

    # === X-axis ticks & labels (6 pt) ===
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        recall_df["model"].str.replace("_", " "),  # friendlier names
        rotation=90,
        fontsize=6,
    )

    # === Baseline and limits ===
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylim([-80, 50])        # positive up to +50, negative to –100
    # ax.set_ylabel("Recall (%)")    # generic y-axis label
    # **NEW: show absolute numbers on the y-axis**
    ax.yaxis.set_major_formatter(abs_fmt)

    # === Custom y-axis annotations ===
    ax.text(
        -5.6,  20, "shape recall (%)",
        fontsize=6, va="center", ha="left", rotation=90
    )
    ax.text(
        -5.6, -50, "scene recall (%)",
        fontsize=6, va="center", ha="left", rotation=90
    )

    # === Title (7 pt) ===
    ax.set_title(
        "Shape vs Scene Recall on IllusionBench-IN",
        fontsize=7
    )

    # === Optional legend ===
    # ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    # === Layout & save ===
    plt.tight_layout()
    out_dir = Path("./results/plots/illusion_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "illusion_benchmark_shape_scene_recall_v2_3.pdf"
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    print("[saved]", out_path)

if __name__ == "__main__":
    main()
