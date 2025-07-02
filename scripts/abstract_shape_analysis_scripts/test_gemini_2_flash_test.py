"""
illusionbench_gemini.py

IllusionBench evaluation with **Gemini Vision**, saving one CSV row per call **and**
resizing every input image to **224 × 224** pixels before sending it to the API.

Quick start
-----------
1. ``pip install google-generativeai datasets pillow tqdm``
2. ``export GOOGLE_API_KEY="<your-api-key>"``
3. *(optional)* ``export GEMINI_MODEL="gemini-2.0-flash"`` – any Vision-capable Gemini variant.
4. ``python illusionbench_gemini.py``

CSV output streams continuously to:
``./results/illusion_benchmark/raw_data/results_<MODEL>.csv``
so progress is never lost.
"""
from __future__ import annotations

import base64
import csv
import io
import os
import time
from typing import List

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

# Google Generative AI SDK
from google import genai
from google.genai import types  # for Part.from_bytes

# === Configuration ==========================================================
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")  # Vision-capable Gemini model
TARGET_SIZE = (224, 224)                                     # resize for each sample
SLEEP_BETWEEN_CALLS = float(os.getenv("GEMINI_RATE_DELAY", "0"))  # seconds
IMAGENET_LABELS_PATH = "./data/imagenet_classes.txt"                 # 1 000 labels
SAVE_DIR = "./results/illusion_benchmark/raw_data/"

# === Utility functions ======================================================

def encode_pil_image(
    img: Image.Image, *, fmt: str = "JPEG", quality: int = 90
) -> bytes:
    """Resize *img* to TARGET_SIZE, convert to RGB, JPEG-encode, return raw bytes."""
    img = img.resize(TARGET_SIZE, Image.BILINEAR)  # 224×224 in-memory
    with io.BytesIO() as buf:
        img.convert("RGB").save(buf, format=fmt, quality=quality)
        return buf.getvalue()


def parse_shape_scene(name: str):
    """Infer `shape` and `scene` tags from an IllusionBench filename."""
    try:
        shape, scene, *_ = name.split("-")
    except ValueError:
        shape, scene = "", ""
    return shape.lower(), scene.lower()


# === Main evaluation loop ====================================================

def main() -> None:  # pragma: no cover
    # Create Gemini client (uses GOOGLE_API_KEY env var)
    client = genai.Client()

    # Prepare CSV (append mode, header only once)
    os.makedirs(SAVE_DIR, exist_ok=True)
    csv_path = os.path.join(SAVE_DIR, f"results_{MODEL_NAME}.csv")
    csv_file = open(csv_path, "a", newline="", buffering=1)  # line-buffered
    writer = csv.DictWriter(
        csv_file, fieldnames=["image_name", "predicted_label", "shape", "scene"]
    )
    if csv_file.tell() == 0:
        writer.writeheader()

    # Load dataset & ImageNet class list
    dataset = load_dataset("arshiahemmat/IllusionBench", split="Illusion_IN")
    with open(IMAGENET_LABELS_PATH) as f:
        imagenet_classes: List[str] = [line.strip() for line in f]

    prompt_template = (
        "You are an ImageNet classifier. Choose exactly one class label from the list provided by the user."
        "Respond with *only* that class name—no quotes, no punctuation, no additional "
        "words. If uncertain, guess the most likely class. Never apologise or mention "
        "uncertainty. Classify this image among the 1000 ImageNet classes:\n{classes}\n"
    )

    # Running tallies
    shape_correct = scene_correct = total = 0

    for item in tqdm(dataset, desc=f"Evaluating with {MODEL_NAME}"):
        # --- Get PIL image ---------------------------------------------------
        img_data = item["image"]
        if isinstance(img_data, dict):
            if "array" in img_data:
                pil_img = Image.fromarray(img_data["array"])
            elif "bytes" in img_data:
                pil_img = Image.open(io.BytesIO(img_data["bytes"]))
            else:
                continue  # malformed sample
        else:
            pil_img = img_data  # already PIL.Image

        # Build multimodal prompt
        prompt = prompt_template.format(classes="\n".join(imagenet_classes))
        img_bytes = encode_pil_image(pil_img)  # resized + encoded

        # Gemini multimodal request
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                prompt,
            ],
        )

        predicted = response.text.strip()
        shape, scene = parse_shape_scene(item["image_name"])

        # Update metrics
        label_lower = predicted.lower()
        if shape and shape in label_lower:
            shape_correct += 1
        elif scene and scene in label_lower:
            scene_correct += 1
        total += 1

        # Write result immediately
        writer.writerow(
            {
                "image_name": item["image_name"],
                "predicted_label": predicted,
                "shape": shape,
                "scene": scene,
            }
        )
        csv_file.flush()

        if SLEEP_BETWEEN_CALLS:
            time.sleep(SLEEP_BETWEEN_CALLS)

    csv_file.close()

    # Print short summary
    if total:
        print(f"Results saved to {csv_path}")
        print(f"Total images processed: {total}")
        print(f"Shape-based predictions: {shape_correct} ({shape_correct/total:.2%})")
        print(f"Scene-based predictions: {scene_correct} ({scene_correct/total:.2%})")


if __name__ == "__main__":
    main()
