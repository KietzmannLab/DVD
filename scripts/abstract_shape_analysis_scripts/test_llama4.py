from transformers import AutoProcessor, Llama4ForConditionalGeneration
import torch

model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

processor = AutoProcessor.from_pretrained(model_id)
model = Llama4ForConditionalGeneration.from_pretrained(
    model_id,
    attn_implementation="flex_attention",
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

url1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
url2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png"
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": url1},
            {"type": "image", "url": url2},
            {"type": "text", "text": "Can you describe how these two images are similar, and how they differ?"},
        ]
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=256,
)

response = processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
print(response)
print(outputs[0])

# from transformers import AutoProcessor, Llama4ForConditionalGeneration
# import torch

# model_id = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

# processor = AutoProcessor.from_pretrained(model_id)
# model = Llama4ForConditionalGeneration.from_pretrained(
#     model_id,
#     attn_implementation="flex_attention",
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
# )

# url1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
# url2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png"
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "url": url1},
#             {"type": "image", "url": url2},
#             {"type": "text", "text": "Can you describe how these two images are similar, and how they differ?"},
#         ]
#     },
# ]

# inputs = processor.apply_chat_template(
#     messages,
#     add_generation_prompt=True,
#     tokenize=True,
#     return_dict=True,
#     return_tensors="pt",
# ).to(model.device)

# outputs = model.generate(
#     **inputs,
#     max_new_tokens=256,
# )

# response = processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
# print(response)
# print(outputs[0])


# from transformers import AutoProcessor, Llama4ForConditionalGeneration
# import torch

# model_id = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"

# processor = AutoProcessor.from_pretrained(model_id)
# model = Llama4ForConditionalGeneration.from_pretrained(
#     model_id,
#     attn_implementation="flex_attention",
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
# )

# url1 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
# url2 = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/datasets/cat_style_layout.png"
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "url": url1},
#             {"type": "image", "url": url2},
#             {"type": "text", "text": "Can you describe how these two images are similar, and how they differ?"},
#         ]
#     },
# ]

# inputs = processor.apply_chat_template(
#     messages,
#     add_generation_prompt=True,
#     tokenize=True,
#     return_dict=True,
#     return_tensors="pt",
# ).to(model.device)

# outputs = model.generate(
#     **inputs,
#     max_new_tokens=256,
# )

# response = processor.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])[0]
# print(response)
# print(outputs[0])



# """
# illusionbench_llama.py

# IllusionBench evaluation with **Llama-4 Vision** (Hugging Face),
# saving one CSV row per image **and** resizing every input image to
# **224 × 224** pixels before sending it to the model.

# Quick start
# -----------
# 1. ``pip install transformers accelerate datasets pillow tqdm``
# 2. *(optional)* ``export HF_TOKEN=...`` – if the model is gated.
# 3. *(optional)* ``export LLAMA_MODEL=meta-llama/Llama-4-Maverick-17B-128E-Instruct``
# 4. ``python illusionbench_llama.py``

# CSV output streams continuously to:
# ``./results/illusion_benchmark/raw_data/results_<MODEL>.csv`` so progress is
# never lost.
# """

# from __future__ import annotations

# import base64
# import csv
# import io
# import os
# import time
# from typing import List

# import torch
# from datasets import load_dataset
# from PIL import Image
# from tqdm import tqdm
# from transformers import AutoProcessor, Llama4ForConditionalGeneration

# # === Configuration =========================================================
# MODEL_ID = os.getenv(
#     # "LLAMA_MODEL", "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
#     "LLAMA_MODEL", "meta-llama/Llama-4-Scout-17B-16E-Instruct",
#     # "LLAMA_MODEL", "meta-llama/Llama-4-Maverick-17B-128E-Original",
# )  # Vision-capable Llama-4 model on Hugging Face
# TARGET_SIZE: tuple[int, int] = (224, 224)  # resize for each sample
# SLEEP_BETWEEN_CALLS = float(os.getenv("LLAMA_RATE_DELAY", "0"))  # seconds
# IMAGENET_LABELS_PATH = "./data/imagenet_classes.txt"  # 1 000 labels
# SAVE_DIR = "./results/illusion_benchmark/raw_data/"
# MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "32"))

# # === Utility functions =====================================================


# def encode_pil_image(
#     img: Image.Image,
#     *,
#     fmt: str = "JPEG",
#     quality: int = 90,
# ) -> str:
#     """Resize *img* to 224 × 224, convert to RGB, JPEG-encode, return Base-64 str."""
#     img = img.resize(TARGET_SIZE, Image.BILINEAR)  # 224×224 in-memory
#     with io.BytesIO() as buf:
#         img.convert("RGB").save(buf, format=fmt, quality=quality)
#         return base64.b64encode(buf.getvalue()).decode("utf-8")


# def parse_shape_scene(name: str):
#     """Infer ``shape`` and ``scene`` tags from an IllusionBench filename."""
#     try:
#         shape, scene, *_ = name.split("-")
#     except ValueError:
#         shape, scene = "", ""
#     return shape.lower(), scene.lower()


# # === Main evaluation loop ==================================================


# def main() -> None:
#     print(f"Loading model: {MODEL_ID} …")
#     processor = AutoProcessor.from_pretrained(MODEL_ID)
#     model = Llama4ForConditionalGeneration.from_pretrained(
#         MODEL_ID,
#         attn_implementation="flex_attention",
#         device_map="auto",
#         torch_dtype=torch.bfloat16,
#     )

#     # Prepare CSV (append mode, header only once)
#     os.makedirs(SAVE_DIR, exist_ok=True)
#     csv_path = os.path.join(SAVE_DIR, f"results_{MODEL_ID.split('/')[-1]}.csv")
#     csv_file = open(csv_path, "a", newline="", buffering=1)
#     writer = csv.DictWriter(
#         csv_file, fieldnames=["image_name", "predicted_label", "shape", "scene"]
#     )
#     if csv_file.tell() == 0:
#         writer.writeheader()

#     # Load dataset & ImageNet class list
#     dataset = load_dataset("arshiahemmat/IllusionBench", split="Illusion_IN")
#     with open(IMAGENET_LABELS_PATH) as f:
#         imagenet_classes: List[str] = [line.strip() for line in f]

#     prompt_template = (
#         "You are an ImageNet classifier. Choose exactly one class label from the "
#         "list provided by the user (No other class beyond these 1000 ImageNet "
#         "classes.). Respond with *only* that class name—no quotes, no punctuation, "
#         "no additional words. If uncertain, guess the most likely class. Never "
#         "apologise or mention uncertainty. Classify this image among the 1000 "
#         "ImageNet classes: \n{classes}\nAnswer with only the class name."
#     )

#     # Running tallies
#     shape_correct = scene_correct = total = 0

#     for item in tqdm(dataset, desc=f"Evaluating with {MODEL_ID}"):
#         # --- Get PIL image ---------------------------------------------------
#         img_data = item["image"]
#         if isinstance(img_data, dict):
#             if "array" in img_data:
#                 pil_img = Image.fromarray(img_data["array"])
#             elif "bytes" in img_data:
#                 pil_img = Image.open(io.BytesIO(img_data["bytes"]))
#             else:
#                 continue  # malformed sample
#         else:
#             pil_img = img_data  # already PIL.Image

#         # Build multimodal prompt
#         prompt = prompt_template.format(classes="\n".join(imagenet_classes))
#         b64_img = encode_pil_image(pil_img)  # resized + encoded

#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "image",
#                         "url": f"data:image/jpeg;base64,{b64_img}",
#                     },
#                     {"type": "text", "text": prompt},
#                 ],
#             }
#         ]

#         # Tokenise & run inference
#         inputs = processor.apply_chat_template(
#             messages,
#             add_generation_prompt=True,
#             tokenize=True,
#             return_dict=True,
#             return_tensors="pt",
#         ).to(model.device)

#         with torch.inference_mode():
#             generated = model.generate(
#                 **inputs,
#                 max_new_tokens=MAX_NEW_TOKENS,
#             )

#         # Decode assistant reply *after* the user prompt
#         pred = processor.batch_decode(
#             generated[:, inputs["input_ids"].shape[-1] :],
#             skip_special_tokens=True,
#         )[0].strip()

#         shape, scene = parse_shape_scene(item["image_name"])

#         # Update metrics
#         label_lower = pred.lower()
#         if shape and shape in label_lower:
#             shape_correct += 1
#         elif scene and scene in label_lower:
#             scene_correct += 1
#         total += 1

#         # Write result immediately
#         writer.writerow(
#             {
#                 "image_name": item["image_name"],
#                 "predicted_label": pred,
#                 "shape": shape,
#                 "scene": scene,
#             }
#         )
#         csv_file.flush()

#         if SLEEP_BETWEEN_CALLS:
#             time.sleep(SLEEP_BETWEEN_CALLS)

#     csv_file.close()

#     # Print short summary
#     if total:
#         print(f"Results saved to {csv_path}")
#         print(f"Total images processed: {total}")
#         print(f"Shape-based predictions: {shape_correct} ({shape_correct/total:.2%})")
#         print(f"Scene-based predictions: {scene_correct} ({scene_correct/total:.2%})")


# if __name__ == "__main__":
#     main()

