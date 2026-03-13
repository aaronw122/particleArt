# === CELL 1: Install dependencies (pinned diffusers to match training script) ===
!pip install diffusers==0.31.0 transformers==4.44.2 accelerate bitsandbytes safetensors peft xformers datasets

# === CELL 2: Upload training data ===
# Upload your curated/ folder (images + .txt captions) to Google Drive,
# then mount it here. Or upload directly via Colab sidebar.

from google.colab import drive
drive.mount('/content/drive')

# Copy training data from Drive to local (faster I/O)
!cp -r "/content/drive/MyDrive/particleArt/curated" /content/training_data

# Verify
import os
images = [f for f in os.listdir("/content/training_data") if f.endswith(".png")]
captions = [f for f in os.listdir("/content/training_data") if f.endswith(".txt")]
print(f"Images: {len(images)}, Captions: {len(captions)}")
assert len(images) == len(captions), f"Mismatch! {len(images)} images but {len(captions)} captions"

# === CELL 3: Build dataset with metadata.csv ===
# The diffusers script ignores .txt files with --instance_data_dir.
# We need an ImageFolder dataset with metadata.csv for per-image captions.

import csv
import shutil

TRIGGER = "prtkl"
src_dir = "/content/training_data"
dataset_dir = "/content/dataset/train"  # HuggingFace ImageFolder format
os.makedirs(dataset_dir, exist_ok=True)

# For each image, find its .txt caption, prepend trigger word,
# copy image into dataset folder, and collect rows for metadata.csv
rows = []
for img_file in sorted(os.listdir(src_dir)):
    if not img_file.endswith(".png"):
        continue
    txt_file = img_file.replace(".png", ".txt")
    txt_path = os.path.join(src_dir, txt_file)
    if os.path.exists(txt_path):
        with open(txt_path, "r") as f:
            caption = f.read().strip()
    else:
        caption = ""
    if not caption.startswith(TRIGGER):
        caption = f"{TRIGGER}, {caption}"
    shutil.copy2(os.path.join(src_dir, img_file), os.path.join(dataset_dir, img_file))
    rows.append((img_file, caption))

# metadata.csv maps each image filename to its caption
with open(os.path.join(dataset_dir, "metadata.csv"), "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["file_name", "text"])
    for fname, caption in rows:
        writer.writerow([fname, caption])

print(f"\nDataset ready: {len(rows)} images with captions")
print("\nSample captions:")
for fname, caption in rows[:3]:
    print(f"  {fname}: {caption}")

# === CELL 3b: Verify dataset loads correctly ===
from datasets import load_dataset

ds = load_dataset("imagefolder", data_dir="/content/dataset", split="train")
print(f"\nDataset loaded: {len(ds)} examples")
print(f"Columns: {ds.column_names}")
print(f"\nFirst example caption: {ds[0]['text']}")
print(f"First example image size: {ds[0]['image'].size}")
assert "text" in ds.column_names, "Caption column 'text' not found!"
assert len(ds) == len(rows), f"Dataset size mismatch: {len(ds)} vs {len(rows)}"
# Verify trigger word is in every caption
for i, example in enumerate(ds):
    assert TRIGGER in example["text"], f"Row {i} missing trigger word: {example['text']}"
print("\nAll checks passed!")

# === CELL 4: Download training script (pinned to v0.31.0) ===
!wget -q https://raw.githubusercontent.com/huggingface/diffusers/v0.31.0/examples/advanced_diffusion_training/train_dreambooth_lora_sdxl_advanced.py

# === CELL 5: Train ===
!accelerate launch train_dreambooth_lora_sdxl_advanced.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
  --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
  --instance_prompt="prtkl" \
  --dataset_name="/content/dataset" \
  --caption_column="text" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=8 \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --use_8bit_adam \
  --learning_rate=1e-5 \
  --lr_scheduler="cosine" \
  --lr_warmup_steps=30 \
  --max_train_steps=300 \
  --rank=16 \
  --output_dir="/content/lora_output" \
  --checkpointing_steps=100 \
  --checkpoints_total_limit=3 \
  --enable_xformers_memory_efficient_attention \
  --snr_gamma=5.0 \
  --seed=42

# === CELL 5b: Check VRAM usage (run while training) ===
!nvidia-smi

# === CELL 6: Inspect training output ===
!ls -la /content/lora_output/

# === CELL 7: Test the LoRA ===
import torch
from diffusers import StableDiffusionXLPipeline, AutoencoderKL
from IPython.display import display

vae = AutoencoderKL.from_pretrained(
    "madebyollin/sdxl-vae-fp16-fix",
    torch_dtype=torch.float16,
)

pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
)
pipe.to("cuda")

# Load LoRA weights
pipe.load_lora_weights("/content/lora_output")

test_prompts = [
    "prtkl, a figure leaping through the air with joy",
    "prtkl, two figures holding hands",
    "prtkl, a figure sitting alone on a bench",
    "prtkl, a figure throwing a ball",
    "prtkl, three figures standing in a circle",
]

for prompt in test_prompts:
    print(f"\n{prompt}")
    image = pipe(
        prompt=prompt,
        negative_prompt="photorealistic, detailed, shading, gradient, gray, color, dense",
        num_inference_steps=30,
        guidance_scale=5.5,
        width=1024,
        height=1024,
    ).images[0]
    display(image)
    safe_name = prompt.replace(" ", "_").replace(",", "").replace("<", "").replace(">", "")[:60]
    image.save(f"/content/lora_output/{safe_name}.png")

# === CELL 8: Save LoRA to Google Drive ===
!cp -r /content/lora_output "/content/drive/MyDrive/particleArt/lora_output"
print("Saved to Google Drive!")
