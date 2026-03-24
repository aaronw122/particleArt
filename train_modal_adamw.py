"""
Train SDXL LoRA on Modal with A10G GPU (24GB VRAM).
AdamW variant with separate LRs for TI and UNet LoRA.

Run:
    uv run modal run train_modal_adamw.py
"""

import modal

app = modal.App("sdxl-lora-training-adamw")

# Separate volume to avoid conflicts with the Prodigy run
output_vol = modal.Volume.from_name("lora-output-adamw-v4", create_if_missing=True)

training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("wget")
    .pip_install([
        "torch",
        "torchvision",
        "diffusers==0.31.0",
        "transformers==4.44.2",
        "accelerate",
        "bitsandbytes",
        "safetensors",
        "peft",
        "xformers",
        "datasets",
        "Pillow",
        "tensorboard",
    ])
    .add_local_dir("images/curated", "/root/training_data", copy=True)
)

TRIGGER = "TOK"


def build_dataset():
    """Build HuggingFace ImageFolder dataset with metadata.csv."""
    import csv
    import os
    import shutil

    src_dir = "/root/training_data"
    dataset_dir = "/root/dataset/train"
    os.makedirs(dataset_dir, exist_ok=True)

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

    with open(os.path.join(dataset_dir, "metadata.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["file_name", "text"])
        for fname, caption in rows:
            writer.writerow([fname, caption])

    print(f"Dataset ready: {len(rows)} images with captions")
    for fname, caption in rows[:3]:
        print(f"  {fname}: {caption}")
    return len(rows)


@app.function(
    gpu="A10G",
    image=training_image,
    volumes={"/output": output_vol},
    timeout=28800,
)
def train():
    import subprocess

    # Build dataset
    num_images = build_dataset()
    print(f"\n=== Training on {num_images} images (AdamW) ===\n")

    # Download training script
    subprocess.run([
        "wget", "-q",
        "https://raw.githubusercontent.com/huggingface/diffusers/v0.31.0/examples/advanced_diffusion_training/train_dreambooth_lora_sdxl_advanced.py",
        "-O", "/root/train_dreambooth_lora_sdxl_advanced.py",
    ], check=True)

    # Train
    result = subprocess.run([
        "accelerate", "launch", "/root/train_dreambooth_lora_sdxl_advanced.py",
        "--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0",
        "--pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix",
        "--instance_prompt=TOK",
        "--dataset_name=/root/dataset",
        "--caption_column=text",
        "--resolution=1024",
        "--train_batch_size=1",
        "--gradient_accumulation_steps=4",
        "--gradient_checkpointing",
        "--mixed_precision=bf16",
        "--use_8bit_adam",
        "--learning_rate=9e-5",
        "--text_encoder_lr=2.5e-4",
        "--train_text_encoder_ti",
        "--train_text_encoder_ti_frac=0.5",
        "--token_abstraction=TOK",
        "--num_new_tokens_per_abstraction=2",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=100",
        "--max_train_steps=1900",
        "--noise_offset=0.0357",
        "--rank=32",
        "--output_dir=/output/results",
        "--checkpointing_steps=100",
        "--checkpoints_total_limit=25",
        "--enable_xformers_memory_efficient_attention",
        "--snr_gamma=5.0",
        "--validation_prompt=TOK, a figure standing with arms at sides",
        "--num_validation_images=1",
        "--validation_epochs=6",
        "--seed=42",
        "--logging_dir=/output/logs",
        "--report_to=tensorboard",
    ])

    # Always commit the volume — checkpoints may exist even if the script crashed
    output_vol.commit()

    if result.returncode != 0:
        print(f"\n⚠ Training script exited with code {result.returncode}")
        print("Checkpoints were still saved to the volume.")
    else:
        print("\n=== Training complete! ===")
    print("Download results with: modal volume get lora-output-adamw /results/ ./lora_output_adamw/")
