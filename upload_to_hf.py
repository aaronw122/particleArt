"""
Upload trained LoRA + TI weights to HuggingFace.

Run:
    uv run python upload_to_hf.py
"""

from io import BytesIO
from huggingface_hub import HfApi, create_repo
from pathlib import Path

REPO_ID = "aaronw122/prtkl-sdxl-lora"
WEIGHTS_DIR = Path("final_weights")
LORA_PATH = WEIGHTS_DIR / "lora" / "checkpoint-1800" / "pytorch_lora_weights.safetensors"
TI_PATH = WEIGHTS_DIR / "ti" / "results_emb.safetensors"

MODEL_CARD = """\
---
library_name: diffusers
license: apache-2.0
base_model: stabilityai/stable-diffusion-xl-base-1.0
tags:
  - stable-diffusion-xl
  - lora
  - textual-inversion
  - particle-art
  - sdxl
  - text-to-image
pipeline_tag: text-to-image
---

# prtkl — Particle Art SDXL LoRA

A fine-tuned SDXL LoRA that generates **particle art** — human figures composed of sparse black dots on white backgrounds.

## Usage

```python
import torch
from diffusers import AutoencoderKL, DiffusionPipeline
from safetensors.torch import load_file

# Load pipeline
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

# Load LoRA weights
pipe.load_lora_weights("aaronw122/prtkl-sdxl-lora", weight_name="pytorch_lora_weights.safetensors")

# Load textual inversion embeddings
from huggingface_hub import hf_hub_download
ti_path = hf_hub_download("aaronw122/prtkl-sdxl-lora", "results_emb.safetensors")
state_dict = load_file(ti_path)
pipe.load_textual_inversion(state_dict["clip_l"], token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
pipe.load_textual_inversion(state_dict["clip_g"], token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)

# Generate
image = pipe(
    "<s0><s1>, a figure dancing with arms raised, white background",
    negative_prompt="photorealistic, detailed, shading, gradient, gray, color, dense",
    num_inference_steps=30,
    guidance_scale=5.0,
).images[0]
image.save("particle_art.png")
```

## Training Details

- **Base model:** SDXL 1.0
- **Method:** Pivotal Tuning (LoRA + Textual Inversion)
- **Optimizer:** AdamW 8-bit
- **Checkpoint:** Step 1800
- **Rank:** 32
- **Trigger tokens:** `<s0><s1>` (mapped from `prtkl`)
- **TI frozen after:** Step 500

## Prompt Tips

- Always prefix prompts with `<s0><s1>,`
- Add "white background" for cleanest results
- Use negative prompt: `"photorealistic, detailed, shading, gradient, gray, color, dense, beige, tan, sepia"`
- Guidance scale 5.0 works well
"""


def main():
    api = HfApi()

    # Create repo (no-op if exists)
    create_repo(REPO_ID, exist_ok=True, repo_type="model")
    print(f"Repo: https://huggingface.co/{REPO_ID}")

    # Upload LoRA weights
    print(f"Uploading LoRA weights: {LORA_PATH}")
    api.upload_file(
        path_or_fileobj=str(LORA_PATH),
        path_in_repo="pytorch_lora_weights.safetensors",
        repo_id=REPO_ID,
    )

    # Upload TI embeddings
    print(f"Uploading TI embeddings: {TI_PATH}")
    api.upload_file(
        path_or_fileobj=str(TI_PATH),
        path_in_repo="results_emb.safetensors",
        repo_id=REPO_ID,
    )

    # Upload model card
    print("Uploading model card")
    api.upload_file(
        path_or_fileobj=BytesIO(MODEL_CARD.encode()),
        path_in_repo="README.md",
        repo_id=REPO_ID,
    )

    print(f"\nDone! View at: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
