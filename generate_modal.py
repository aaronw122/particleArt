"""
Generate images using the trained LoRA on Modal.

Run:
    uv run modal run generate_modal.py
"""

import modal

app = modal.App("sdxl-lora-generate")

output_vol = modal.Volume.from_name("lora-output")

inference_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch",
        "diffusers==0.31.0",
        "transformers==4.44.2",
        "accelerate",
        "safetensors",
        "peft",
        "Pillow",
    ])
)

NEGATIVE_PROMPT = "photorealistic, detailed, shading, gradient, gray, color, dense"

PROMPTS = [
    "<s0><s1>, a figure standing with arms at sides",
    "<s0><s1>, a figure mid-leap against a white background",
    "<s0><s1>, two figures intertwined in a dance",
    "<s0><s1>, a figure dissolving into particles",
    "<s0><s1>, a figure sitting cross-legged, meditating",
]


@app.function(
    gpu="A10G",
    image=inference_image,
    volumes={"/output": output_vol},
    timeout=600,
)
def generate():
    import os
    import torch
    from diffusers import AutoencoderKL, DiffusionPipeline
    from safetensors.torch import load_file

    out_dir = "/output/generated_v3"
    os.makedirs(out_dir, exist_ok=True)

    # Use fp16-safe VAE to avoid black images
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16,
    )

    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        vae=vae,
        torch_dtype=torch.float16,
        variant="fp16",
    ).to("cuda")

    # Reload volume to get latest weights from training
    output_vol.reload()

    # Load LoRA weights
    pipe.load_lora_weights("/output/results", weight_name="pytorch_lora_weights.safetensors")

    # Load TI embeddings (pivotal tuning creates these)
    state_dict = load_file("/output/results/results_emb.safetensors")
    pipe.load_textual_inversion(state_dict["clip_l"], token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
    pipe.load_textual_inversion(state_dict["clip_g"], token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)

    for i, prompt in enumerate(PROMPTS):
        print(f"[{i+1}/{len(PROMPTS)}] {prompt}")
        image = pipe(
            prompt,
            negative_prompt=NEGATIVE_PROMPT,
            num_inference_steps=30,
            guidance_scale=5.0,
            generator=torch.Generator("cuda").manual_seed(42 + i),
        ).images[0]
        path = f"{out_dir}/{i:02d}.png"
        image.save(path)
        print(f"  Saved: {path}")

    output_vol.commit()
    print(f"\nDone! Download with: uv run modal volume get lora-output /generated_v3/ lora_output/generated_v3/")
