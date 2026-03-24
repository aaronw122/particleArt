"""
Generate images from multiple checkpoints to compare quality at different training steps.
Tests both Prodigy and AdamW volumes.

Run:
    uv run modal run generate_modal_sweep.py
"""

import modal

app = modal.App("sdxl-lora-sweep")

adamw_vol = modal.Volume.from_name("lora-output-adamw-v4")

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

NEGATIVE_PROMPT = "photorealistic, detailed, shading, gradient, gray, color, dense, beige, tan, sepia, parchment, warm tones, blue, colored dots, 3D, lighting"

PROMPTS = [
    "<s0><s1>, a figure standing with arms crossed over chest, white background",
    "<s0><s1>, a figure leaning forward with arms behind, white background",
    "<s0><s1>, a figure lunging forward, white background",
    "<s0><s1>, a figure standing with hands on hips, white background",
    "<s0><s1>, a figure stepping off a ledge, white background",
    "<s0><s1>, a figure with arms raised in a V shape, white background",
    "<s0><s1>, a figure kicking one leg forward, white background",
    "<s0><s1>, a figure arching backward, white background",
    "<s0><s1>, a figure with one knee up like climbing stairs, white background",
    "<s0><s1>, a figure shielding eyes with one hand looking into distance, white background",
    "<s0><s1>, a figure with arms stretched to one side leaning away, white background",
    "<s0><s1>, a figure mid-jump with legs tucked, white background",
    "<s0><s1>, a figure standing on tiptoes reaching upward, white background",
]

GUIDANCE_SCALES = [5.0]

CHECKPOINTS = [1700]


@app.function(
    gpu="A10G",
    image=inference_image,
    volumes={"/adamw": adamw_vol},
    timeout=1800,
)
def generate():
    import os
    import torch
    from diffusers import AutoencoderKL, DiffusionPipeline
    from safetensors.torch import load_file

    # Load base model once
    vae = AutoencoderKL.from_pretrained(
        "madebyollin/sdxl-vae-fp16-fix",
        torch_dtype=torch.float16,
    )

    for vol_name, vol_path, vol in [("adamw", "/adamw", adamw_vol)]:
        vol.reload()

        # TI embeddings frozen after step 500 — saved in results root from original run
        ti_path = f"{vol_path}/results/results_emb.safetensors"

        for step in CHECKPOINTS:
            print(f"\n=== {vol_name} / checkpoint-{step} ===")

            # Fresh pipeline for each checkpoint to avoid weight contamination
            pipe = DiffusionPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                vae=vae,
                torch_dtype=torch.float16,
                variant="fp16",
            ).to("cuda")

            # Load LoRA from checkpoint (or final weights for step 1900)
            if step == 1900:
                lora_path = f"{vol_path}/results"
            else:
                lora_path = f"{vol_path}/results/checkpoint-{step}"
            pipe.load_lora_weights(lora_path, weight_name="pytorch_lora_weights.safetensors")

            # Load TI embeddings
            state_dict = load_file(ti_path)
            pipe.load_textual_inversion(state_dict["clip_l"], token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder, tokenizer=pipe.tokenizer)
            pipe.load_textual_inversion(state_dict["clip_g"], token=["<s0>", "<s1>"], text_encoder=pipe.text_encoder_2, tokenizer=pipe.tokenizer_2)

            out_dir = f"{vol_path}/sweep/{vol_name}_step{step}"
            os.makedirs(out_dir, exist_ok=True)

            img_idx = 0
            for guidance in GUIDANCE_SCALES:
                for i, prompt in enumerate(PROMPTS):
                    print(f"  [cfg={guidance}] [{i+1}/{len(PROMPTS)}] {prompt[:60]}...")
                    image = pipe(
                        prompt,
                        negative_prompt=NEGATIVE_PROMPT,
                        num_inference_steps=30,
                        guidance_scale=guidance,
                        generator=torch.Generator("cuda").manual_seed(42 + i),
                    ).images[0]
                    path = f"{out_dir}/cfg{guidance}_{i:02d}.png"
                    image.save(path)
                    img_idx += 1

            # Free VRAM
            del pipe
            torch.cuda.empty_cache()

        vol.commit()

    print("\n=== Sweep complete! ===")
    print("Download with:")
    print("  uv run modal volume get lora-output /sweep/ lora_output_prodigy/sweep/")
    print("  uv run modal volume get lora-output-adamw /sweep/ lora_output_adamw/sweep/")
